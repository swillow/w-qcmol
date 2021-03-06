#include <cmath>
#include <algorithm>
#include <armadillo>
#include <thread>
#include "RHF.hpp"
#include "RHFGrad.hpp"

using namespace std;
using namespace libint2;



namespace willow {  namespace qcmol {


static unsigned int num_threads;

static arma::vec compute_nuclear_gradient (const vector<Atom>& atoms,
					   const vector<QAtom>& atoms_Q,
					   arma::vec& grd_Q);


void compute_2body_deriv_ints (const int thread_id,
			       const BasisSet& obs,
			       const vector<Atom>& atoms,
			       const arma::mat& Dm,
			       const arma::mat& Schwartz,
			       arma::vec& result);


template <libint2::Operator obtype>
void compute_1body_deriv_ints (const int thread_id,
			       const BasisSet& obs,
			       const vector<Atom>& atoms,
			       const arma::mat& Dm,
			       arma::vec& result);

void compute_1body_deriv_nuclear (const int thread_id,
				  const BasisSet& obs,
				  const vector<Atom>& atoms,
				  const arma::mat& Dm,
				  const vector<QAtom>& atoms_Q,
				  arma::vec& grd,
				  arma::vec& grd_Q);

//-----
//
//----
RHFGrad::RHFGrad (const vector<Atom>& atoms,
		  const BasisSet& bs,
		  const Integrals& ints,
		  const int qm_chg,
		  const bool l_print,
		  const vector<QAtom>& atoms_Q )
  : RHF (atoms, bs, ints, qm_chg, l_print, atoms_Q)
{

  num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 1;
  
  arma::mat D = densityMatrix ();
  arma::mat W = energyWeightDensityMatrix ();
  
  const auto natom = atoms.size();
  const auto nbf   = bs.nbf();

  vector<arma::vec> grd_kin_t (num_threads);
  vector<arma::vec> grd_nuc_t (num_threads);
  vector<arma::vec> grd_nuc_Q_t(num_threads);
  vector<arma::vec> frc_orth_t(num_threads);

  vector<std::thread> t_grd_kin (num_threads);
  vector<std::thread> t_grd_nuc (num_threads);
  vector<std::thread> t_frc_orth(num_threads);

  vector<QAtom> bq_Q;
  
  for (int id = 0; id < num_threads; ++id) {
    // Kinetic 
    t_grd_kin[id] =
      std::thread (compute_1body_deriv_ints<libint2::Operator::kinetic>,
		   id,
		   std::cref(bs),
		   std::cref(atoms),
		   std::cref(D),
		   std::ref(grd_kin_t[id]) );
    // Nuclear Der 
    t_grd_nuc[id] =
      std::thread (compute_1body_deriv_nuclear,
		   id,
		   std::cref(bs),
		   std::cref(atoms),
		   std::cref(D),
		   std::cref(atoms_Q), 
		   std::ref(grd_nuc_t[id]),
		   std::ref(grd_nuc_Q_t[id]) );
    
    // Overlap
    t_frc_orth[id] =
      std::thread (compute_1body_deriv_ints<libint2::Operator::overlap>,
		   id,
		   std::cref(bs),
		   std::cref(atoms),
		   std::cref(W),
		   std::ref(frc_orth_t[id]) );
  }


  // Join
  for (auto id = 0; id < num_threads; ++id) {
    t_grd_kin[id].join();
    t_grd_nuc[id].join();
    t_frc_orth[id].join();
  }

  for (auto id = 1; id < num_threads; ++id) {
    grd_kin_t[0]  += grd_kin_t[id];
    grd_nuc_t[0]  += grd_nuc_t[id];
    grd_nuc_Q_t[0] += grd_nuc_Q_t[id];
    frc_orth_t[0] += frc_orth_t[id];
  }
  
  // Nuclear Repulsion
  arma::vec grd_rep_Q (3*atoms_Q.size(), arma::fill::zeros);
  arma::vec grd_rep  = compute_nuclear_gradient (atoms, atoms_Q, grd_rep_Q);

  // Two electron Integral
  vector<arma::vec>   grd_eri_t (num_threads);
  vector<std::thread> t_grd_eri (num_threads);

  for (auto id = 0; id < num_threads; ++id) {
    t_grd_eri[id] =
      std::thread (compute_2body_deriv_ints,
		   id,
		   std::cref(bs),
		   std::cref(atoms),
		   std::cref(D),
		   std::cref(ints.Km),
		   std::ref(grd_eri_t[id]) );
  }

  for (auto id = 0; id < num_threads; ++id)
    t_grd_eri[id].join();


  for (auto id = 1; id < num_threads; ++id)
    grd_eri_t[0] += grd_eri_t[id];

  m_grad = grd_kin_t[0] + grd_nuc_t[0] - frc_orth_t[0] + grd_rep + grd_eri_t[0];
  m_grad_Q = grd_nuc_Q_t[0] + grd_rep_Q;
  
  if (l_print) {
    cout << "Total Gradient (QM): " << endl;
  
    for (auto i = 0; i < natom; ++i)
      cout << (i + 1)
	   << "  " << m_grad(3*i)
	   << "  " << m_grad(3*i+1) 
	   << "  " << m_grad(3*i+2) << endl;

    auto natom_Q = atoms_Q.size();

    if (natom_Q > 0) {
      cout << "Total Gradient (BQ): " << endl;
      for (auto i = 0; i < natom_Q; ++i)
	cout << (i + 1)
	     << "  " << m_grad_Q(3*i)
	     << "  " << m_grad_Q(3*i+1) 
	     << "  " << m_grad_Q(3*i+2) << endl;
    }
    
  }
}




template<libint2::Operator obtype>
void compute_1body_deriv_ints (const int thread_id,
			       const BasisSet& bs,
			       const vector<Atom>& atoms,
			       const arma::mat& Dm,
			       arma::vec& grd)
{
  const auto nshl = bs.size();
  const auto nbf  = bs.nbf();
  const auto natom = atoms.size();

  const unsigned deriv_order = 1;
  
  constexpr auto nopers = libint2::operator_traits<obtype>::nopers;

  const auto nresults = nopers * libint2::num_geometrical_derivatives (natom, deriv_order);

  grd = arma::vec(nresults, arma::fill::zeros);

  libint2::Engine engine (obtype,
			  bs.max_nprim(),
			  bs.max_l(),
			  deriv_order);

  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();
  auto shell2atom = bs.shell2atom(atoms);

  for (auto s1 = 0, s12 = 0; s1 != nshl; ++s1){
    auto bf1  = shell2bf[s1];
    auto nbf1 = bs[s1].size();
    auto at1  = shell2atom[s1];
    
    assert (at1 != -1);
    
    for (auto s2 = 0; s2 <= s1; ++s2, ++s12){

      if (s12 % num_threads != thread_id) continue;
      
      auto bf2  = shell2bf[s2];
      auto nbf2 = bs[s2].size();
      auto at2  = shell2atom[s2];
      
      auto nbf12 = nbf1*nbf2;
      
      engine.compute (bs[s1], bs[s2]);
      
      assert(deriv_order == 1);
      
      // 1. Process derivatives with respect to the Gaussian origins first
      //
      for (unsigned int d = 0; d != 6; ++d) { // 2 centers x 3 axes
	auto iat = d < 3 ? at1 : at2;
	auto op_start = (3*iat + d%3)*nopers;
	auto op_fence = op_start + nopers;
	const auto* buf_idx = buf[d];
	if (buf_idx == nullptr) continue;

	for (unsigned int op = op_start; op != op_fence; ++op) {

	  for (auto ib = 0, ij = 0; ib < nbf1; ib++)
	    for (auto jb = 0; jb < nbf2; jb++, ij++) {
	      const auto val = buf_idx[ij];
		
	      grd(op) += Dm(bf1+ib,bf2+jb)*val;
	      
	      if (s1 != s2) {
		grd(op) += Dm(bf2+jb,bf1+ib)*val;
	      }
		
	    }
	} // exit op
      } // exit d
	
      
    }
  }

}



void compute_1body_deriv_nuclear (const int thread_id,
				  const BasisSet& bs,
				  const vector<Atom>& atoms,
				  const arma::mat& Dm,
				  const vector<QAtom>& atoms_Q,
				  arma::vec& grd,
				  arma::vec& grd_Q)
{
  const auto nshl = bs.size();
  const auto nbf  = bs.nbf();
  const auto natom = atoms.size();
  const auto natom_Q = atoms_Q.size();

  const unsigned deriv_order = 1;
  constexpr auto nopers = libint2::operator_traits<Operator::nuclear>::nopers;

  const auto nresults = nopers * libint2::num_geometrical_derivatives (natom, deriv_order);

  grd   = arma::vec(nresults, arma::fill::zeros);
  grd_Q = arma::vec(atoms_Q.size()*3, arma::fill::zeros);
  
  libint2::Engine engine (Operator::nuclear,
			  bs.max_nprim(),
			  bs.max_l(),
			  deriv_order);

  const auto& buf = engine.results ();
  
  // nuclear attraction ints engine
  vector<pair<double, array<double,3>>> q;
  for (const auto& atom : atoms){
    q.push_back( {static_cast<double> (atom.atomic_number), 
	  {{atom.x, atom.y, atom.z}}} );
  }

  for (auto iq = 0; iq < natom_Q; ++iq) {
    q.push_back ( {atoms_Q[iq].charge,
	  {{atoms_Q[iq].x, atoms_Q[iq].y, atoms_Q[iq].z}}} );
  }
  
  engine.set_params(q);
  
  
  auto shell2bf = bs.shell2bf();
  auto shell2atom = bs.shell2atom(atoms);
  
  for (auto s1 = 0, s12 = 0; s1 != nshl; ++s1){
    auto bf1  = shell2bf[s1];
    auto nbf1 = bs[s1].size();
    auto at1  = shell2atom[s1];
    
    assert (at1 != -1);
    
    for (auto s2 = 0; s2 <= s1; ++s2, ++s12){

      if (s12 % num_threads != thread_id) continue;
      
      auto bf2  = shell2bf[s2];
      auto nbf2 = bs[s2].size();
      auto at2  = shell2atom[s2];
      
      auto nbf12 = nbf1*nbf2;
      
      engine.compute (bs[s1], bs[s2]);
      
      assert(deriv_order == 1);
      
      // 1. Process derivatives with respect to the Gaussian origins first
      //
      for (unsigned int d=0; d != 6; ++d){ // 2 centers x 3 axes = 6 cartesian geometric derivatives
	
	auto iat = d < 3 ? at1 : at2;
	auto op_start = (3*iat + d%3)*nopers;
	auto op_fence = op_start + nopers;

	const auto* buf_idx = buf[d];
	if (buf_idx == nullptr) continue;
	
	for (unsigned int op = op_start; op != op_fence; ++op) {
	  
	  for (auto ib = 0, ij = 0; ib < nbf1; ib++)
	    for (auto jb = 0; jb < nbf2; jb++, ij++) {
	      const auto val = buf_idx[ij];

	      grd(op) += Dm(bf1+ib,bf2+jb)*val;
	      
	      if (s1 != s2) {
		grd(op) += Dm(bf2+jb,bf1+ib)*val;
	      }
	      
	    }
	}
	
      } // d
      
      // 2. Process derivatives of nuclear Coulomb operators,
      for (unsigned int iat = 0; iat != natom; ++iat) {
	for (unsigned int ixyz = 0; ixyz != 3; ++ixyz) {
	    
	  const auto* buf_idx = buf[6 + iat*3 + ixyz];
	  
	  auto op_start = (3*iat + ixyz) * nopers;
	  auto op_fence = op_start + nopers;
	    
	  for (unsigned int op = op_start;
	       op != op_fence; ++op) {
	      
	    for (auto ib = 0, ij = 0; ib < nbf1; ib++)
	      for (auto jb = 0; jb < nbf2; jb++, ij++) {
		const double val = buf_idx[ij];
		
		grd(op) += Dm(bf1+ib,bf2+jb)*val;
		if (s1 != s2) 
		  grd(op) += Dm(bf2+jb,bf1+ib)*val;
		
	      }
	  }
	}
      } // iat =

      for (unsigned int iq = 0; iq != natom_Q; ++iq) {
	for (unsigned int ixyz = 0; ixyz != 3; ++ixyz) {
	    
	  const auto* buf_idx = buf[6 + (natom + iq)*3 + ixyz];
	  auto op_start = (3*iq + ixyz) * nopers;
	  auto op_fence = op_start + nopers;
	    
	  for (unsigned int op = op_start;
	       op != op_fence; ++op) {
	      
	    for (auto ib = 0, ij = 0; ib < nbf1; ib++)
	      for (auto jb = 0; jb < nbf2; jb++, ij++) {
		const double val = buf_idx[ij];
		
		grd_Q(op) += Dm(bf1+ib,bf2+jb)*val;
		if (s1 != s2) 
		  grd_Q(op) += Dm(bf2+jb,bf1+ib)*val;
		
	      }
	  }
	}
      } // iQ =

      
    }
  
  }

}



arma::vec compute_nuclear_gradient (const vector<Atom>& atoms,
				    const vector<QAtom>& atoms_Q,
				    arma::vec& grd_Q)
{
  arma::vec grd (3*atoms.size(), arma::fill::zeros);
  if (atoms_Q.size() > 0) 
    grd_Q.zeros();
  
  for (auto i = 0; i < atoms.size(); ++i) {
    double chg_i = (double)atoms[i].atomic_number;
    double xi = atoms[i].x;
    double yi = atoms[i].y;
    double zi = atoms[i].z;

    for (auto j = 0; j < i; ++j) {
      double chg_j = (double)atoms[j].atomic_number;
      // calculate distance
      
      double dx = xi - atoms[j].x;
      double dy = yi - atoms[j].y;
      double dz = zi - atoms[j].z;
  
      double rij2 = dx*dx + dy*dy + dz*dz;
      double rij3 = sqrt(rij2)*rij2;

      arma::vec tmp(3);
      
      tmp(0) = chg_i*chg_j/rij3*dx;
      tmp(1) = chg_i*chg_j/rij3*dy;
      tmp(2) = chg_i*chg_j/rij3*dz;

      grd(3*i  ) -= tmp(0);
      grd(3*i+1) -= tmp(1);
      grd(3*i+2) -= tmp(2);
      grd(3*j  ) += tmp(0);
      grd(3*j+1) += tmp(1);
      grd(3*j+2) += tmp(2);
    }


    for (auto j = 0; j < atoms_Q.size(); ++j) {
      double chg_j = atoms_Q[j].charge;
      // calculate distance
      
      double dx = xi - atoms_Q[j].x;
      double dy = yi - atoms_Q[j].y;
      double dz = zi - atoms_Q[j].z;
  
      double rij2 = dx*dx + dy*dy + dz*dz;
      double rij3 = sqrt(rij2)*rij2;

      arma::vec tmp(3);
      
      tmp(0) = chg_i*chg_j/rij3*dx;
      tmp(1) = chg_i*chg_j/rij3*dy;
      tmp(2) = chg_i*chg_j/rij3*dz;

      grd(3*i  ) -= tmp(0);
      grd(3*i+1) -= tmp(1);
      grd(3*i+2) -= tmp(2);

      grd_Q(3*j  ) += tmp(0);
      grd_Q(3*j+1) += tmp(1);
      grd_Q(3*j+2) += tmp(2);
    }
    
  }

  return grd;

}



// --


void compute_2body_deriv_ints (const int thread_id,
			       const BasisSet& bs,
			       const vector<Atom>& atoms,
			       const arma::mat& Dm,
			       const arma::mat& Schwartz,
			       arma::vec& grd)
{

  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();
  const auto natom   = atoms.size();
  const auto shell2atom = bs.shell2atom(atoms);

  grd =  arma::vec(3*natom, arma::fill::zeros);

  libint2::Engine engine  (libint2::Operator::coulomb,
			   bs.max_nprim(),
			   bs.max_l(),
			   1);
  
  const auto precision = numeric_limits<double>::epsilon();
  engine.set_precision (precision);
  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();

  // loop over permutationally-unique set of shells
  for (auto s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];
    auto nbf1      = bs[s1].size();
    auto iat       = shell2atom[s1];
    
    for (auto s2 = 0; s2 <= s1; ++s2) {
      auto bf2_first = shell2bf[s2];
      auto nbf2      = bs[s2].size();
      auto jat       = shell2atom[s2];
      auto s12_cut   = Schwartz(s1,s2);

      for (auto s3 = 0; s3 <= s1; ++s3) {
	auto bf3_first = shell2bf[s3];
	auto nbf3      = bs[s3].size();
	auto kat       = shell2atom[s3];

	const auto s4_max = (s1 == s3) ? s2 : s3;
	for (auto s4 = 0; s4 <= s4_max; ++s4, ++s1234) {

	  if (s1234 % num_threads != thread_id) continue;
	  
	  auto bf4_first = shell2bf[s4];
	  auto nbf4      = bs[s4].size();
	  auto lat       = shell2atom[s4];
	  auto s34_cut   = Schwartz(s3,s4);
	  
	  if (s12_cut*s34_cut < precision) {
	    continue;
	  }

	  auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
	  auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
	  auto s13_s24_deg = (s1 == s3) ? ((s2 == s4) ? 1.0 : 2.0) : 2.0;
	  auto s1234_deg = s12_deg*s34_deg*s13_s24_deg;
	  
	  engine.compute2<Operator::coulomb,BraKet::xx_xx,1>
	    (bs[s1], bs[s2], bs[s3], bs[s4]);

	  arma::vec tmp(12, arma::fill::zeros);
	  
	  
	  for (auto di = 0; di != 12; di++) {
	    const auto shset = buf[di];

	    if (shset == nullptr) continue;

	    double sum = 0.0;
	    for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
	      const auto bf1 = f1 + bf1_first;
	      
	      for (auto f2 = 0; f2 != nbf2; ++f2) {
		const auto bf2 = f2 + bf2_first;
		
		for (auto f3 = 0; f3 != nbf3; ++f3) {
		  const auto bf3 = f3 + bf3_first;
		  
		  for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		    const auto bf4 = f4 + bf4_first;
		    
		    const auto eri4  = shset[f1234];
		    const auto value = eri4*s1234_deg;
		    
		    sum += 2.0*Dm(bf1,bf2)*Dm(bf3,bf4)*value;
		    sum -= 0.5*Dm(bf1,bf3)*Dm(bf2,bf4)*value;
		    sum -= 0.5*Dm(bf1,bf4)*Dm(bf2,bf3)*value;
		  }
		}
	      }
	    }
	    
	    tmp(di) = 0.25*sum;
	  }// end di
	  
	  // store the gradient

	  grd(3*iat  ) += tmp(0);
	  grd(3*iat+1) += tmp(1);
	  grd(3*iat+2) += tmp(2);
	  grd(3*jat  ) += tmp(3);
	  grd(3*jat+1) += tmp(4);
	  grd(3*jat+2) += tmp(5);
	  grd(3*kat  ) += tmp(6);
	  grd(3*kat+1) += tmp(7);
	  grd(3*kat+2) += tmp(8);
	  grd(3*lat  ) += tmp(9);
	  grd(3*lat+1) += tmp(10);
	  grd(3*lat+2) += tmp(11);

	}
      }
    }
  }

}


} }  // namespace willow::qcmol
