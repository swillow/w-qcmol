#include <thread>
#include "Molecule.hpp"
#include "Integrals.hpp"
#include "RHF.hpp"
#include "diis.hpp"

using namespace std;
using namespace libint2;

namespace willow { namespace qcmol {

static unsigned int num_threads;


void compute_2body_ints_memory (const int thread_id,
				const arma::vec& TEI,
				const arma::mat& Dm,
				arma::mat& Gm)
{
  const auto nbf = Dm.n_rows;

  // return G = J - 0.5K
  arma::mat Jm (nbf,nbf);
  arma::mat Km (nbf,nbf);

  Jm.zeros();
  Km.zeros();

  for (auto i = 0, ijkl = 0; i < nbf; ++i) {
    auto it = i;
    
    for (auto j = 0; j <= it; ++j) {
      auto ij = INDEX(it,j);
      
      for (auto k = 0; k <= it; ++k)
	for (auto l = 0; l <= k; ++l, ++ijkl) {

	  if (ijkl % num_threads != thread_id) continue;
	  
	  auto kl = INDEX(k,l);
	  
	  if (kl > ij) continue;

	  auto ijkl = INDEX(ij,kl);
	  auto eri4 = TEI (ijkl);
	  // (ij|kl)
	  Jm(it,j) += Dm(k,l)*eri4;
	  Km(it,l) += Dm(k,j)*eri4;
	  if (k > l) {
	    // (ij|lk)
	    Jm(it,j) += Dm(l,k)*eri4;
	    Km(it,k) += Dm(l,j)*eri4;
	  }
	  if (it > j) {
	    // (ji|kl)
	    Jm(j,it) += Dm(k,l)*eri4;
	    Km(j,l)  += Dm(k,it)*eri4;
	    if (k > l) {
	      // (ji|lk)
	      Jm(j,it) += Dm(l,k)*eri4;
	      Km(j,k)  += Dm(l,it)*eri4;
	    }
	  }
	  if (ij != kl) {
	    // (kl|ij)
	    Jm(k,l) += Dm(it,j)*eri4;
	    Km(k,j) += Dm(it,l)*eri4;
	    
	    if (it > j) {
	      // (kl|ji)
	      Jm(k,l)  += Dm(j,it)*eri4;
	      Km(k,it) += Dm(j,l)*eri4;
	    }
	    if(k > l) {
		// (lk|ij)
	      Jm(l,k) += Dm(it,j)*eri4;
	      Km(l,j) += Dm(it,k)*eri4;

	      if (it > j) {
		// (lk|ji)
		Jm(l,k)  += Dm(j,it)*eri4;
		Km(l,it) += Dm(j,k)*eri4;
	      }
	    }
	  }
	  
	}
    }

  }


  Gm = (Jm - 0.5*Km);

}




static arma::mat g_matrix (const arma::vec& TEI, arma::mat& Dm)
{

  vector<arma::mat> Gt(num_threads);

  vector<std::thread> t_G(num_threads);

  for (int id = 0; id < num_threads; ++id) {
    t_G[id] = std::thread (compute_2body_ints_memory,
			   id, std::cref(TEI),
			   std::cref(Dm),
			   std::ref(Gt[id]));
  }

  for (int id = 0; id < num_threads; ++id) {
    t_G[id].join();
  }

  for (auto id = 1; id < num_threads; ++id) {
    Gt[0] += Gt[id];
  }

  return 0.5* (Gt[0] + Gt[0].t());
  
}



static double compute_nuclear_repulsion_energy (const vector<Atom>& atoms,
						const vector<QAtom>& atoms_Q)
{
  auto enuc = 0.0;
  
  for (auto i = 0; i < atoms.size(); ++i) {
    double qi = static_cast<double> (atoms[i].atomic_number);
    for (auto j = i+1; j < atoms.size(); ++j) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij*xij + yij*yij + zij*zij;
      auto r = sqrt(r2);
      double qj = static_cast<double> (atoms[j].atomic_number);
      
      enuc += qi*qj / r;
    }

    
    for (auto j = 0; j < atoms_Q.size(); ++j) {
      auto xij = atoms[i].x - atoms_Q[j].x;
      auto yij = atoms[i].y - atoms_Q[j].y;
      auto zij = atoms[i].z - atoms_Q[j].z;
      auto r2 = xij*xij + yij*yij + zij*zij;
      if (r2 > 0.01) { // to avoid the overlapped Qs
	auto r = sqrt(r2);
	double qj = atoms_Q[j].charge;
	enuc += qi*qj / r;
      }
    }
  }

  return enuc;
  
}




arma::mat compute_soad (const vector<Atom>& atoms)
{
  // compute number of atomic orbitals
  size_t nao = 0;
  for (const auto& atom : atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2)
      nao += 1;
    else if (Z <= 10)
      nao += 5;
    else
      throw "SOAD with Z > 10 is not yet supported ";
  }

  arma::mat Dm (nao, nao);
  Dm.zeros();

  size_t ao_offset = 0; // first AO of this atom
  for(const auto& atom: atoms) {
    const auto Z = atom.atomic_number;
    if (Z == 1 || Z == 2) { // H, He
      Dm (ao_offset, ao_offset) = Z; // all electrons go to the 1s
      ao_offset += 1;
    }
    else if (Z <= 10) {
      Dm(ao_offset, ao_offset) = 2; // 2 electrons go to the 1s
      Dm(ao_offset+1, ao_offset+1) = (Z == 3) ? 1 : 2; // Li? only 1 electron in 2s, else 2 electrons
      // smear the remaining electrons in 2p orbitals
      const double num_electrons_per_2p = (Z > 4) ? (double)(Z - 4)/3 : 0;
      for(auto xyz=0; xyz!=3; ++xyz)
        Dm(ao_offset+2+xyz, ao_offset+2+xyz) = num_electrons_per_2p;
      ao_offset += 5;
    }
  }

  return Dm; //
    
}




double electronic_HF_energy (arma::mat& Dm, arma::mat& Hm, arma::mat& Fm)
{// See Eq. (3.184) from Szabo and Ostlund

  const auto nbf = Dm.n_rows;

  double Ehf = 0.0;
  
  for (auto i = 0; i < nbf; i++) 
    for (auto j = 0; j < nbf; j++)
      Ehf += 0.5*Dm(i,j)*(Hm(i,j) + Fm(i,j));
  
  return Ehf;
  
}



RHF::RHF (const vector<Atom>& atoms,
	  const BasisSet& bs,
	  const Integrals& ints,
	  const int qm_chg,
	  const bool l_print,
	  const vector<QAtom>& atoms_Q )
{

  num_threads = std::thread::hardware_concurrency();
  
  // 
  E_nuc = compute_nuclear_repulsion_energy (atoms, atoms_Q);
  //
  const double t_elec = num_electrons(atoms) - qm_chg;
  m_nocc   = round(t_elec/2);

  // -- Core Guess
  arma::mat Hm    = ints.Tm + ints.Vm;
  const auto nbf  = ints.SmInvh.n_rows; 
  const auto nmo  = ints.SmInvh.n_cols; 

  arma::mat Dm(nbf,nbf);
  arma::mat Fm(nbf,nbf);
  
  Dm.zeros();
  Fm.zeros();
  
  {// use SOAD to guess density matrix
    arma::mat Dm_min = compute_soad (atoms);
    BasisSet  bs_min ("STO-3G", atoms);

    Fm = Hm + compute_2body_fock_general (bs, Dm_min, bs_min, true);
    eig_solver.compute (ints.SmInvh, Fm);
    Dm = densityMatrix();
  }

  // ********************
  // SCF Loop
  // ********************
  
  DIIS diis (ints.Sm, ints.SmInvh, Fm);
  
  // compute HF energy
  E_hf = electronic_HF_energy (Dm, Hm, Fm);

  const auto maxiter = 100;
  const auto conv = 1.e-7;
  
  auto iter   = 0;
  auto rmsd   = 1.0;
  auto E_diff = 0.0;
  

  // Prepare for incremental Fock build
  
  do {
    iter++;
    auto E_hf_last = E_hf;
    auto Dm_last = Dm;

    // Build a New Fock Matrix

    if (ints.l_eri_direct) {
      Fm = Hm + compute_2body_fock(bs, Dm, ints.Km);
    }
    else {
      Fm = Hm + g_matrix(ints.TEI, Dm);
    }
    
    Fm = diis.getF (Fm, Dm);

    eig_solver.compute (ints.SmInvh, Fm);
    Dm = densityMatrix ();

    E_hf = electronic_HF_energy (Dm, Hm, Fm);

    E_diff = E_hf - E_hf_last;
    rmsd = norm(Dm - Dm_last);
    /*    
    auto sum = 0.0;
    for (auto ib = 0; ib < nbf; ib++)
      for (auto jb = 0; jb < nbf; jb++) {
	sum += D(ib,jb)*ints.Sm_(jb,ib); // 2 electons per MO
      }
    */

    if (l_print) {
      printf (" %02d %20.12f %20.12f %20.12f %20.12f\n",
	      iter, E_hf, E_hf+E_nuc, E_diff, rmsd);
    }
    
  } while ( (fabs(E_diff) > conv) && (iter < maxiter));

  if (l_print) 
    cout << "EHF " << E_hf << "  " << E_hf + E_nuc << endl;

}


arma::mat RHF::energyWeightDensityMatrix ()
{

  const arma::vec E = eig_solver.eigenvalues();
  const arma::mat C = eig_solver.eigenvectors();
  arma::mat W (C.n_rows, C.n_rows); 
  W.zeros();
  
  for (auto n = 0; n < m_nocc; n++)
    W += 2.0*E(n)*C.col(n)*arma::trans(C.col(n)) ;

  return W;
}


arma::mat RHF::densityMatrix ()
{
  // Form Density Matrix
  // 2.0 : number of occupied electron on each MO.
  
  const arma::mat Cm     = eig_solver.eigenvectors();
  const auto nbf         = Cm.n_rows;
  const arma::mat Cm_occ = Cm.submat(0, 0, nbf-1, m_nocc-1); 
  arma::mat Dm           = 2.0*Cm_occ*Cm_occ.t();
  
  return Dm;

}



}  }  // namespace willow::qcmol

