#include "Integrals.hpp"
#include <iostream>
#include <thread>

using namespace std;
using namespace libint2;



namespace willow { namespace qcmol {



static unsigned int num_threads;

static arma::mat compute_linear_dependence (arma::mat& Sm);


template <Operator obtype>
void compute_1body_ints (const int thread_id,
			 const BasisSet& bs,
			 arma::mat& result,
			 const vector<Atom>& atoms,
			 const vector<QAtom>& Q_atoms);
				

void compute_schwartz_ints (const int thread_id,
			    const BasisSet& bs,
			    arma::mat& result);


arma::mat compute_shellblock_norm(const BasisSet& obs,
				  const arma::mat& A);


void compute_2body_ints (const int thread_id,
			 const BasisSet&bs,
			 const arma::mat& Km,
			 arma::vec& result);


void compute_2body_ints_direct (const int thread_id,
				const BasisSet& bs,
				const arma::mat& Dm,
				const arma::mat& Schwartz,
				arma::mat& result);


Integrals::Integrals (const vector<Atom>& atoms,
		      const BasisSet& bs,
		      const vector<QAtom>& Q_atoms)
{
  num_threads = std::thread::hardware_concurrency ();
  //cout << "NUM THREAD " << num_threads << endl;
  
  if (num_threads == 0)
    num_threads = 1;
  
  vector<arma::mat> St(num_threads);
  vector<arma::mat> Tt(num_threads);
  vector<arma::mat> Vt(num_threads);
  
  vector<std::thread> t_S(num_threads);
  vector<std::thread> t_T(num_threads);
  vector<std::thread> t_V(num_threads);
  
  
  for (int id = 0; id < num_threads; ++id) {
    
    t_S[id] = std::thread (compute_1body_ints<Operator::overlap>,
			   id, std::cref(bs), std::ref(St[id]),
			   atoms, Q_atoms);
    t_T[id] = std::thread (compute_1body_ints<Operator::kinetic>,
			   id, std::cref(bs), std::ref(Tt[id]),
			   atoms, Q_atoms);
    t_V[id] = std::thread (compute_1body_ints<Operator::nuclear>,
			   id, std::cref(bs), std::ref(Vt[id]),
			   atoms, Q_atoms);
  }
  
  // Join
  for (auto id = 0; id < num_threads; ++id) {
    t_S[id].join();
    t_T[id].join();
    t_V[id].join();
  }

  Sm = St[0];
  Tm = Tt[0];
  Vm = Vt[0];
  
  for (auto id = 1; id < num_threads; ++id) {
    Sm += St[id];
    Tm += Tt[id];
    Vm += Vt[id];
  }
  
  // Sinvh_
  SmInvh = compute_linear_dependence (Sm);


  // ERI
  vector<arma::mat>   Kt(num_threads);
  vector<std::thread> t_K(num_threads);

  for (auto id = 0; id < num_threads; ++id) {
    t_K[id] = std::thread (compute_schwartz_ints,
			   id, std::cref(bs), std::ref(Kt[id])); 
  }
  
  // Join
  for (auto id = 0; id < num_threads; ++id) {
    t_K[id].join();
  }

  Km = Kt[0];

  for (auto id = 1; id < num_threads; ++id) {
    Km += Kt[id];
  }
  
  // Two
  l_eri_direct = true;

  if (bs.nbf() < 200) {
    l_eri_direct = false;

    vector<arma::vec>   eri_t(num_threads);
    vector<std::thread> t_eri(num_threads);

    for (auto id = 0; id < num_threads; ++id) {
      t_eri[id] = std::thread (compute_2body_ints,
			       id,
			       std::cref(bs), 
			       std::cref(Km),
			       std::ref(eri_t[id]) );
    }

    for (auto id = 0; id < num_threads; ++id)
      t_eri[id].join();
  
    for (auto id = 1; id < num_threads; ++id) {
      eri_t[0] += eri_t[id];
    }

    TEI = eri_t[0];
    
  }

  
}


template <Operator obtype>
void compute_1body_ints (const int thread_id,
			 const BasisSet& bs,
			 arma::mat& result,
			 const vector<Atom>& atoms,
			 const vector<QAtom>& Q_atoms) 
{
  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();

  result = arma::mat (nbf,nbf);
  result.zeros();
  
  // construct the overlap integrals engine
  libint2::Engine engine (obtype, bs.max_nprim(), bs.max_l(), 0);

  if (obtype == Operator::nuclear) {
    vector<pair<double, array<double,3>>> q;
    for (const auto& atom : atoms) {
      q.push_back ( {static_cast<double> (atom.atomic_number),
	    {{atom.x, atom.y, atom.z}}} );
    }
    
    if (Q_atoms.size() > 0) {
      
      for (const auto& q_atom : Q_atoms) {
	q.push_back ( {q_atom.charge,
	      {{q_atom.x, q_atom.y, q_atom.z}} } );
      }
    }
    
    engine.set_params (q);
  }
  
  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();

  // Loop over unique shell pairs, {s1, s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over
  // Hermitian operators : (1|2) = (2|1)
  for (auto s1 = 0, s12 = 0; s1 != nshells; ++s1) {
    auto bf1  = shell2bf[s1]; // first basis function in this shell
    auto nbf1 = bs[s1].size(); //shells[s1].size ();

    for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {

      if (s12 % num_threads != thread_id) continue;
      
      auto bf2   = shell2bf[s2]; // first basis function in this shell
      auto nbf2  = bs[s2].size(); //shells[s2].size ();

      // compute shell pair; return is the pointer to the buffer
      engine.compute (bs[s1], bs[s2]);
      const auto* buf0 = buf[0];
      
      for (auto ib = 0, ij = 0; ib < nbf1; ib++) 
	for (auto jb = 0; jb < nbf2; jb++, ij++) {
	  const double val = buf0[ij];
	    
	  result(bf1+ib,bf2+jb) = val;
	  result(bf2+jb,bf1+ib) = val;
	}
      
    }
    
  } // s1
  

}






arma::mat compute_linear_dependence (arma::mat& Sm)
// Orthogonalize Basis (Canonical Orthogonalization)
{
  arma::vec Sval;
  arma::mat Svec;
  bool ok = arma::eig_sym (Sval, Svec, Sm);

  if (! ok) {
    std::cout << "Unable to diagonalize matrix ! \n";
    throw std::runtime_error ("Error in eig_sym. \n");
  }
  
  const double tol = 1.0e-6;
  const size_t Nbf = Svec.n_rows;
  size_t Nlin = 0;

  for (size_t i = 0; i < Nbf; i++) {
    if (Sval(i) >= tol) Nlin++;
  }

  // Number of linearly dependent basis functions
  size_t Ndep = Nbf - Nlin;


  // Form Return Matrix
  arma::mat SmInvh (Nbf, Nlin);
  
  for (size_t i = 0; i < Nlin; i++) {
    SmInvh.col(i) = Svec.col(Ndep+i)/sqrt(Sval(Ndep+i));
  }

  if (Ndep > 0) {
    cout << "*** WARNING *** \n \t" << Ndep << 
      "\t LINEARLY DEPENDENT AND/OR REDUNDANT CARTESIAN FUNCTION \n";
  }

  return SmInvh;

}


void compute_schwartz_ints (const int thread_id,
			    const BasisSet& bs,
			    arma::mat& result)
{

  const auto nsh   = bs.size();

  result = arma::mat (nsh, nsh);
  result.zeros();
  
  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine = libint2::Engine(Operator::coulomb,
					   bs.max_nprim(),
					   bs.max_l(),
					   0);
  
  const auto precision = numeric_limits<double>::epsilon();
  engine.set_precision(precision);
  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();
  
  // loop over permutationally-unique set of shells
  for (auto s1 = 0, s12 = 0; s1 != nsh; ++s1) {
    auto nbf1 = bs[s1].size();
    
    for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {
      
      if (s12%num_threads != thread_id) continue;
      
      auto nbf2 = bs[s2].size();
      auto nbf12 = nbf1*nbf2;

      engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	(bs[s1], bs[s2], bs[s1], bs[s2]);
      const auto* buf0 = buf[0];
      
      auto max_value = 0.0;

      if (buf0 != nullptr) {
	
	for (auto f1 = 0, f12 = 0; f1 < nbf1; ++f1)
	  for (auto f2 = 0; f2 < nbf2; ++f2, ++f12) {
	    const auto eri4 = abs(buf0[f12*nbf12 + f12]);
	    const auto val  = sqrt(eri4);
	    if (max_value < val) max_value = val;
	  }
      }
      result(s1,s2) = result(s2,s1) = max_value;
    }
  }

}



void compute_2body_ints (const int thread_id,
			 const BasisSet& bs,
			 const arma::mat& Schwartz,
			 arma::vec& result)
{

  const auto nbf     = bs.nbf(); 
  const auto nshells = bs.size();
  const auto nints = ((nbf*(nbf+1)/2)*((nbf*(nbf + 1)/2) + 1)/2);

  //auto result = new double[nints];
  result = arma::vec(nints);
  result.zeros();

  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine = libint2::Engine(Operator::coulomb,
					   bs.max_nprim(),
					   bs.max_l(),
					   0);
  
  const auto precision = numeric_limits<double>::epsilon();
  engine.set_precision(precision);
  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();
  auto nsave = 0;
  auto ncount = 0;

  // loop over permutationally-unique set of shells
  for (auto s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];
    auto nbf1      = bs[s1].size();
    
    for (auto s2 = 0; s2 <= s1; ++s2) {
      //if (s12 % num_threads != thread_id) continue;
      
      auto s12_cut   = Schwartz(s1,s2);
      auto bf2_first = shell2bf[s2];
      auto nbf2      = bs[s2].size();

      for (auto s3 = 0; s3 <= s1; ++s3) {
	auto bf3_first = shell2bf[s3];
	auto nbf3      = bs[s3].size();

	const auto s4_max = (s1 == s3) ? s2 : s3;
	for (auto s4 = 0; s4 <= s4_max; ++s4, ++s1234) {
	  auto bf4_first = shell2bf[s4];
	  auto nbf4      = bs[s4].size();
	  auto s34_cut   = Schwartz(s3,s4);

	  if (s1234 % num_threads != thread_id) continue;
	  
	  if (s12_cut*s34_cut < precision) {
	    ++ncount;
	    continue;
	  }

	  //const auto* buf = engine.compute (bs[s1], bs[s2], bs[s3], bs[s4]);
	  engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	    (bs[s1], bs[s2], bs[s3], bs[s4]);
	  const auto* buf0 = buf[0];

	  if (buf0 == nullptr) continue;
	  
	  for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
	    const auto bf1 = f1 + bf1_first;
	    
	    for (auto f2 = 0; f2 != nbf2; ++f2) {
	      const auto bf2 = f2 + bf2_first;
	      
	      for (auto f3 = 0; f3 != nbf3; ++f3) {
		const auto bf3 = f3 + bf3_first;
		
		for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		  const auto bf4 = f4 + bf4_first;
		  const auto eri4 = buf0[f1234];
		  
		  auto ij = INDEX(bf1, bf2);
		  auto kl = INDEX(bf3, bf4);
		  auto ijkl = INDEX (ij, kl);

		  //if (ijkl < nints) {
		  result(ijkl) = eri4;
		  //}
		  //else {
		  //  cout << "ERROR IJKL " << ijkl << endl;
		  //}
		  
		}
	      }
	    }
	  }
	}
      }
    }
  }

  //cout << " JUMP NUM " << ncount << endl;

}




arma::mat compute_2body_fock (const BasisSet& bs,
			      const arma::mat& Dm, // density matrix
			      const arma::mat& Schwartz)
{

  vector<arma::mat>   Gt(num_threads);

  vector<std::thread> t_G(num_threads);

  for (int id = 0; id < num_threads; ++id) {
    t_G[id] = std::thread (compute_2body_ints_direct,
			   id, std::cref(bs), std::cref(Dm),
			   std::cref(Schwartz),
			   std::ref(Gt[id]));
  }

  for (int id = 0; id < num_threads; ++id) {
    t_G[id].join();
  }

  for (auto id = 1; id < num_threads; ++id) {
    Gt[0] += Gt[id];
  }

  return 0.25* (Gt[0] + Gt[0].t());
  
}



void compute_2body_ints_direct (const int thread_id,
				const BasisSet& bs,
				const arma::mat& Dm, // density matrix
				const arma::mat& Schwartz,
				arma::mat& Gm)
{
  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();
  Gm = arma::mat (nbf, nbf);
  
  Gm.zeros();
    
  // matrix of norms of shell blocks
  arma::mat D_shblk_norm = compute_shellblock_norm (bs, Dm);
  
  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine(Operator::coulomb,
			 bs.max_nprim(),
			 bs.max_l(),
			 0);

  const auto precision = numeric_limits<double>::epsilon();
  engine.set_precision (precision);
  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();
  
  // loop over permutationally-unique set of shells
  for (auto s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
    
    auto bf1_first = shell2bf[s1]; // first basis function in this shell
    auto nbf1      = bs[s1].size(); // number of basis functions in this shell
    
    for (auto s2 = 0; s2 <= s1; ++s2) {
      
      auto bf2_first = shell2bf[s2]; 
      auto nbf2      = bs[s2].size(); 
      
      const auto Dnorm12 = D_shblk_norm(s1,s2);

      for (auto s3 = 0; s3 <= s1; ++s3) {
	
	auto bf3_first = shell2bf[s3]; 
	auto nbf3      = bs[s3].size();
	
	const auto Dnorm123 =  max(D_shblk_norm(s1,s3),
				   max (D_shblk_norm(s2,s3), Dnorm12));

	const auto s4_max = (s1 == s3) ? s2 : s3;
	for (auto s4 = 0; s4 <= s4_max; ++s4, ++s1234)  {

	  if (s1234 % num_threads != thread_id) continue;
	  
	  auto bf4_first = shell2bf[s4];
	  auto nbf4      = bs[s4].size();
	  
	  const auto Dnorm1234 = max(D_shblk_norm(s1,s4),
				     max(D_shblk_norm(s2,s4),
					 max(D_shblk_norm(s3,s4),Dnorm123)));

	  if (Dnorm1234 * Schwartz(s1,s2) * Schwartz(s3,s4) < precision)
	    continue;

	  // compute the permutational degeneracy
	  // of the given shell set
	  auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
	  auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
	  auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0: 2.0) : 2.0;
	  auto s1234_deg = s12_deg*s34_deg*s12_34_deg;

	  engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	    (bs[s1], bs[s2], bs[s3], bs[s4]);
	  const auto* buf0 = buf[0];

	  if (buf0 == nullptr) continue;
	  
	  for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
	    const auto bf1 = f1 + bf1_first;
	    
	    for (auto f2 = 0; f2 != nbf2; ++f2) {
	      const auto bf2 = f2 + bf2_first;
	      
	      for (auto f3 = 0; f3 != nbf3; ++f3) {
		const auto bf3 = f3 + bf3_first;
	    
		for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		  const auto bf4 = f4 + bf4_first;

		  const auto value = buf0[f1234];

		  const auto value_scal_by_deg = value * s1234_deg;
		  
		  Gm(bf1,bf2) += Dm(bf3,bf4) * value_scal_by_deg;
		  Gm(bf3,bf4) += Dm(bf1,bf2) * value_scal_by_deg;
		  Gm(bf1,bf3) -= 0.25*Dm(bf2,bf4) * value_scal_by_deg;
		  Gm(bf2,bf4) -= 0.25*Dm(bf1,bf3) * value_scal_by_deg;
		  Gm(bf1,bf4) -= 0.25*Dm(bf2,bf3) * value_scal_by_deg;
		  Gm(bf2,bf3) -= 0.25*Dm(bf1,bf4) * value_scal_by_deg;
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  // Symmetrize the result and return
  
  //return 0.25* (Gm + Gm.t()); 
  
}


arma::mat compute_2body_fock_general (const BasisSet& bs,
				      const arma::mat& Dm, // density matrix
				      const BasisSet& Dm_bs,
				      bool  Dm_is_diagonal)
{

  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();
  const auto nbf_Dm  = Dm_bs.nbf();
  const auto nshells_Dm = Dm_bs.size();

  assert( (Dm.n_cols == Dm.n_rows) && (Dm.n_cols == nbf_Dm) );
  
  arma::mat Gm (nbf, nbf, arma::fill::zeros);
  
  // matrix of norms of shell blocks

  const auto precision = numeric_limits<double>::epsilon();
  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine (Operator::coulomb,
			  max(bs.max_nprim(),Dm_bs.max_nprim()),
			  max(bs.max_l(), Dm_bs.max_l()),
			  0);

  engine.set_precision (precision);
  const auto& buf = engine.results();
  
  auto shell2bf    = bs.shell2bf();
  auto shell2bf_Dm = Dm_bs.shell2bf();

  // loop over permutationally-unique set of shells
  for (auto s1 = 0; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];
    auto nbf1      = bs[s1].size();
    
    for (auto s2 = 0; s2 <= s1; ++s2) {
      auto bf2_first = shell2bf[s2];
      auto nbf2      = bs[s2].size();
      
      for (auto s3 = 0; s3 < nshells_Dm; ++s3) {
	auto bf3_first = shell2bf_Dm[s3];
	auto nbf3      = Dm_bs[s3].size();

	const auto s4_begin = Dm_is_diagonal ? s3 : 0;
	const auto s4_fence = Dm_is_diagonal ? s3+1 : nshells_Dm;
	
	for (auto s4 = s4_begin; s4 != s4_fence; ++s4)  {
	  auto bf4_first = shell2bf_Dm[s4];
	  auto nbf4      = Dm_bs[s4].size();
	  
	  // compute the permutational degeneracy
	  // of the given shell set
	  auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

	  if (s3 >= s4) {
	    auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
	    auto s1234_deg = s12_deg*s34_deg;
	    
	    //const auto* buf_J = engine.compute (bs[s1], bs[s2],
	    //					Dm_bs[s3], Dm_bs[s4]);
	    engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	      (bs[s1], bs[s2], Dm_bs[s3], Dm_bs[s4]);

	    const auto* buf_J = buf[0];
	    
	    if (buf_J != nullptr) {
	      
	      for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
		const auto bf1 = f1 + bf1_first;
		for (auto f2 = 0; f2 != nbf2; ++f2) {
		  const auto bf2 = f2 + bf2_first;
		  for (auto f3 = 0; f3 != nbf3; ++f3) {
		    const auto bf3 = f3 + bf3_first;
		    for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		      const auto bf4 = f4 + bf4_first;
		      
		      const auto value = buf_J[f1234];
		      const auto value_scal_by_deg = value * s1234_deg;
		      
		      Gm(bf1,bf2) += 2.0*Dm(bf3,bf4)*value_scal_by_deg;
		  
		    }
		  }
		}
	      } // f1
	    } // end if
	    
	    engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	      (bs[s1], Dm_bs[s3], bs[s2], Dm_bs[s4]);
	    const auto* buf_K = buf[0];

	    if (buf_K != nullptr) {
	      
	      for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
		const auto bf1 = f1 + bf1_first;
		for (auto f3 = 0; f3 != nbf3; ++f3) {
		  const auto bf3 = f3 + bf3_first;
		  for (auto f2 = 0; f2 != nbf2; ++f2) {
		    const auto bf2 = f2 + bf2_first;
		    for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		      const auto bf4 = f4 + bf4_first;
		      
		      const auto value = buf_K[f1234];
		      const auto value_scal_by_deg = value * s12_deg;
		      
		      Gm(bf1,bf2) -= Dm(bf3,bf4)*value_scal_by_deg;
		      
		    }
		  }
		}
	      } // f1
	    } // end if buf_K
	    
	  } // if (s3 >= s4)
	}
      }
    }
  }
  
  // Symmetrize the result and return
  
  return 0.25*(Gm + Gm.t());
  
}



arma::mat compute_shellblock_norm (const BasisSet& bs,
				   const arma::mat& A)
{
  const auto nsh = bs.size();
  arma::mat Ash (nsh, nsh); //= Matrix::Zero(nsh, nsh);
  Ash.zeros();
  
  auto shell2bf = bs.shell2bf();

  for (size_t s1 = 0; s1 != nsh; ++s1) {
    const auto s1_first = shell2bf[s1];
    const auto s1_last  = s1_first + bs[s1].size() - 1;

    for (size_t s2 = 0; s2 != nsh; ++s2) {
      const auto s2_first = shell2bf[s2];
      const auto s2_last  = s2_first + bs[s2].size() - 1;

      auto Ablk = A.submat (s1_first, s2_first,
			    s1_last,  s2_last);
      Ash(s1,s2) = norm(Ablk, "inf");
    }
  }
    
  return Ash;
  
}


} } // namespace willow::qcmol
