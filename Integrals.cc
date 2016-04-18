#include "Integrals.hpp"
#include <iostream>

using namespace std;
using namespace libint2;


namespace willow { namespace qcmol {


static arma::mat compute_linear_dependence (arma::mat& Sm);

template <Operator obtype>
arma::mat compute_1body_ints (const BasisSet& bs,
			      const vector<Atom>& atoms = vector<Atom>(),
			      const vector<QAtom>& Q_atoms = vector<QAtom>() );


arma::mat compute_schwartz_ints (const BasisSet& bs);


arma::mat compute_shellblock_norm(const BasisSet& obs,
				  const arma::mat& A);


double* compute_2body_ints (const BasisSet&bs,
			    const arma::mat& Km);


Integrals::Integrals (const vector<Atom>& atoms,
		      const BasisSet& bs,
		      const vector<QAtom>& Q_atoms)
{

  // Compute Overlap Integrals
  Sm = compute_1body_ints<Operator::overlap> (bs);

  // Compute Kinetic Integrals
  Tm = compute_1body_ints<Operator::kinetic> (bs);
  
  // Compute Nuclear Attraction Integrals
  Vm = compute_1body_ints<Operator::nuclear> (bs, atoms, Q_atoms);

  // Sinvh_
  SmInvh = compute_linear_dependence (Sm);
  
  Km = compute_schwartz_ints (bs); 

  // Two
  l_eri_direct = true;

  if (bs.nbf() < 200) {
    l_eri_direct = false;
    TEI = compute_2body_ints (bs, Km);
  }

  
}


template <Operator obtype>
arma::mat compute_1body_ints (const BasisSet& bs,
			      const vector<Atom>& atoms,
			      const vector<QAtom>& Q_atoms) 
{
  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();

  arma::mat result (nbf,nbf);
  
  // construct the overlap integrals engine
  libint2::Engine engine (obtype, bs.max_nprim(), bs.max_l(), 0);
  
  // Nuclear Attraction Integrals Engine needs to know where the charges sit...
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
  
  auto shell2bf = bs.shell2bf();

  // Loop over unique shell pairs, {s1, s2} such that s1 >= s2
  // this is due to the permutational symmetry of the real integrals over
  // Hermitian operators : (1|2) = (2|1)
  for (auto s1 = 0; s1 != nshells; ++s1) {
    auto bf1  = shell2bf[s1]; // first basis function in this shell
    auto nbf1 = bs[s1].size(); //shells[s1].size ();

    for (auto s2 = 0; s2 <= s1; ++s2) {
      auto bf2   = shell2bf[s2]; // first basis function in this shell
      auto nbf2  = bs[s2].size(); //shells[s2].size ();

      // compute shell pair; return is the pointer to the buffer
      const auto* buf = engine.compute (bs[s1], bs[s2]);

      for (auto ib = 0, ij = 0; ib < nbf1; ib++) 
	for (auto jb = 0; jb < nbf2; jb++, ij++) {
	  const double val = buf[ij];
	    
	  result(bf1+ib,bf2+jb) = val;
	  result(bf2+jb,bf1+ib) = val;
	}
      
    }
    
  } // s1
  
  return result;

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
    cout << "*** WARNING *** " << Ndep << 
      " LINEARLY DEPENDENT AND/OR REDUNDANT CARTESIAN FUNCTION ";
  }

  return SmInvh;

}


arma::mat compute_schwartz_ints (const BasisSet& bs)
{

  const auto nsh   = bs.size();
  const auto nsize = (nsh*(nsh+1))/2;

  arma::mat K(nsh, nsh);
  K.zeros();
  
  // Construct the 2-electron repulsion integrals engine
  auto epsilon = 0.0;
  libint2::Engine engine = libint2::Engine(Operator::coulomb, bs.max_nprim(),
					   bs.max_l(), 0, epsilon);
  //engine.set_precision(0.0);
  
  auto shell2bf = bs.shell2bf();
  
  auto nsave = 0;

  // loop over permutationally-unique set of shells
  for (auto s1 = 0, s12 = 0; s1 != nsh; ++s1) {
    auto nbf1 = bs[s1].size();
    
    for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {
      auto nbf2 = bs[s2].size();
      auto nbf12 = nbf1*nbf2;
      
      const auto* buf = engine.compute (bs[s1], bs[s2], bs[s1], bs[s2]);

      auto max_value = 0.0;
      for (auto f1 = 0, f12 = 0; f1 < nbf1; ++f1)
	for (auto f2 = 0; f2 < nbf2; ++f2, ++f12) {
	  const auto eri4 = abs(buf[f12*nbf12 + f12]);
	  const auto val  = sqrt(eri4);

	  if (max_value < val) max_value = val;
	}

      K(s1,s2) = K(s2,s1) = max_value;
	  
    }
  }

  return K;

}



double* compute_2body_ints (const BasisSet& bs,
			    const arma::mat& Schwartz)
{

  const auto nbf     = bs.nbf(); 
  const auto nshells = bs.size();
  const auto nints = ((nbf*(nbf+1)/2)*((nbf*(nbf + 1)/2) + 1)/2);

  auto result = new double[nints];

  fill (result, result+nints, 0.0);

  const auto precision = numeric_limits<double>::epsilon();
  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine = libint2::Engine(Operator::coulomb,
					   bs.max_nprim(),
					   bs.max_l(),
					   0);
  engine.set_precision(precision);
  
  auto shell2bf = bs.shell2bf();
  auto nsave = 0;
  auto ncount = 0;

  // loop over permutationally-unique set of shells
  for (auto s1 = 0, s12 = 0; s1 != nshells; ++s1) {
    
    for (auto s2 = 0; s2 <= s1; ++s2, ++s12) {
      auto s12_cut   = Schwartz(s1,s2);

      for (auto s3 = 0; s3 <= s1; ++s3) {

	const auto s4_max = (s1 == s3) ? s2 : s3;
	for (auto s4 = 0; s4 <= s4_max; ++s4) {
	  auto s34_cut   = Schwartz(s3,s4);
	  
	  if (s12_cut*s34_cut < precision) {
	    ++ncount;
	    continue;
	  }

	  // swap
	  const auto swap_bra = (bs[s1].contr[0].l < bs[s2].contr[0].l);
	  const auto swap_ket = (bs[s3].contr[0].l < bs[s4].contr[0].l);
	  const auto swap_braket =
	    ((bs[s1].contr[0].l + bs[s2].contr[0].l) >
	     (bs[s3].contr[0].l + bs[s4].contr[0].l));

	  // In order to reduce a bug in engine.h
	  auto t1 = s1;
	  auto t2 = s2;
	  auto t3 = s3;
	  auto t4 = s4;

	  if (swap_braket) {
	    if (swap_ket) {
	      t1 = s4;
	      t2 = s3;
	    }
	    else {
	      t1 = s3;
	      t2 = s4;
	    }
	    if (swap_bra) {
	      t3 = s2;
	      t4 = s1;
	    }
	    else {
	      t3 = s1;
	      t4 = s2;
	    }
	  }
	  else {
	    if (swap_bra) {
	      t1 = s2;
	      t2 = s1;
	    }
	    if (swap_ket) {
	      t3 = s4;
	      t4 = s3;
	    }
	  }

	  auto bf1_first = shell2bf[t1];
	  auto nbf1      = bs[t1].size();
	  
	  auto bf2_first = shell2bf[t2];
	  auto nbf2      = bs[t2].size();
	  
	  auto bf3_first = shell2bf[t3];
	  auto nbf3      = bs[t3].size();
	  
	  auto bf4_first = shell2bf[t4];
	  auto nbf4      = bs[t4].size();
	  
	  //const auto* buf = engine.compute (bs[s1], bs[s2], bs[s3], bs[s4]);
	  const auto* buf = engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	    (bs[t1], bs[t2], bs[t3], bs[t4]);
	  
	  for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
	    const auto bf1 = f1 + bf1_first;
	    
	    for (auto f2 = 0; f2 != nbf2; ++f2) {
	      const auto bf2 = f2 + bf2_first;
	      
	      for (auto f3 = 0; f3 != nbf3; ++f3) {
		const auto bf3 = f3 + bf3_first;
		
		for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		  const auto bf4 = f4 + bf4_first;
		  const auto eri4 = buf[f1234];
		  
		  auto ij = INDEX(bf1, bf2);
		  auto kl = INDEX(bf3, bf4);
		  auto ijkl = INDEX (ij, kl);

		  if (ijkl < nints) {
		    result[ijkl] = eri4;
		  }
		  else {
		    cout << "ERROR IJKL " << ijkl << endl;
		  }
		  
		}
	      }
	    }
	  }
	}
      }
    }
  }

  //cout << " JUMP NUM " << ncount << endl;

  return result;

}




arma::mat compute_2body_fock (const BasisSet& bs,
			      const arma::mat& Dm, // density matrix
			      const arma::mat& Schwartz)
{

  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();
  arma::mat Gm (nbf, nbf);
  
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
	for (auto s4 = 0; s4 <= s4_max; ++s4)  {
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

	  //const auto* buf = engine.compute (bs[s1], bs[s2], bs[s3], bs[s4]);
	  const auto* buf = engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	    (bs[s1], bs[s2], bs[s3], bs[s4]);
	  
	  for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
	    const auto bf1 = f1 + bf1_first;
	    
	    for (auto f2 = 0; f2 != nbf2; ++f2) {
	      const auto bf2 = f2 + bf2_first;
	      
	      for (auto f3 = 0; f3 != nbf3; ++f3) {
		const auto bf3 = f3 + bf3_first;
	    
		for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		  const auto bf4 = f4 + bf4_first;

		  const auto value = buf[f1234];

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
  
  return 0.25* (Gm + Gm.t()); 
  
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
  
  arma::mat Gm (nbf, nbf);
  
  Gm.zeros();
    
  // matrix of norms of shell blocks

  const auto precision = numeric_limits<double>::epsilon();
  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine (Operator::coulomb,
			  max(bs.max_nprim(),Dm_bs.max_nprim()),
			  max(bs.max_l(), Dm_bs.max_l()),
			  0);

  engine.set_precision (precision);
  
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
	    const auto* buf_J = engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	      (bs[s1], bs[s2], Dm_bs[s3], Dm_bs[s4]);
	  
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

	    const auto* buf_K = engine.compute2<Operator::coulomb,BraKet::xx_xx,0>
	      (bs[s1], Dm_bs[s3], bs[s2], Dm_bs[s4]);

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
