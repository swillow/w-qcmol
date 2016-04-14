#include "RHF.hpp"
#include "diis.hpp"
//#include "Integrals.hpp"
//#include "Molecule.hpp"
//#include "Matrix.hpp"


arma::mat g_matrix (double* TEI, arma::mat& Dm)
{
  const auto nbf = Dm.n_rows;

  // return G = J - 0.5K
  arma::mat Jm (nbf,nbf);
  arma::mat Km (nbf,nbf);

  Jm.zeros();
  Km.zeros();

  for (auto i = 0; i < nbf; i++) {
    auto it = i;
    
    for (auto j = 0; j <= it; j++) {
      auto ij = INDEX(it,j);
      
      for (auto k = 0; k <= it; k++)
	for (auto l = 0; l <= k; l++) {
	  auto kl = INDEX(k,l);
	  
	  if (kl > ij) continue;

	  auto ijkl = INDEX(ij,kl);
	  auto eri4 = TEI [ijkl];
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


  return (Jm - 0.5*Km);

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
	  const Integrals& ints)
{
  // 
  E_nuc = compute_nuclear_repulsion_energy (atoms);
  //
  auto nocc = num_electrons(atoms)/2;

  // Create Shells Centered on Atomic Positions
  //auto shells  = bs.sh_list;
  
  // -- Core Guess
  arma::mat Hm    = ints.Tm + ints.Vm;
  const auto nbf  = ints.SmInvh.n_rows; 
  const auto nmo  = ints.SmInvh.n_cols; 

  arma::mat Dm(nbf,nbf);
  Dm.zeros();
  
  {// Hcore
    eig_solver.compute (ints.SmInvh, Hm);
    Dm = densityMatrix(nocc);
  }

  // pre-compute data for Schwartz bounds
  auto Km = ints.Km;

  // ********************
  // SCF Loop
  // ********************
  
  arma::mat Fm = Hm;
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
    
    Fm = Hm + g_matrix(ints.TEI, Dm);
    Fm = diis.getF (Fm, Dm);

    eig_solver.compute (ints.SmInvh, Fm);
    Dm = densityMatrix (nocc);
    if (iter < 3) {
      Dm = 0.7*Dm + 0.3*Dm_last;
    }
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

  //  printf (" %02d %20.12f %20.12f %20.12f %20.12f\n",
  //  	    iter, E_hf, E_hf+E_nuc, E_diff, rmsd);
  } while ( (fabs(E_diff) > conv) && (iter < maxiter));

  cout << "EHF " << E_hf << "  " << E_hf + E_nuc << endl;

  // return ehf;

}


arma::mat RHF::energyWeightDensityMatrix (const int nocc) 
{

  const arma::vec E = eig_solver.eigenvalues();
  const arma::mat C = eig_solver.eigenvectors();
  arma::mat W (C.n_rows, C.n_rows); 
  W.zeros();
  
  for (auto n = 0; n < nocc; n++)
    W += 2.0*E(n)*C.col(n)*arma::trans(C.col(n)) ;

  return W;
}


arma::mat RHF::densityMatrix (const int nocc)
{
  // Form Density Matrix
  // 2.0 : number of occupied electron on each MO.
  
  const arma::mat Cm     = eig_solver.eigenvectors();
  const auto nbf         = Cm.n_rows;
  const arma::mat Cm_occ = Cm.submat(0, 0, nbf-1, nocc-1); 
  arma::mat Dm           = 2.0*Cm_occ*Cm_occ.t();
  
  return Dm;

}



