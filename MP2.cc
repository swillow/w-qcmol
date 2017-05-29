#include "RHF.hpp"
#include "MP2.hpp"



namespace willow { namespace qcmol {



MP2::MP2 (const std::vector<libint2::Atom>& atoms,
	  const libint2::BasisSet& bs,
	  const Integrals& ints,
	  const int qm_chg,
	  const bool l_print,
	  const std::vector<QAtom>& atoms_Q)
  : RHF (atoms, bs, ints, qm_chg, l_print, atoms_Q)
{
  
  erhf = rhf_energy ();
  emp2 = 0.0;
  
  if (ints.l_eri_direct) {
    if (l_print)
      std::cout << "DIRECT EMP2 CALCULATION IS NOT READY. TT    "  << std::endl;

    return;
  }

  nocc   = get_nocc();
  ncore  = compute_ncore(atoms);
  nbf    = ints.SmInvh.n_rows;
  nmo    = ints.SmInvh.n_cols; 

  iocc1  = ncore;
  iocc2  = nocc - 1;
  ivir1  = nocc;
  ivir2  = nmo - 1;

  // active occupied MO
  // active virtual  MO
  naocc = nocc - ncore;
  navir = nmo  - nocc;
  nvir  = navir;

  
  eval = eig_solver.eigenvalues();
  Cmat = eig_solver.eigenvectors();

  if (l_print) {
    std::cout << "nmo " << nmo << std::endl;
    std::cout << "naocc " << naocc << std::endl;
    std::cout << "navir " << navir << std::endl;
  }
  
  //
  // save (moi moa | moj mob)
  // E(2) = sum_{i,j,a,b} {2 (ia | jb)(ia | jb) - (ia|jb)(ib|ja)}
  //
  
  
  // J = coloumb-like integral (ia|jb)
  // K = exchange-like integral (ib|ja) or (ja|ib)
  // Note Same Spin     = J - K
  // NOTE Opposite Spin = J
  
  //
  // MO INTS (ia|jb)
  arma::mat mo_hf_int = mo_half_ints (ints.TEI.memptr() );
  arma::mat mo_tei    = mo_iajb_ints (mo_hf_int);
  
  // 
  // T(im,am|jm,bm)
  // 2*(im,am|jm,bm) - (im,bm|jm,am)
  //
  
  for (auto im = 0; im < naocc; im++) {
    auto moi = iocc1 + im;
    
    for (auto am = 0; am < navir; am++) {
      auto moa = ivir1 + am;
      
      for (auto moj = iocc1; moj < nocc; moj++) {
	
        for (auto bm = 0; bm < navir; bm++) {
	  auto mob = ivir1 + bm;
	  
	  double den = eval(moi) + eval(moj) - eval(moa) - eval(mob);
	  
	  auto mia = im*navir + am;
	  auto mib = im*navir + bm;
	  
	  auto mjb = moj*navir + bm;
	  auto mja = moj*navir + am;
	  
	  auto o_iajb = mo_tei(mjb, mia);
	  auto o_ibja = mo_tei(mja, mib);
	  
	  auto t_iajb = (2.0*o_iajb - o_ibja)/den;
	  
	  emp2 += t_iajb * o_iajb;
	  
	} // bm
	
      } // moj
    }  // am
  }    // im

  if (l_print)
    std::cout << "EMP2     " << emp2 << std::endl;

}


}  }  // namespace willow::qcmol
