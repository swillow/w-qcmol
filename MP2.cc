#include "RHF.hpp"
#include "MP2.hpp"


using namespace std;
using namespace libint2;

namespace willow { namespace qcmol {


#define MINDEX(i,a,j,b,nocc,nvir)  ( ( ((i)*(nvir) + (a))*(nocc) + (j))*(nvir) + (b) ) 


MP2::MP2 (const vector<Atom>& atoms,
	  const BasisSet& bs,
	  const Integrals& ints,
	  const int qm_chg,
	  const bool l_print,
	  const vector<QAtom>& atoms_Q)
  : RHF (atoms, bs, ints, qm_chg, l_print, atoms_Q)
{
  
  //RHF rhf (atoms, bs, ints, qm_chg, l_print, atoms_Q);

  erhf = energy ();
  emp2 = 0.0;
  
  if (ints.l_eri_direct) {
    if (l_print)
      cout << "DIRECT EMP2 CALCULATION IS NOT READY. TT    "  << endl;

    return;
  }

  const double t_elec = num_electrons (atoms) - qm_chg;
  nocc   = round(t_elec/2); 
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

  eval = eig_solver.eigenvalues();
  Cmat = eig_solver.eigenvectors();

  if (l_print) {
    cout << "nmo " << nmo << endl;
    cout << "naocc " << naocc << endl;
    cout << "navir " << navir << endl;
  }
  
  //
  // save (moi moa | moj mob)
  // E(2) = sum_{i,j,a,b} {2 (ia | jb)(ia | jb) - (ia|jb)(ib|ja)}
  //
  
  
  fss   = 1.0;
  fos   = 1.0;
  bool o_mp2_scs = false;

  if (o_mp2_scs) {
    fss = 1.0/3.0;
    fos = 1.2; // 6.0/5.0;
  }

  // J = coloumb-like integral (ia|jb)
  // K = exchange-like integral (ib|ja) or (ja|ib)
  // Note Same Spin     = J - K
  // NOTE Opposite Spin = J
  // Hence, Total En(2) = (fss+fos) J  - fss K
  //
  auto mhfss = -0.5*fss;
  auto hfss  =  0.5*fss;
  auto mfos  =  0.5*fos;
  
  // MO INTS (ia|jb)
  double* mo_tei = mo_ints(ints.TEI.memptr() );
  
  // 
  // T(im,am|jm,bm)
  // 2*(im,am|jm,bm) - (im,bm|jm,am)
  //
  
  for (auto im = 0; im < naocc; im++) {
    auto moi = iocc1 + im;
    
    for (auto jm = 0; jm < naocc; jm++) {
      auto moj = iocc1 + jm;
    
      for (auto bm = 0; bm < navir; bm++) {
	auto mob = ivir1 + bm;

	for (auto am = 0; am < navir; am++) {
	  auto moa = ivir1 + am;
	  
	  double den = eval(moi) + eval(moj) - eval(moa) - eval(mob);

	  auto mia = INDEX(moi,moa);
	  auto mjb = INDEX(moj,mob);
	  auto miajb = INDEX(mia,mjb);
	  auto mib = INDEX(moi,mob);
	  auto mja = INDEX(moj,moa);
	  auto mibja = INDEX(mib,mja);

	  auto o_iajb = mo_tei[miajb];
	  auto o_ibja = mo_tei[mibja];
	  auto t_iajb = ((fss+fos)*o_iajb - fos*o_ibja)/den;

	  emp2 += t_iajb * o_iajb;

	} // im

      } // am
    }  // jm
  }

  if (l_print)
    cout << "EMP2     " << emp2 << endl;

}


}  }  // namespace willow::qcmol
