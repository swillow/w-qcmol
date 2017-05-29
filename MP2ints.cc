#include "RHF.hpp"
#include "MP2.hpp"


namespace willow { namespace qcmol {


arma::mat MP2::mo_half_ints (const double* ao_tei)
{ // calculate (ia|uv) (iocc1:iocc2,1:ivir2,nbf,nbf)
  const auto nints = INDEX(nbf,0);
  
  const arma::mat Cocc = Cmat.submat (0, iocc1, nbf-1, iocc2);
  const arma::mat Cvir = Cmat.submat (0, ivir1, nbf-1, ivir2);

  arma::mat result (naocc*navir, nints, arma::fill::zeros);
  arma::mat tei_uv (nbf, nbf, arma::fill::zeros);
  arma::vec mo_ia  (navir*naocc, arma::fill::zeros);
  arma::mat mo_ia_uv(mo_ia.memptr(), navir, naocc, false);
  
  for (auto ijc = 0; ijc < nints; ijc++) {

    tei_uv.zeros();

    for (auto kc = 0, klc = 0; kc < nbf; kc++) {
      for (auto lc = 0; lc <= kc; lc++, klc++) {
	
	auto ijklc = INDEX(ijc,klc);
	auto eri4  = ao_tei[ijklc];
	tei_uv(kc,lc) = eri4;
	tei_uv(lc,kc) = eri4;
      }
    }
    
    // Cocc^T (nbf,nocc) tei_uv(nbf,nbf) Cvir (nbf, navir)
    // (navir, nacc) 
    mo_ia_uv = Cvir.t()*tei_uv*Cocc;
    result.col(ijc) = mo_ia;
  }
  
  // return (nints, naocc*navir)
  return result.t();
  
}





arma::mat MP2::mo_iajb_ints (const arma::mat& mo_hf_int)
{

  arma::mat result (nocc*navir, naocc*navir);
  const arma::mat Cocc = Cmat.submat (0,     0, nbf-1, iocc2);
  const arma::mat Cvir = Cmat.submat (0, ivir1, nbf-1, ivir2);

  arma::vec v_mo_jb (nocc*navir, arma::fill::zeros);
  arma::mat m_mo_jb (v_mo_jb.memptr(), navir, nocc, false);
  arma::mat ao_uv (nbf, nbf);
  
  for (auto im = 0, iam = 0; im < naocc; im++) {
    auto moi = iocc1 + im;
    for (auto am = 0; am < navir; am++, iam++) {
      auto moa = ivir1 + am;

      arma::vec mo_ia_uv = mo_hf_int.col(iam);
      
      for (auto ib = 0, ijb = 0; ib < nbf; ib++)
	for (auto jb = 0; jb <= ib; jb++, ijb++) {
	  double tmp = mo_ia_uv(ijb);
	  ao_uv(ib,jb) = tmp;
	  ao_uv(jb,ib) = tmp;
	}
      
      // (navir, nocc) = Cvir(nbf,navir) * (nbf,nbf) (nbf, nocc)
      m_mo_jb = Cvir.t()*ao_uv*Cocc; // OK
/*      
      for (auto moj = iocc1; moj < nocc; moj++) {
	for (auto bm = 0; bm < navir; bm++) {
	  auto mob = ivir1 + bm;

	  double den = eval(moi) + eval(moj) - eval(moa) - eval(mob);
	  m_mo_jb(bm,moj) /= den;
	}
      }
*/     
      result.col(iam) = v_mo_jb;
      
    } // am
  } // im

  return result;
}


} } // namespace willow::qcmol
