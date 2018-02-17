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

    for (auto am = 0; am < navir; am++, iam++) {
      arma::vec mo_ia_uv = mo_hf_int.col(iam);
      
      for (auto ib = 0, ijb = 0; ib < nbf; ib++)
	for (auto jb = 0; jb <= ib; jb++, ijb++) {
	  double tmp = mo_ia_uv(ijb);
	  ao_uv(ib,jb) = tmp;
	  ao_uv(jb,ib) = tmp;
	}
      
      // (navir, nocc) = Cvir(nbf,navir) * (nbf,nbf) (nbf, nocc)
      m_mo_jb = Cvir.t()*ao_uv*Cocc; // OK

      result.col(iam) = v_mo_jb;
      
    } // am
  } // im

  return result;
}




arma::mat MP2::mo_half_ints_direct (const int im,
				    const libint2::BasisSet& bs,
				    const arma::mat& Schwartz)
{ // calculate (uv|am) =  (nbf, nbf, ivir1:ivir2, im)
  const auto nbf     = bs.nbf();
  const auto nshells = bs.size();
  const auto nints   = INDEX(nbf,0);
  
  const arma::mat Cocc = Cmat.submat (0, iocc1, nbf-1, iocc2);
  const arma::mat Cvir = Cmat.submat (0, ivir1, nbf-1, ivir2);

  arma::mat result (navir, nints, arma::fill::zeros);
  arma::mat tei_ui (nbf,   nints, arma::fill::zeros);

  // Construct the 2-electron repulsion integrals engine
  libint2::Engine engine = libint2::Engine(libint2::Operator::coulomb,
					   bs.max_nprim(),
					   bs.max_l(),
					   0);
  
  const auto precision = std::numeric_limits<double>::epsilon();
  engine.set_precision(precision);
  const auto& buf = engine.results();
  
  auto shell2bf = bs.shell2bf();

  for (auto s1 = 0; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];
    auto nbf1      = bs[s1].size();
    
    for (auto s2 = 0; s2 <= s1; ++s2) {
      
      auto s12_cut   = Schwartz(s1,s2);
      auto bf2_first = shell2bf[s2];
      auto nbf2      = bs[s2].size();

      for (auto s3 = 0; s3 != nshells; ++s3) {
	auto bf3_first = shell2bf[s3];
	auto nbf3      = bs[s3].size();

	const auto s4_max = s3;
	for (auto s4 = 0; s4 <= s4_max; ++s4) {
	  auto bf4_first = shell2bf[s4];
	  auto nbf4      = bs[s4].size();
	  auto s34_cut   = Schwartz(s3,s4);
	  if (s12_cut*s34_cut < precision) {
	    continue;
	  }

	  engine.compute2<libint2::Operator::coulomb,libint2::BraKet::xx_xx,0>
	    (bs[s1], bs[s2], bs[s3], bs[s4]);
	  const auto* buf0 = buf[0];
	 
	  if (buf0 == nullptr) continue;
	  
	  for (auto f1 = 0, f1234 = 0; f1 != nbf1; ++f1) {
	    const auto bf1 = f1 + bf1_first;
	    for (auto f2 = 0; f2 != nbf2; ++f2) {
	      const auto bf2 = f2 + bf2_first;
	      const auto ijc = INDEX(bf1, bf2);
	      
	      double tmp = 0.0;
	      for (auto f3 = 0; f3 != nbf3; ++f3) {
		const auto bf3 = f3 + bf3_first;
		for (auto f4 = 0; f4 != nbf4; ++f4, ++f1234) {
		  const auto bf4 = f4 + bf4_first;
		  const auto eri4 = buf0[f1234];

		  if (bf1 < bf2) continue;
		  
		  tei_ui(bf3,ijc) += eri4*Cocc(bf4,im);
		  if (bf3 > bf4 && s3 != s4) {
		    tei_ui(bf4,ijc) += eri4*Cocc(bf3,im);
		  }
		  
		}
	      }

	    } // f2
	  }   // f1
	  //----
	} // s4
      } // s3
    } // s2
  }   // s1

  for (auto ijc = 0; ijc < nints; ijc++) {
    result.col(ijc) = Cvir.t()*tei_ui.col(ijc);
  }
  
  // return (nints, navir)
  return result.t();
}


arma::mat MP2::mo_iajb_ints_direct (const arma::mat& mo_hf_int)
{

  arma::mat result (nocc*navir, navir);
  const arma::mat Cocc = Cmat.submat (0,     0, nbf-1, iocc2);
  const arma::mat Cvir = Cmat.submat (0, ivir1, nbf-1, ivir2);

  arma::vec v_mo_jb (nocc*navir, arma::fill::zeros);
  arma::mat m_mo_jb (v_mo_jb.memptr(), navir, nocc, false);
  arma::mat ao_uv (nbf, nbf);
  
  for (auto am = 0; am < navir; am++) {
    arma::vec mo_ia_uv = mo_hf_int.col(am);
    
    for (auto ib = 0, ijb = 0; ib < nbf; ib++)
      for (auto jb = 0; jb <= ib; jb++, ijb++) {
	double tmp = mo_ia_uv(ijb);
	ao_uv(ib,jb) = tmp;
	ao_uv(jb,ib) = tmp;
      }
      
    // (navir, nocc) = Cvir(nbf,navir) * (nbf,nbf) (nbf, nocc)
    m_mo_jb = Cvir.t()*ao_uv*Cocc; // OK
    
    result.col(am) = v_mo_jb;
      
  } // am

  return result;
}



} } // namespace willow::qcmol
