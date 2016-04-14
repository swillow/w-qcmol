#include "mp2grad.hpp"

#define INDEX(i,j) ( (i >= j) ? (((i)*((i)+1)/2) + (j)) : (((j)*((j)+1)/2) + (i)))

arma::mat MP2Grad::cphf_solve (const arma::mat& L_ai) 
{
  const double zthresh = 1.0e-4;
  const double zmaxiter = 100;
  
  const arma::mat Cocc = Cmat.submat(0,    0, nbf-1, iocc2);
  const arma::mat Cvir = Cmat.submat(0,ivir1, nbf-1, ivir2);
  
  arma::mat Pm(navir,nocc);
  Pm.zeros();

  for (auto a = 0; a < navir; ++a)  {
    auto moa = ivir1 + a;
    for (auto i = 0; i < nocc; ++i) {
      auto moi = i;
      Pm(a,i) = L_ai(a,i)/(eval(moa) - eval(moi));
    }
  }

  // == Z-vector iteraction ==
  for (auto iter = 0; iter != zmaxiter; ++iter) {
    // J part 4(ia|jb)
    // K part -(aj|bi) - (ab|ij)
    // back transform: v(nbf,nvirt) pvo(a,i) o(nbf,nmo)^t
    arma::mat Pao = Cvir*Pm*Cocc.t();
    arma::mat m_ao(nbf,nbf);
    cout  << "ITER " << iter << endl;

    for (auto i = 0; i < nbf; i++) 
      for (auto j = 0; j < nbf; j++) {
	
	auto sum = 0.0;

	for (auto k = 0; k < nbf; k++) {
	  for (auto l = 0; l < nbf; l++) {

	    auto ij = INDEX(i,j);
	    auto kl = INDEX(k,l);
	    
	    auto ijkl = INDEX(ij,kl);
	    auto eri4_j = ao_tei[ijkl];

	    auto il = INDEX(i,l);
	    auto kj = INDEX(k,j);
	    auto ilkj = INDEX(il,kj);
	    auto eri4_k1 = ao_tei[ilkj];

	    auto ik = INDEX(i,k);
	    auto jl = INDEX(j,l);
	    auto ikjl = INDEX(ik,jl);
	    auto eri4_k2 = ao_tei[ikjl];
	    
	    sum += (4.0*eri4_j - eri4_k1 - eri4_k2)*Pao(k,l);
	  }
	} // k

	m_ao(i,j) = sum;
      }
  
    // m_ao --> m_ai
    // m_ai(nvir,nocc) = Cvir(nbf,nvir)^T m_ao(nbf,nbf) Cocc(nbf,nocc)
    arma::mat m_ai = Cvir.t()*m_ao*Cocc;
    arma::mat Pold = Pm;

    // Iterative solution of Pm
    Pm.zeros();
    for (auto a = 0; a < navir; a++) {
      auto moa = ivir1+a;
      for (auto i = 0; i < nocc; i++) {
	auto moi = i;
	Pm(a,i) = (L_ai(a,i) - m_ai(a,i))/(eval(moa) - eval(moi));
      }
    }
    Pm = 0.3*Pm + 0.6*Pold;
    auto rms = norm (Pm - Pold);
    if (rms < 1.0e-4) break;
  }
  
  return Pm;

}



