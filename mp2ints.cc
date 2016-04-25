#include "RHF.hpp"
#include "MP2.hpp"


namespace willow { namespace qcmol {


// (ia|jb) = (bj|ai)
// store (ia <= jb)
double* MP2::mo_ints (const double* ao_tei)
{ // calculate (ia|jb) (iocc1:iocc2,1:ivir2,iocc1:iocc2,ivir1:ivir2)
  const auto nints = ((nmo*(nmo+1)/2)*((nmo*(nmo+1)/2)+1)/2);
  const auto nmod  = nbf%5;

  auto result = new double[nints];
  std::fill (result, result+nints,0.0);

  arma::cube ao_ijk (nbf, nbf, nbf);
  arma::mat  ao_ij  (nbf, nbf);
  arma::vec  ao_i   (nbf);
  
  for (auto moi = iocc1; moi <= iocc2; moi++) {

    ao_ijk.zeros();
    
    for (auto ic = 0; ic < nbf-nmod; ic+=5) {
      //---
      for (auto jc = 0; jc <= ic; jc++) {
	auto ijc = INDEX(ic,jc);
	
	for (auto kc = 0; kc <= ic; kc++ ) 
	  for (auto lc = 0; lc <= kc; lc += 2) {
	    
	    auto klc = INDEX(kc,lc);
	    
	    if (klc > ijc) continue;
	    
	    auto ijklc = INDEX(ijc,klc);
	    auto eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic,jc,kc) += eri4*Cmat(lc,moi);
	    if (kc > lc) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic,jc,lc) += eri4*Cmat(kc,moi);
	    }
	    if (ic > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic,kc) += eri4*Cmat(lc,moi);
	      
	      if (kc > lc) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic,lc) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc,ic) += eri4*Cmat(jc,moi);
	      if (ic > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc,jc) += eri4*Cmat(ic,moi);
	      }
	      if (kc > lc) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc,kc,ic) += eri4*Cmat(jc,moi);
		
		if (ic > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc,kc,jc) += eri4*Cmat(ic,moi);
		}
	      }
	    } // if (ijc != klc
	    
	    
	    //----------------------
	    auto lc1 = lc+1;
	    if (lc1 > kc) continue;
	    klc = INDEX(kc,lc1);
	    if (klc > ijc) continue;
	    
	    ijklc = INDEX(ijc,klc);
	    eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic,jc,kc) += eri4*Cmat(lc1,moi);
	    if (kc > lc1) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic,jc,lc1) += eri4*Cmat(kc,moi);
	    }
	    if (ic > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic,kc) += eri4*Cmat(lc1,moi);
	      
	      if (kc > lc1) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic,lc1) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc1,ic) += eri4*Cmat(jc,moi);
	      if (ic > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc1,jc) += eri4*Cmat(ic,moi);
	      }
	      if (kc > lc1) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc1,kc,ic) += eri4*Cmat(jc,moi);
		
		if (ic > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc1,kc,jc) += eri4*Cmat(ic,moi);
		}
	      }
	    } // if (ijc != klc
	  }
      } // jc
      

      //---
      auto ic1 = ic+1;
      for (auto jc = 0; jc <= ic1; jc++) {
	auto ijc = INDEX(ic1,jc);
	
	for (auto kc = 0; kc <= ic1; kc++ ) 
	  for (auto lc = 0; lc <= kc; lc += 2) {
	    
	    auto klc = INDEX(kc,lc);
	    
	    if (klc > ijc) continue;
	    
	    auto ijklc = INDEX(ijc,klc);
	    auto eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc,moi);
	    if (kc > lc) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc,moi);
	      
	      if (kc > lc) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	    
	    
	    //----------------------
	    auto lc1 = lc+1;
	    if (lc1 > kc) continue;
	    klc = INDEX(kc,lc1);
	    if (klc > ijc) continue;
	    
	    ijklc = INDEX(ijc,klc);
	    eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc1,moi);
	    if (kc > lc1) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc1) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc1,moi);
	      
	      if (kc > lc1) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc1) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc1,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc1,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc1) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc1,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc1,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	  }
      } // jc


      //---
      ic1 = ic+2;
      for (auto jc = 0; jc <= ic1; jc++) {
	auto ijc = INDEX(ic1,jc);
	
	for (auto kc = 0; kc <= ic1; kc++ ) 
	  for (auto lc = 0; lc <= kc; lc += 2) {
	    
	    auto klc = INDEX(kc,lc);
	    
	    if (klc > ijc) continue;
	    
	    auto ijklc = INDEX(ijc,klc);
	    auto eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc,moi);
	    if (kc > lc) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc,moi);
	      
	      if (kc > lc) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	    
	    
	    //----------------------
	    auto lc1 = lc+1;
	    if (lc1 > kc) continue;
	    klc = INDEX(kc,lc1);
	    if (klc > ijc) continue;
	    
	    ijklc = INDEX(ijc,klc);
	    eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc1,moi);
	    if (kc > lc1) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc1) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc1,moi);
	      
	      if (kc > lc1) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc1) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc1,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc1,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc1) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc1,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc1,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	  }
      } // jc


      //---
      ic1 = ic+3;
      for (auto jc = 0; jc <= ic1; jc++) {
	auto ijc = INDEX(ic1,jc);
	
	for (auto kc = 0; kc <= ic1; kc++ ) 
	  for (auto lc = 0; lc <= kc; lc += 2) {
	    
	    auto klc = INDEX(kc,lc);
	    
	    if (klc > ijc) continue;
	    
	    auto ijklc = INDEX(ijc,klc);
	    auto eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc,moi);
	    if (kc > lc) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc,moi);
	      
	      if (kc > lc) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	    
	    
	    //----------------------
	    auto lc1 = lc+1;
	    if (lc1 > kc) continue;
	    klc = INDEX(kc,lc1);
	    if (klc > ijc) continue;
	    
	    ijklc = INDEX(ijc,klc);
	    eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc1,moi);
	    if (kc > lc1) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc1) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc1,moi);
	      
	      if (kc > lc1) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc1) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc1,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc1,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc1) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc1,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc1,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	  }
      } // jc

      //---
      ic1 = ic+4;
      for (auto jc = 0; jc <= ic1; jc++) {
	auto ijc = INDEX(ic1,jc);
	
	for (auto kc = 0; kc <= ic1; kc++ ) 
	  for (auto lc = 0; lc <= kc; lc += 2) {
	    
	    auto klc = INDEX(kc,lc);
	    
	    if (klc > ijc) continue;
	    
	    auto ijklc = INDEX(ijc,klc);
	    auto eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc,moi);
	    if (kc > lc) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc,moi);
	      
	      if (kc > lc) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	    
	    
	    //----------------------
	    auto lc1 = lc+1;
	    if (lc1 > kc) continue;
	    klc = INDEX(kc,lc1);
	    if (klc > ijc) continue;
	    
	    ijklc = INDEX(ijc,klc);
	    eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic1,jc,kc) += eri4*Cmat(lc1,moi);
	    if (kc > lc1) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic1,jc,lc1) += eri4*Cmat(kc,moi);
	    }
	    if (ic1 > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic1,kc) += eri4*Cmat(lc1,moi);
	      
	      if (kc > lc1) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic1,lc1) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc1,ic1) += eri4*Cmat(jc,moi);
	      if (ic1 > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc1,jc) += eri4*Cmat(ic1,moi);
	      }
	      if (kc > lc1) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc1,kc,ic1) += eri4*Cmat(jc,moi);
		
		if (ic1 > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc1,kc,jc) += eri4*Cmat(ic1,moi);
		}
	      }
	    } // if (ijc != klc
	  }
      } // jc

    } // ic

    for (auto ic = nbf-nmod; ic < nbf; ic++) {

      for (auto jc = 0; jc <= ic; jc++) {
	auto ijc = INDEX(ic,jc);
	
	for (auto kc = 0; kc <= ic; kc++ ) 
	  for (auto lc = 0; lc <= kc; lc += 2) {
	    
	    auto klc = INDEX(kc,lc);
	    
	    if (klc > ijc) continue;
	    
	    auto ijklc = INDEX(ijc,klc);
	    auto eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic,jc,kc) += eri4*Cmat(lc,moi);
	    if (kc > lc) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic,jc,lc) += eri4*Cmat(kc,moi);
	    }
	    if (ic > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic,kc) += eri4*Cmat(lc,moi);
	      
	      if (kc > lc) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic,lc) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc,ic) += eri4*Cmat(jc,moi);
	      if (ic > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc,jc) += eri4*Cmat(ic,moi);
	      }
	      if (kc > lc) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc,kc,ic) += eri4*Cmat(jc,moi);
		
		if (ic > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc,kc,jc) += eri4*Cmat(ic,moi);
		}
	      }
	    } // if (ijc != klc
	    
	    
	    //----------------------
	    auto lc1 = lc+1;
	    if (lc1 > kc) continue;
	    klc = INDEX(kc,lc1);
	    if (klc > ijc) continue;
	    
	    ijklc = INDEX(ijc,klc);
	    eri4  = ao_tei[ijklc];
	    
	    // (ic,jc,|kc,lc) ---> (ic,jc,|kc,mob)
	    ao_ijk(ic,jc,kc) += eri4*Cmat(lc1,moi);
	    if (kc > lc1) {
	      // (ic,jc,|lc,kc) ---> (ic,jc,|kc,mob)
	      ao_ijk(ic,jc,lc1) += eri4*Cmat(kc,moi);
	    }
	    if (ic > jc) {
	      // (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
	      ao_ijk(jc,ic,kc) += eri4*Cmat(lc1,moi);
	      
	      if (kc > lc1) {
		// (jc,ic,|kc,lc) ---> (ic,jc,|kc,mob)
		ao_ijk(jc,ic,lc1) += eri4*Cmat(kc,moi);
	      }
	    }
	    
	    if (ijc != klc) {
	      
	      // (kc,lc,|ic,jc) ---> (kc,lc,|ic,mob)
	      ao_ijk(kc,lc1,ic) += eri4*Cmat(jc,moi);
	      if (ic > jc) {
		// (kc,lc,|jc,ic) ---> (kc,lc,|jc,mob)
		ao_ijk(kc,lc1,jc) += eri4*Cmat(ic,moi);
	      }
	      if (kc > lc1) {
		// (lc,kc,|ic,jc) ---> (lc,kc,|ic,mob)
		ao_ijk(lc1,kc,ic) += eri4*Cmat(jc,moi);
		
		if (ic > jc) {
		  // (lc,kc,|jc,ic) ---> (lc,kc,|jc,mob)
		  ao_ijk(lc1,kc,jc) += eri4*Cmat(ic,moi);
		}
	      }
	    } // if (ijc != klc
	  }
      } // jc
    }

    // DONE SAVE ao_ijk toward (mu,nu|kappa,moi)
    for (auto moa = ivir1; moa <= ivir2; moa++) {
      auto iam = INDEX(moi,moa);
      ao_ij.zeros();
      
      for (auto kc = 0; kc < nbf-nmod; kc += 5) {
	auto cval1 = Cmat(kc,moa);
	auto cval2 = Cmat(kc+1,moa);
	auto cval3 = Cmat(kc+2,moa);
	auto cval4 = Cmat(kc+3,moa);
	auto cval5 = Cmat(kc+4,moa);
	ao_ij += (cval1*ao_ijk.slice(kc) + cval2*ao_ijk.slice(kc+1) +
		  cval3*ao_ijk.slice(kc+2) + cval4*ao_ijk.slice(kc+3) +
		  cval5*ao_ijk.slice(kc+4));
      }
      for (auto kc = nbf-nmod; kc < nbf; kc++) {
	auto cval1 = Cmat(kc,moa);
	ao_ij += cval1*ao_ijk.slice(kc);
      }

      // done ao_ij : (mu, nu | moa, moi)

      //---
      for (auto moj = 0; moj < iocc1; moj++) {
	ao_i.zeros ();

	for (auto jc = 0; jc < nbf-nmod; jc += 5) {
	  auto cval1 = Cmat(jc,moj);
	  auto cval2 = Cmat(jc+1,moj);
	  auto cval3 = Cmat(jc+2,moj);
	  auto cval4 = Cmat(jc+3,moj);
	  auto cval5 = Cmat(jc+4,moj);
	  ao_i += (cval1*ao_ij.col(jc) + cval2*ao_ij.col(jc+1) + 
		   cval3*ao_ij.col(jc+2) + cval4*ao_ij.col(jc+3) +
		   cval5*ao_ij.col(jc+4) );
	}

	for (auto jc = nbf-nmod; jc < nbf; jc++) {
	  auto cval1 = Cmat(jc,moj);
	  ao_i += cval1*ao_ij.col(jc);
	}


	for (auto mob = ivir1; mob <= ivir2; mob++) {
	  auto jbm = INDEX(moj,mob);

	  auto iajbm = INDEX(iam,jbm);
	  
	  auto sum = 0.0;

	  for (auto ic = 0; ic < nbf - nmod; ic += 5) {
	    sum += (ao_i(ic)*Cmat(ic,mob) + ao_i(ic+1)*Cmat(ic+1,mob) +
		    ao_i(ic+2)*Cmat(ic+2,mob) + ao_i(ic+3)*Cmat(ic+3,mob) +
		    ao_i(ic+4)*Cmat(ic+4,mob) );
	  }

	  for (auto ic = nbf - nmod; ic < nbf; ic++) {
	    sum += ao_i(ic)*Cmat(ic,mob);
	  }

	  result[iajbm] = sum;
	}
  
      }
      
      for (auto moj = iocc1; moj <= iocc2; moj++) {

	ao_i.zeros ();
	
	for (auto jc = 0; jc < nbf-nmod; jc += 5) {
	  auto cval1 = Cmat(jc,moj);
	  auto cval2 = Cmat(jc+1,moj);
	  auto cval3 = Cmat(jc+2,moj);
	  auto cval4 = Cmat(jc+3,moj);
	  auto cval5 = Cmat(jc+4,moj);
	  ao_i += (cval1*ao_ij.col(jc) + cval2*ao_ij.col(jc+1) + 
		   cval3*ao_ij.col(jc+2) + cval4*ao_ij.col(jc+3) +
		   cval5*ao_ij.col(jc+4) );
	}

	for (auto jc = nbf-nmod; jc < nbf; jc++) {
	  auto cval1 = Cmat(jc,moj);
	  ao_i += cval1*ao_ij.col(jc);
	}

	// done ao_i : (mu, moj | moa, moi)
	for (auto mob = ivir1; mob <= moa; mob++) {
	  auto jbm = INDEX(moj,mob);
	  
	  if (jbm > iam) continue;

	  auto iajbm = INDEX(iam,jbm);
	  
	  auto sum = 0.0;

	  for (auto ic = 0; ic < nbf - nmod; ic += 5) {
	    sum += (ao_i(ic)*Cmat(ic,mob) + ao_i(ic+1)*Cmat(ic+1,mob) +
		    ao_i(ic+2)*Cmat(ic+2,mob) + ao_i(ic+3)*Cmat(ic+3,mob) +
		    ao_i(ic+4)*Cmat(ic+4,mob) );
	  }

	  for (auto ic = nbf - nmod; ic < nbf; ic++) {
	    sum += ao_i(ic)*Cmat(ic,mob);
	  }

	  result[iajbm] = sum;

	}
      }
    }
  }

  return result;

}



}  }  // namespace willow::qcmol

