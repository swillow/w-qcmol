#ifndef RHFGRAD_HPP
#define RHFGRAD_HPP

#include <vector>
#include <libint2.h>
#include <armadillo>
//#include "Molecule.hpp"
#include "Integrals.hpp" // struct QAtom
#include "RHF.hpp"


namespace willow { namespace qcmol {



class RHFGrad : public RHF {
  
public:
  // Constructor
  RHFGrad (const std::vector<libint2::Atom>& atoms,
	   const libint2::BasisSet& bs,
	   const Integrals& ints,
	   const int qm_chg = 0,
	   const bool l_print = true,
	   const std::vector<QAtom>& atoms_Q = std::vector<QAtom>() );

  arma::vec get_rhf_grad ()   { return m_grad;}
  arma::vec get_rhf_grad_Q () { return m_grad_Q;}
  
private:
  
  arma::vec m_grad;
  arma::vec m_grad_Q;
  
};



}  }  // namespace willow::qcmol

#endif
