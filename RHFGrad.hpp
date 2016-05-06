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
  
  arma::vec grad;
  
};



}  }  // namespace willow::qcmol

#endif
