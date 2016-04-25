#ifndef MP2_HPP_INCLUDED
#define MP2_HPP_INCLUDED 1

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <armadillo>
#include "Molecule.hpp"
#include "Integrals.hpp"


namespace willow { namespace qcmol {


class MP2 {
public:

  MP2 (const std::vector<libint2::Atom>& atoms,
       const libint2::BasisSet& bs,
       const Integrals& ints,
       const int qm_chg = 0,
       const bool l_print = true,
       const std::vector<QAtom>& atoms_Q = std::vector<QAtom> ());

  double mp2_energy () { return emp2;};
  double rhf_energy () { return erhf;};
  
protected:
  int nocc;
  int ncore;
  int nmo;
  int nbf;

  int iocc1;
  int iocc2;
  int ivir1;
  int ivir2;

  int naocc;
  int navir;

  double fss, fos;
  
  // Eigen Values and Vectors
  arma::vec eval;
  arma::mat Cmat;

  double*   mo_ints (const double* ao_tei);

  double emp2;
  double erhf;
  
};


}  } // namespace willow::qcmol


#endif
