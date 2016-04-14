#ifndef MP2_HPP_INCLUDED
#define MP2_HPP_INCLUDED 1

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <armadillo>
#include "Molecule.hpp"
#include "Integrals.hpp"

class MP2 {
public:

  MP2 (const vector<Atom>& atoms,
       const BasisSet& bs,
       const Integrals& ints);

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
  double* ao_tei;
  double* mo_tei;
  // Eigen Values and Vectors
  arma::vec eval;
  arma::mat Cmat;

  double*   mo_ints ();

};


#endif
