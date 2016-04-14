#ifndef RHF_HPP
#define RHF_HPP

#include <armadillo>
#include "Molecule.hpp"
#include "EigenSolver.hpp"
#include "Integrals.hpp"

class RHF
{
public:
  RHF(const vector<Atom>& atoms,
      const BasisSet& bs,
      const Integrals& ints);

  arma::mat densityMatrix (const int nocc);
  arma::mat energyWeightDensityMatrix (const int nocc);

  EigenSolver eig_solver;
  
  double E_nuc;
  double E_hf;
};

extern arma::mat g_matrix (double* TEI, arma::mat& D);

#endif
