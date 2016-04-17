#ifndef RHF_HPP
#define RHF_HPP

#include <armadillo>
#include "Molecule.hpp"
#include "EigenSolver.hpp"
#include "Integrals.hpp"



namespace willow { namespace qcmol {



class RHF
{
public:
  
  RHF(const vector<Atom>& atoms,
      const BasisSet& bs,
      const Integrals& ints,
      const bool l_print = true);

  arma::mat densityMatrix (const int nocc);
  arma::mat energyWeightDensityMatrix (const int nocc);

  double energy () { return (E_nuc + E_hf);};
  
  EigenSolver eig_solver;
  
  double E_nuc;
  double E_hf;
};

extern arma::mat g_matrix (double* TEI, arma::mat& D);



} }  // namespace willow::qcmol


#endif
