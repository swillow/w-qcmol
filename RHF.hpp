#ifndef RHF_HPP
#define RHF_HPP

#include <armadillo>
#include "EigenSolver.hpp"
#include "Integrals.hpp"



namespace willow { namespace qcmol {



class RHF
{
public:
  
  RHF(const std::vector<libint2::Atom>& atoms,
      const libint2::BasisSet& bs,
      const Integrals& ints,
      const int qm_chg = 0,
      const bool l_print = true,
      const std::vector<QAtom>& atoms_Q = std::vector<QAtom>() );

  arma::mat densityMatrix (const int nocc);
  arma::mat energyWeightDensityMatrix (const int nocc);

  double energy () { return (E_nuc + E_hf);};
  
  EigenSolver eig_solver;
  
  double E_nuc;
  double E_hf;
};



} }  // namespace willow::qcmol


#endif
