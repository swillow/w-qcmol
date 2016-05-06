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

  arma::mat densityMatrix ();
  arma::mat energyWeightDensityMatrix ();

  double rhf_energy  () { return (E_nuc + E_hf);};
  double get_nocc() { return m_nocc;};
  EigenSolver eig_solver;
  
private:

  size_t m_nocc;
  double E_nuc;
  double E_hf;
  
};



} }  // namespace willow::qcmol


#endif
