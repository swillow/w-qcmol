#ifndef ESP_HPP
#define ESP_HPP

#include <libint2.hpp>
#include <vector>
#include <armadillo>
#include "Molecule.hpp"
#include "Integrals.hpp"
#include "RHF.hpp"

namespace willow { namespace qcmol {


class ESP : public RHF
{
  
public:
  ESP (const std::vector<libint2::Atom>& atoms,
       const libint2::BasisSet& bs,
       const Integrals& ints,
       const int qm_chg = 0,
       const bool l_print = true,
       const std::vector<QAtom>& atoms_Q = std::vector<QAtom>());

  arma::vec get_atomic_charges () { return m_qf; };
  
private:

  arma::vec m_qf;
  
  struct Grid {
    double x, y, z;
    double val;
  };
  
  const double ang2bohr = 1.0/0.52917721092; // 2010 CODATA value
  
  std::vector<Grid> grids;

  void esp_esp  (const std::vector<libint2::Atom>& atoms,
		 const libint2::BasisSet& bs,
		 const arma::mat& Dm);
  
  void esp_grid (const std::vector<libint2::Atom>& atoms);
  
  arma::vec esp_fit (const std::vector<libint2::Atom>& atoms);
  
};



} } // namespace willow::qcmol


#endif
