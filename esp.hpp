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
  
  const double ang2bohr = 1.0/0.52917721067; // 2014 CODATA value
  
  std::vector<arma::vec3> m_grids;
  arma::vec               m_grids_val;
  
  void esp_grid (const std::vector<libint2::Atom>& atoms);
  
  arma::vec esp_fit (const std::vector<libint2::Atom>& atoms);
  
};



} } // namespace willow::qcmol


#endif
