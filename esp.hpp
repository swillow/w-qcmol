#ifndef ESP_HPP
#define ESP_HPP

#include <vector>
#include <armadillo>
#include "Molecule.hpp"
#include "Integrals.hpp"


class ESP
{
private:

  struct Grid {
    double x, y, z;
    double val;
  };
  
  const double ang2bohr = 1.0/0.52917721092; // 2010 CODATA value
  vector<Grid> grids;

  void esp_esp  (const vector<Atom>& atoms,
		 const BasisSet& bs,
		 const arma::mat& Dm);
  
  void esp_grid (const vector<Atom>& atoms);
  vector<double> esp_fit (const vector<Atom>& atoms);
  
public:
  ESP (const vector<Atom>& atoms,
       const BasisSet& bs,
       const Integrals& ints);
  
};

#endif
