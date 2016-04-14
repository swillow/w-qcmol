#ifndef INTEGRALS_HPP
#define INTEGRALS_HPP

#include <vector>
#include <libint2.hpp>
//#include "Matrix.hpp"
#include <armadillo>

using namespace std;
using namespace libint2; // BasisSet, Atom

#define INDEX(i,j) ( (i >= j) ? (((i)*((i)+1)/2) + (j)) : (((j)*((j)+1)/2) + (i)))


class Integrals {
public:
  // Constructor
  Integrals (const vector<Atom>& atoms, const BasisSet& bs);
  // Destructor
  ~Integrals () {delete [] TEI;}

  // Data members
  arma::mat Sm;  // Overlap
  arma::mat Tm;  // Kinetic-energy
  arma::mat Vm;  // Nuclear-Attraction
  arma::mat SmInvh;  // Orthogonalize Basis
  arma::mat Km;  // Schwartz screening
  double* TEI;  // Store 2-e integrals
  
};

extern arma::mat compute_shellblock_norm(const BasisSet& obs,
				  const arma::mat& A);

#endif
