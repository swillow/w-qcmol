#ifndef W_INTEGRALS_HPP
#define W_INTEGRALS_HPP

#include <vector>
#include <libint2.hpp>
#include <armadillo>
#include "Molecule.hpp"


using namespace std;
using namespace libint2; // BasisSet, Atom


namespace willow { namespace qcmol {


#define INDEX(i,j) ( (i >= j) ? (((i)*((i)+1)/2) + (j)) : (((j)*((j)+1)/2) + (i)))


class Integrals {
public:
  // Constructor
  Integrals (const vector<Atom>& atoms, const BasisSet& bs,
	     const vector<QAtom>& Q_atoms=vector<QAtom>() );
  
  // Destructor
  ~Integrals () { if ( ! l_eri_direct) delete [] TEI;}

  // Data members
  arma::mat Sm;  // Overlap
  arma::mat Tm;  // Kinetic-energy
  arma::mat Vm;  // Nuclear-Attraction
  arma::mat SmInvh;  // Orthogonalize Basis
  arma::mat Km;  // Schwartz screening
  double* TEI;  // Store 2-e integrals
  bool    l_eri_direct; // 
  
};

extern arma::mat compute_shellblock_norm(const BasisSet& obs,
					 const arma::mat& A);

extern arma::mat compute_2body_fock (const BasisSet& bs,
				     const arma::mat& D,
				     const arma::mat& Schwartz);

extern arma::mat compute_2body_fock_general (const BasisSet& bs,
					     const arma::mat& Dm,
					     const BasisSet& Dm_bs,
					     bool Dm_is_diagonal=false);


} } // namespace willow::qcmol


#endif
