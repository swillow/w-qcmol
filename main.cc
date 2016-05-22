#if __cplusplus <= 199711L
# error "QCMOL requires C++11 support"
#endif

// standard C++ headers
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <armadillo>
#include "Molecule.hpp"
#include "Integrals.hpp"
#include "RHF.hpp"
#include "RHFGrad.hpp"
#include "MP2.hpp"
#include "esp.hpp"

using namespace std;
using namespace libint2;
using namespace willow;

int main (int argc, char *argv[]) 
{
  
  try {
    // Initialize molecule

    libint2::initialize ();

    cout << std::setprecision(6);
    cout << std::fixed;
    
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";

    vector<Atom> atoms = qcmol::read_geometry (filename);
    //--
    vector<qcmol::QAtom> atoms_Q;
    qcmol::QAtom qatom;
    qatom.charge = 0.5;
    qatom.x      = 3.0/0.52917721067;
    qatom.y      = 0.0;
    qatom.z      = 0.0;
    atoms_Q.push_back(qatom);
    //----
    
    // Create Basis Set
    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::BasisSet bs  ("aug-cc-pVDZ", atoms);
    
    
    qcmol::Integrals ints (atoms, bs, atoms_Q);
    

    //qcmol::RHF rhf (atoms,bs,ints);
    qcmol::RHFGrad rhf (atoms,bs,ints, 0, true, atoms_Q);
    //qcmol::MP2 mp2 (atoms,bs,ints);
    //qcmol::ESP esp (atoms, bs, ints);

    libint2::finalize ();

  } // end of try block;
  
  catch (const char* ex) {
    cerr << "Caught exception: " << ex << endl;
    return 1;
  }
  catch (string& ex) {
    cerr << "Caught exception: " << ex << endl;
    return 1;
  }
  catch (exception& ex) {
    cerr << ex.what() << endl;
    return 1;
  }
  catch (...) {
    cerr << "Caught Unknown Exception: " << endl;
    return 1;
  }

  return 0;

}
