#if __cplusplus <= 199711L
# error "Hartree-Fock test requires C++11 support"
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
#include "MP2.hpp"
#include "esp.hpp"
#include "mp2grad.hpp"
#include "RHFGrad.hpp"

using namespace std;
using namespace libint2;

int main (int argc, char *argv[]) 
{
  
  try {
    // Initialize molecule
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";

    vector<Atom> atoms = read_geometry (filename);
    //cout << mol << endl;

    // Create Basis Set
    libint2::Shell::do_enforce_unit_normalization(false);
    BasisSet bs  ("aug-cc-pVDZ", atoms);
    //shell_funcs_setup();
    
    // Integrations (one-electron, two-electron Integrals)
    //  Timer timer;
    //timer.start ("TWO ELECTRON INTEGRALS");
    Integrals ints (atoms, bs);
    //timer.end();

    //RHF rhf (atoms,bs,ints);
    //MP2 mp2 (atoms,bs,ints);
    //MP2Grad mp2 (atoms,bs,ints);
    //RHFGrad rhf_grad (atoms, bs, ints);
    ESP esp (atoms, bs, ints);
    
    //MP2Grad mp2 (mol, bs, ints);
    /*
    cout << "EHF   " << rhf.E_hf << endl;
    cout << "E_nuc " << rhf.E_nuc << endl;
    cout << "nocc  " << rhf.nocc << endl;
    cout << "nmol  " << rhf.nmol << endl;
    cout << "nbf   " << rhf.nbf  << endl;
    */

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
