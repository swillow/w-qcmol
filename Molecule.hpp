#ifndef W_MOLECULE_HPP
#define W_MOLECULE_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <libint2.hpp>


using namespace std;
using namespace libint2;


namespace willow { namespace qcmol {


struct QAtom {
  double charge;
  double x, y, z;
};


inline vector<Atom> read_geometry(const string& filename) {
  
  //std::cout << "Will read geometry from " << filename << std::endl;
  ifstream is(filename);
  assert(is.good());

  // to prepare for MPI parallelization, we will read the entire file into a string that can be
  // broadcast to everyone, then converted to an std::istringstream object that can be used just like std::ifstream
  ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise throw an exception
  if ( filename.rfind(".xyz") != string::npos)
    return read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}


inline int num_electrons (const vector<Atom>& atoms)
{
  auto nelec = 0;
  for (auto i = 0; i < atoms.size(); ++i)
    nelec += atoms[i].atomic_number;

  return nelec;
  
}


inline double compute_nuclear_repulsion_energy (const vector<Atom>& atoms)
{
  auto enuc = 0.0;
  //  cout << "N E " << atoms.size() << endl;
  
  for (auto i = 0; i < atoms.size(); ++i) {
    double qi = static_cast<double> (atoms[i].atomic_number);
    for (auto j = i+1; j < atoms.size(); ++j) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij*xij + yij*yij + zij*zij;
      auto r = sqrt(r2);
      double qj = static_cast<double> (atoms[j].atomic_number);
      
      enuc += qi*qj / r;
      //cout << i << " " << j << " " << enuc << endl;
    }
  }


  return enuc;
  
}


inline int compute_ncore (const vector<Atom>& atoms)
{
  auto ncore = 0;

  for (auto i = 0; i < atoms.size(); i++) {
    if (atoms[i].atomic_number >= 3 && atoms[i].atomic_number <= 10)
      ncore++;
  }

  return ncore;
  
}



}   } // namespace willow::qcmol



#endif
