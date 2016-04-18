#ifndef W_MOLECULE_HPP
#define W_MOLECULE_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <libint2.hpp>


namespace willow { namespace qcmol {



inline std::vector<libint2::Atom> read_geometry(const std::string& filename) {
  
  //std::cout << "Will read geometry from " << filename << std::endl;
  std::ifstream is(filename);
  assert(is.good());

  // to prepare for MPI parallelization, we will read the entire file into a string that can be
  // broadcast to everyone, then converted to an std::istringstream object that can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise throw an exception
  if ( filename.rfind(".xyz") != std::string::npos)
    return libint2::read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}


inline int num_electrons (const std::vector<libint2::Atom>& atoms)
{
  auto nelec = 0;
  for (auto i = 0; i < atoms.size(); ++i)
    nelec += atoms[i].atomic_number;

  return nelec;
  
}



inline int compute_ncore (const std::vector<libint2::Atom>& atoms)
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
