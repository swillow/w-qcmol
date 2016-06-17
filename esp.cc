#include <thread>
#include <vector>
#include <armadillo>
#include "RHF.hpp"
#include "esp.hpp"

//
// Electrostatic Potential (ESP)
// : calculates partial atomic charges that fit
//   the quantum mechanical electrostatic potential on the selected grid points
//

using namespace std;
using namespace libint2;

namespace willow { namespace qcmol {


static unsigned int num_threads;

static void esp_esp (const int thread_id,
		     const vector<Atom>& atoms,
		     const BasisSet& bs,
		     const arma::mat& Dm,
		     const vector<arma::vec3>& grids,
		     arma::vec& results);



ESP::ESP (const vector<Atom>& atoms,
	  const BasisSet& bs,
	  const Integrals& ints,
	  const int qm_chg,
	  const bool l_print,
	  const vector<QAtom>& atoms_Q)
  : RHF (atoms, bs, ints, qm_chg, l_print, atoms_Q)
{

  num_threads = std::thread::hardware_concurrency();

  if (num_threads == 0)
    num_threads = 1;

  const arma::mat Dm = densityMatrix();

  esp_grid (atoms);

  // ---- Thread
  vector<arma::vec>   grid_t(num_threads);
  vector<std::thread> t_esp (num_threads);

  for (auto id = 0; id < num_threads; ++id) {
    t_esp[id] = std::thread (esp_esp, id,
			     std::cref(atoms),
			     std::cref(bs),
			     std::cref(Dm),
			     std::cref(m_grids),
			     std::ref(grid_t[id]));
  }
  
  // Join
  for (auto id = 0; id < num_threads; ++id) {
    t_esp[id].join();
  }

  m_grids_val = grid_t[0];

  for (auto id = 1; id < num_threads; ++id) {
    m_grids_val += grid_t[id];
  }
  //---- (done: Thread) ---

  
  m_qf = esp_fit (atoms);
  
  if (l_print) {
    for (auto i = 0; i < atoms.size(); i++)
      cout << "ESP " << i+1 << "  " << m_qf(i) << endl;
  }

  
}


void esp_esp (const int thread_id,
	      const vector<Atom>& atoms,
	      const BasisSet& bs,
	      const arma::mat& Dm,
	      const vector<arma::vec3>& grids,
	      arma::vec& result)
{

  const auto nsize   = grids.size();

  result = arma::vec(nsize, arma::fill::zeros);
  
  const auto nshells = bs.size();
  const auto nbf     = bs.nbf();
  
  libint2::Engine engine (libint2::Operator::nuclear,
			  bs.max_nprim(),
			  bs.max_l(),
			  0);

  const auto& buf = engine.results();
  
  const auto shell2bf = bs.shell2bf ();

  arma::mat Vm(nbf,nbf);
  
  for (auto ig = 0; ig < grids.size(); ++ig) {
    
    if (ig % num_threads != thread_id) continue;
    
    const auto xg = grids[ig](0);
    const auto yg = grids[ig](1);
    const auto zg = grids[ig](2);

    // Interaction with electron density
    vector<pair<double,array<double,3>>> q;
    q.push_back ( {1.0, {{xg, yg, zg}}} );

    engine.set_params (q);
	
    // calc Vm
    Vm.zeros();
    
    for (auto s1 = 0; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1];
      auto nbf1 = bs[s1].size();

      for (auto s2 = 0; s2 != nshells; ++s2) {
	auto bf2 = shell2bf[s2];
	auto nbf2 = bs[s2].size();
	
	engine.compute (bs[s1], bs[s2]);
	const auto* buf0 = buf[0];
	
	for (auto ib = 0, ij = 0; ib < nbf1; ib++) {
	  for (auto jb = 0; jb < nbf2; jb++, ij++) {
	    const double val = buf0[ij];
	    Vm(bf1+ib,bf2+jb) = val;
	    Vm(bf2+jb,bf1+ib) = val;
	  }
	} // ib
	
      } // s2
    }  // s1

    //
    // get electrostatic potential on the grid points
    // -- from electron density
    double gval = 0.0;
    
    for (auto i = 0; i < nbf; i++)
      for (auto j = 0; j < nbf; j++)
	gval += Dm(i,j)*Vm(i,j);
    
    // get electrostatic potential on the grid points
    // -- from nuclei
    auto zval = 0.0;
    
    for (auto ia = 0; ia < atoms.size(); ++ia) {
      auto xij = atoms[ia].x - xg;
      auto yij = atoms[ia].y - yg;
      auto zij = atoms[ia].z - zg;
      auto r2  = xij*xij + yij*yij + zij*zij;
      auto r   = sqrt(r2);
      zval += static_cast<double>(atoms[ia].atomic_number)/r;
    }

    result(ig) = gval + zval;
  }
  
}
		     

arma::vec ESP::esp_fit (const vector<Atom>& atoms)
{

  // set up matrix of linear coefficients
  auto natoms = atoms.size();
  auto ndim   = natoms+1;

  arma::mat am(ndim,ndim);
  arma::vec bv(ndim);
  
  am.zeros();
  bv.zeros();
  
  for (auto i = 0; i < atoms.size(); ++i) {
    auto xi = atoms[i].x;
    auto yi = atoms[i].y;
    auto zi = atoms[i].z;
    
    for (auto j = 0; j < atoms.size(); ++j) {
      auto xj = atoms[j].x;
      auto yj = atoms[j].y;
      auto zj = atoms[j].z;
      
      auto sum = 0.0;

      for (auto k = 0; k < m_grids.size(); ++k) {
	auto xg = m_grids[k](0);
	auto yg = m_grids[k](1);
	auto zg = m_grids[k](2);

	auto rig2 = (xi-xg)*(xi-xg) + (yi-yg)*(yi-yg) + (zi-zg)*(zi-zg);
	auto rjg2 = (xj-xg)*(xj-xg) + (yj-yg)*(yj-yg) + (zj-zg)*(zj-zg);
	  
	sum += 1.0/sqrt(rig2*rjg2);
      }

      am(i,j) = sum;
      am(j,i) = sum;
    }
    
    am(i,natoms) = 1.0;
    am(natoms,i) = 1.0;
  }

  // construct column vector b

  for (auto i = 0; i < atoms.size(); ++i) {
    auto xi = atoms[i].x;
    auto yi = atoms[i].y;
    auto zi = atoms[i].z;

    auto sum = 0.0;
    for (auto k = 0; k < m_grids.size(); ++k) {
      auto xg = m_grids[k](0);
      auto yg = m_grids[k](1);
      auto zg = m_grids[k](2);
      auto val = m_grids_val(k);

      auto rig2 = (xi-xg)*(xi-xg) + (yi-yg)*(yi-yg) + (zi-zg)*(zi-zg);

      sum += val/sqrt(rig2);
    }
    bv(i) = sum;
  }
  bv(natoms) = 0.0; // b(natoms) = charge;

  arma::mat am_inv = am.i();
  
  arma::vec qf(natoms);
  qf.zeros();
  
  for (auto i = 0; i < natoms; ++i) {
    auto sum = 0.0;

    for (auto j = 0; j < ndim; ++j) {
    //for (auto j = 0; j < atoms.size(); ++j) {
      sum = sum + am_inv(i,j)*bv(j);
    }
    qf(i) = sum;
    //cout << " qf " << sum << endl;
  }

  return qf;
  
}



void ESP::esp_grid (const vector<Atom>& atoms)
{

  vector<double> radius(11);
  
  // A
  radius[0] = 0.0; // Ghost
  radius[1] = 0.3; // H
  radius[2] = 1.22;// He
  radius[3] = 1.23;// Li
  radius[4] = 0.89; // Be
  radius[5] = 0.88; // B
  radius[6] = 0.77; // C
  radius[7] = 0.70; // N
  radius[8] = 0.66; // O
  radius[9] = 0.58; // F
  radius[10]= 1.60; // Ne

  for (auto i = 1; i <= 10; i++) {
    radius[i] = (radius[i] + 0.7)*ang2bohr;
  }
  double grd_min_x = atoms[0].x;
  double grd_min_y = atoms[0].y;
  double grd_min_z = atoms[0].z;
  
  double grd_max_x = atoms[0].x;
  double grd_max_y = atoms[0].y;
  double grd_max_z = atoms[0].z;
  
  for (auto ia = 1; ia < atoms.size(); ++ia) {

    grd_min_x = min(grd_min_x, atoms[ia].x);
    grd_min_y = min(grd_min_y, atoms[ia].y);
    grd_min_z = min(grd_min_z, atoms[ia].z);

    grd_max_x = max(grd_max_x, atoms[ia].x);
    grd_max_y = max(grd_max_y, atoms[ia].y);
    grd_max_z = max(grd_max_z, atoms[ia].z);

  }
  //
  // calculate the grid size
  //
  // A --> au
  const double rcut = 3.0*ang2bohr;
  const double rcut2= rcut*rcut;
  const double spac = 0.5*ang2bohr;
  
  const int ngrid_x = int ((grd_max_x - grd_min_x + 2.0*rcut)/spac) + 1;
  const int ngrid_y = int ((grd_max_y - grd_min_y + 2.0*rcut)/spac) + 1;
  const int ngrid_z = int ((grd_max_z - grd_min_z + 2.0*rcut)/spac) + 1;

  const auto small = 1.0e-8;
  arma::vec3 vg;
  
  for(auto iz = 0; iz <  ngrid_z; ++iz)
    for(auto iy = 0; iy <  ngrid_y; ++iy)
      for(auto ix = 0; ix <  ngrid_x; ++ix) {

	vg.zeros();
	
	vg(0) = grd_min_x - rcut + ix*spac;
	vg(1) = grd_min_y - rcut + iy*spac;
	vg(2) = grd_min_z - rcut + iz*spac;

	double dmin = rcut2;
	
	bool lupdate = true;
	
	for (auto ia = 0; ia < atoms.size(); ++ia) {
	  int iz    = atoms[ia].atomic_number;
	  auto rad2 = radius[iz]*radius[iz];

	  auto dx = vg(0) - atoms[ia].x;
	  auto dy = vg(1) - atoms[ia].y;
	  auto dz = vg(2) - atoms[ia].z;
	  
	  auto d2  = dx*dx + dy*dy + dz*dz;

	  if ( (rad2 - d2) > small) {
	    lupdate = false;
	    break;
	  }

	  if ( (dmin-d2) > small) dmin = d2;
	  
	}

	if (lupdate) {
	  if ( (rcut2 - dmin) > small) {
	    m_grids.push_back (vg);
	  }
	}
	
      }

}


} } // namespace willow::qcmol
