#include "esp.hpp"
#include "RHF.hpp"
//
// Electrostatic Potential (ESP)
// : calculates partial atomic charges that fit
//   the quantum mechanical electrostatic potential on the selected grid points
//
ESP::ESP (const vector<Atom>& atoms,
	  const BasisSet& bs,
	  const Integrals& ints)
{

  RHF rhf (atoms, bs, ints);
  const auto nocc = num_electrons(atoms)/2;
  const arma::mat Dm = rhf.densityMatrix(nocc);

  esp_grid (atoms);
  
  esp_esp (atoms, bs, Dm);

  vector<double> qf = esp_fit (atoms);

  for (auto i = 0; i < atoms.size(); i++)
    cout << "ESP " << i+1 << "  " << qf[i] << endl;
  
}


void ESP::esp_esp (const vector<Atom>& atoms,
		   const BasisSet& bs,
		   const arma::mat& Dm)
{
  const auto nshells = bs.size();
  const auto nbf     = bs.nbf();
  
  libint2::OneBodyEngine engine (libint2::OneBodyEngine::nuclear,
				 bs.max_nprim(),
				 bs.max_l(),
				 0);

  const auto shell2bf = bs.shell2bf ();

  arma::mat Vm(nbf,nbf);
  
  for (auto ig = 0; ig < grids.size(); ++ig) {
    const auto xg = grids[ig].x;
    const auto yg = grids[ig].y;
    const auto zg = grids[ig].z;

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
	
	const auto* buf = engine.compute (bs[s1], bs[s2]);
	
	for (auto ib = 0, ij = 0; ib < nbf1; ib++) {
	  for (auto jb = 0; jb < nbf2; jb++, ij++) {
	    const double val = buf[ij];
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

    grids[ig].val = gval + zval;
  }
  
}
		     

vector<double> ESP::esp_fit (const vector<Atom>& atoms)
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

      for (auto k = 0; k < grids.size(); ++k) {
	auto xg = grids[k].x;
	auto yg = grids[k].y;
	auto zg = grids[k].z;

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
    for (auto k = 0; k < grids.size(); ++k) {
      auto xg = grids[k].x;
      auto yg = grids[k].y;
      auto zg = grids[k].z;
      auto val = grids[k].val;

      auto rig2 = (xi-xg)*(xi-xg) + (yi-yg)*(yi-yg) + (zi-zg)*(zi-zg);

      sum += val/sqrt(rig2);
    }
    bv(i) = sum;
  }
  bv(natoms) = 0.0; // b(natoms) = charge;

  arma::mat am_inv = am.i();
  
  vector<double> qf(natoms);
  
  for (auto i = 0; i < natoms; ++i) {
    auto sum = 0.0;

    for (auto j = 0; j < ndim; ++j) {
    //for (auto j = 0; j < atoms.size(); ++j) {
      sum = sum + am_inv(i,j)*bv(j);
    }
    qf[i] = sum;
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
  
  for(auto iz = 0; iz <  ngrid_z; ++iz)
    for(auto iy = 0; iy <  ngrid_y; ++iy)
      for(auto ix = 0; ix <  ngrid_x; ++ix) {
	
	double xg = grd_min_x - rcut + ix*spac;
	double yg = grd_min_y - rcut + iy*spac;
	double zg = grd_min_z - rcut + iz*spac;

	double dmin = rcut2;
	
	bool lupdate = true;
	
	for (auto ia = 0; ia < atoms.size(); ++ia) {
	  int iz = atoms[ia].atomic_number;
	  auto rad2 = radius[iz]*radius[iz];

	  auto x = atoms[ia].x;
	  auto y = atoms[ia].y;
	  auto z = atoms[ia].z;
	  
	  auto d2  = (xg-x)*(xg-x) + (yg-y)*(yg-y) + (zg-z)*(zg-z);

	  if ( (rad2 - d2) > small) {
	    lupdate = false;
	    break;
	  }

	  if ( (dmin-d2) > small) dmin = d2;
	  
	}

	if (lupdate) {
	  if ( (rcut2 - dmin) > small) {
	    grids.push_back ({xg,yg,zg,0.0});
	  }
	}
	
      }

}
