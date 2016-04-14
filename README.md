# w-qcmol
A simple program to calculate the HF/MP2 energy.

I developed this program based on a file of "evaleev/libint/tests/hartree-fock/hartree-fock++.cc".
A key difference from "hartree-fock++.cc" is that I used "armadilo" for a linear algabra instead of Eigen.
This code will be used in w-pimd (or w-bomd) in the calculation of the partial atomic charges on a molecule 
via electrostatic potential scheme (esp.cc).
