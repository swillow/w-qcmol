# w-qcmol
A simple program to calculate the HF/MP2 energy.

I developed this program based on a file of "evaleev/libint/tests/hartree-fock/hartree-fock++.cc".
A key difference from "hartree-fock++.cc" is that I used \<armadillo\> for a linear algebra instead of Eigen.
This code will be used in w-pimd (or w-bomd) for the calculation of the partial atomic charges of a molecule 
(see esp.cc).
