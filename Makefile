#use these variable to set if we will use mpi or not 

AR=ar
ARFLAGS=-qs
RANLIB=ranlib

LIBINT2PATH=/usr/local/libint/2.2.0-alpha
LIBINT2INCLUDES = -I$(LIBINT2PATH)/include -I$(LIBINT2PATH)/include/libint2

EXECUTABLE = qcmol.x

# change to icpc for Intel
CXX = g++
HOME = .
.SUFFIXES: .cc 


CFLAGS = -O3 -ffast-math    -march=native -std=c++11 $(LIBINT2INCLUDES)  
#CFLAGS = -g   -march=native -std=c++11 $(LIBINT2INCLUDES)  
#LIBS =  -L$(LIBINT2PATH)/lib -lint2 -larmadillo -lblas -llapack 
#LIBS =  -L$(LIBINT2PATH)/lib -lint2 -lopenblas
LIBS =  -L$(LIBINT2PATH)/lib -lint2  -larmadillo -lblas -llapack

#SRC = shellfuncs.cc Integrals.cc EigenSolver.cc RHF.cc \
#      RHFGrad.cc obara-saika.cc mp2.cc mp2ints.cc mp2grad.cc  cphf.cc \
#      main.cc

SRC = Integrals.cc RHF.cc RHFGrad.cc mp2ints.cc MP2.cc esp.cc cphf.cc main.cc

COBJ=$(SRC:.cc=.o)

#.C.o :
#	$(CXX) -fPIC $(FLAGS) $(OPT) -c $< -o $@
.cc.o :
	$(CXX) $(CFLAGS) -c $< -o $@

.f90.o :
	$(FC) $(FFLAGS) -c $< -o $@

all	: $(EXECUTABLE) 


$(EXECUTABLE) : $(COBJ)
	$(CXX)   $(FLAGS) -o  $(EXECUTABLE) $(COBJ) $(FOBJ) $(LIBS) 

hartree-fock++.x : hartree-fock++.o
	$(CXX)   $(FLAGS) -o  hartree-fock++.x hartree-fock++.o  $(LIBS) -lpthread

clean:
	rm *.o 

# DO NOT DELETE
