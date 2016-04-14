#ifndef DIIS_HPP
#define DIIS_HPP

#include <armadillo>
#include <vector>

class DIIS {

public:
  DIIS() {}
  DIIS(const arma::mat& Sm, const arma::mat& SmInvh, const arma::mat& Fm) 
    : Sm_(Sm), SmInvh_(SmInvh), FmOld_(Fm)
  {
    //o_started_ = true;
    err_cutoff_ = 0.05;
  }

  arma::mat getF (arma::mat& Fm, arma::mat& Dm) 
  {
    // compute error matrix
    arma::mat FmNew(Fm.n_rows, Fm.n_cols);
    FmNew.zeros();
    arma::mat errMat = Fm*Dm*Sm_ - Sm_*Dm*Fm;
    // transform it to the orthonormal basis 
    errMat = SmInvh_.t()*errMat*SmInvh_;

    arma::vec err = arma::vectorise (errMat);
    
    max_err_ = arma::max(err);
    auto tmp = fabs(arma::min(err));
    
    if ( tmp > max_err_) 
      max_err_ = tmp;
    
    if ( max_err_ > err_cutoff_ ) {
      // Do simple averaging until DIIS starts
      // cout << "Simple Mixed " << endl;
      FmNew = 0.5*Fm + 0.5*FmOld_;
      FmOld_ = Fm;
    }
    else {
      // cout << "DIIS Mixed " << endl;
      FmList_.push_back(Fm);
      ErrList_.push_back(err);

      int N = ErrList_.size();
      // Original, Pulay's DIIS
      arma::mat B(N+1,N+1);
      B.zeros();
      // RHS vector
      arma::vec A(N+1);
      A.zeros();

      // Compute errors;
      for (auto i = 0; i < N; i++) 
	for (auto j = 0; j <= i; j++) {
	  B(i,j) = arma::dot(ErrList_[i], ErrList_[j]);
	  B(j,i) = B(i,j);
	}

      // Fill in the rest of B
      for (auto i = 0; i < N; i++) {
	B(N,i) = -1.0;
	B(i,N) = -1.0;
      }
      // Fill in A
      A(N) = -1.0;

      // Solve B*X = A
      arma::vec X;
      bool succ = arma::solve (X,B,A);

      if (succ) {
	// Form weighted Fock Matrix
	FmNew.zeros();
	for (auto i = 0; i < FmList_.size(); i++) 
	  FmNew += X(i)*FmList_[i];
      }
      else {
	FmNew = 0.5*Fm + 0.5*FmOld_;
      }
      
    }

    return FmNew;
    
  }
    
    
private:
  
  //bool o_started_;
  double err_cutoff_;
  double max_err_;
  arma::mat Sm_;
  arma::mat SmInvh_; // Half-inverse Overlap Matrix
  arma::mat FmOld_;
  
  vector<arma::mat> FmList_;
  vector<arma::vec> ErrList_;
  
};

#endif
