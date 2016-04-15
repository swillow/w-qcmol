#ifndef EIGEN_SOLVER_H
#define EIGEN_SOLVER_H

#include <cassert>
#include <armadillo>



namespace willow { namespace qcmol {


class EigenSolver
{
public:

  EigenSolver (): m_is_init(false), m_eig_vals(), m_eig_vecs() {}

  // Sinvh.n_rows = nbf
  // Sinvh.n_cols = nmo
  // F.n_rows = nbf
  // F.n_cols = nbf
  // solve WF(nmo, nmo) = Sinv^T(nmo,nbf) F(nbf,nbf) Sinv(nbf,nmo)
  // dsyevd(WF) or eig_sym ()
  // save m_eig_vecs (nbf,nmo) = Sinvh(nbf,nmo)*Cvec(nmo,nmo)
  //
  EigenSolver  (const arma::mat Sinvh, const arma::mat F):
    m_is_init(true),
    m_eig_vals(Sinvh.n_cols),
    m_eig_vecs(Sinvh.n_rows, Sinvh.n_cols) {

    compute (Sinvh, F);

  }
    
  void compute (const arma::mat Sinvh, const arma::mat F) {
    
    const int nbf = Sinvh.n_rows;
    const int nmo = Sinvh.n_cols;
    
    if (!m_is_init) {
      m_is_init = true;
      m_eig_vals.set_size (nmo);
      m_eig_vecs.set_size (nbf, nmo);
    }

    // F' 
    arma::mat wf = Sinvh.t()*F*Sinvh; // (nmo, nmo)

    // Update Orbitals and Energies
    arma::mat eg_vecs;
    bool ok = arma::eig_sym (m_eig_vals, eg_vecs, wf);

    // Transform back to non-orthogonal basis
    m_eig_vecs = Sinvh*eg_vecs; // (nbf, nmol)
    
  }

  arma::vec eigenvalues() const {
    assert (m_is_init);
    return m_eig_vals;
  }
  
  arma::mat eigenvectors() const {
    assert (m_is_init);
    return m_eig_vecs;
  }

protected:
  bool      m_is_init;
  arma::vec m_eig_vals;
  arma::mat m_eig_vecs;
};



} } // namespace willow::qcmol

#endif
