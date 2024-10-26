#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <armadillo>
#include <libposthf/motran/motran_2e3c.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/arrays/memory_pool.h>
#include "ri_eomip_r.h"
#include <complex>

namespace libgmbpt{
using namespace libposthf;
using namespace libqints;
using namespace arma;
using namespace std;

namespace{
    double std_real(double x) { return x; }
    double std_real(const std::complex<double> &x) { return x.real(); }
    std::complex<double> std_conj(const std::complex<double> &x) 
    { 
        std::complex<double> z (x.real(), -x.imag());
        return z; 
    }
}

/// GPP: RI-EOMIP-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: OPTIMIZED CODE
template<>
void ri_eomip_r<double,double>::ccs_restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, Mat<double> &BQhp_a, 
    Mat<double> &BQoh_a, Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Row<double> &r1,
    Row<double> &residual, Col<double> &e_orb,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, rowvec &sigma) {


    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        
        exci = 0.0;       
        double t2ab = 0.0, t2ba = 0.0;
        double r_ijab = 0.0, r_ijba = 0.0;
 
        // intermediates
        arma::rowvec sigma_H (n_occ, fill::zeros);
        arma::rowvec sigma_I (n_occ, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        // iQ += BQov_a * t1;
        iQ += BQov_a * vectorise(t1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;

        /// step 4:
        #pragma omp parallel
        {
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj

                double delta_ij = e_orb(i) + e_orb(j);
                
                if(i == j) {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            //t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yai[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            //t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            //t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yai[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                // Yai[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                    }
                }
            }
            #pragma omp critical (Yai)
            {
                Yai += Yai_local;
            }
        }

        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        arma::Mat<double> sigma_0 = -r1 * E_oo;
        
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> BQi = BQoh * -r1.st(); // (n_aux*n_occ, n_occ)*n_occ
        BQi.reshape(n_aux,n_occ);

        #pragma omp parallel
        {
            arma::rowvec sigma_I_local (n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+b];
                        
                        //for(size_t Q = 0; Q < n_aux; Q++) {

                            //r_ijab += BQi[(i*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            //r_ijba += BQi[(j*n_aux+Q)]*BQhp_a[(i*n_vir*n_aux+b*n_aux+Q)];
                            
                        //}

                        //r_ijab = r_ijab / (delta_ijab + exci);
                        //r_ijba = r_ijba / (delta_ijab + exci);

                        
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar[(i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            // Y_bar_local[(i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            Y_bar_local[(i*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                        + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        // sigma_I(i) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        // sigma_I_local(i) += (2.0 * r_ijab - r_ijba) * Fov_hat(j,b);
                        sigma_I_local(i) += c_os * r_ijab * Fov_hat(j,b)
                                            + c_ss * (r_ijab - r_ijba) * Fov_hat(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
        }
        


        //transformed vector
        //arma::rowvec sigma (n_occ, fill::zeros);
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {

                // sigma_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        sigma_H(i) -= Y_bar[(k*n_aux+P)] * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }
                
                sigma(i) = sigma_0(i) + sigma_I(i) + sigma_H(i);
                // excit += (sigma(i)*r1(i)) / pow(norm(r1,"fro"),2);
            }
        } // end parallel

    }
}

template<>
void ri_eomip_r<double,double>::davidson_restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, Mat<double> &BQhp_a, 
    Mat<double> &BQoh_a, Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Row<double> &r1,
    Row<double> &residual, Col<double> &e_orb,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, rowvec &sigma) {


    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        
        // intermediates
        arma::rowvec sigma_H (n_occ, fill::zeros);
        arma::rowvec sigma_I (n_occ, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        // iQ += BQov_a * t1;
        iQ += BQov_a * vectorise(t1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;

        /// step 4:
        #pragma omp parallel
        {
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj

                double delta_ij = e_orb(i) + e_orb(j);
                double t2ab = 0.0;
                double t2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yai[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yai[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                // Yai[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                    }
                }
            }
            #pragma omp critical (Yai)
            {
                Yai += Yai_local;
            }
        }

        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        arma::Mat<double> sigma_0 = -r1 * E_oo;
        
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> BQi = BQoh * -r1.st(); // (n_aux*n_occ, n_occ)*n_occ
        BQi.reshape(n_aux,n_occ);

        #pragma omp parallel
        {
            arma::rowvec sigma_I_local (n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQi[(i*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            r_ijba += BQi[(j*n_aux+Q)]*BQhp_a[(i*n_vir*n_aux+b*n_aux+Q)];
                            
                        }

                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);

                        
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar[(i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            // Y_bar_local[(i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            Y_bar_local[(i*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                        + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        // sigma_I(i) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        // sigma_I_local(i) += (2.0 * r_ijab - r_ijba) * Fov_hat(j,b);
                        sigma_I_local(i) += c_os * r_ijab * Fov_hat(j,b)
                                            + c_ss * (r_ijab - r_ijba) * Fov_hat(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
        }
        


        //transformed vector
        //arma::rowvec sigma (n_occ, fill::zeros);
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {

                // sigma_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        sigma_H(i) -= Y_bar[(k*n_aux+P)] * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }
                
                sigma(i) = sigma_0(i) + sigma_I(i) + sigma_H(i);
            }
        } // end parallel
    }
}

template<>
void ri_eomip_r<double,double>::diis_restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, Mat<double> &BQhp_a, 
    Mat<double> &BQoh_a, Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Row<double> &r1,
    Row<double> &residual, Col<double> &e_orb,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, rowvec &sigma) {


    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        
        double excit=0.0;
        
        // intermediates
        arma::rowvec sigma_H (n_occ, fill::zeros);
        arma::rowvec sigma_I (n_occ, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        // iQ += BQov_a * t1;
        iQ += BQov_a * vectorise(t1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;

        /// step 4:
        #pragma omp parallel
        {
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj

                double delta_ij = e_orb(i) + e_orb(j);
                double t2ab = 0.0;
                double t2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yai[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yai[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                // Yai[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                    }
                }
            }
            #pragma omp critical (Yai)
            {
                Yai += Yai_local;
            }
        }

        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        arma::Mat<double> sigma_0 = -r1 * E_oo;
        
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> BQi = BQoh * -r1.st(); // (n_aux*n_occ, n_occ)*n_occ
        BQi.reshape(n_aux,n_occ);

        #pragma omp parallel
        {
            arma::rowvec sigma_I_local (n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQi[(i*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            r_ijba += BQi[(j*n_aux+Q)]*BQhp_a[(i*n_vir*n_aux+b*n_aux+Q)];
                            
                        }

                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);

                        
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar[(i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            // Y_bar_local[(i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            Y_bar_local[(i*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                        + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        // sigma_I(i) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        // sigma_I_local(i) += (2.0 * r_ijab - r_ijba) * Fov_hat(j,b);
                        sigma_I_local(i) += c_os * r_ijab * Fov_hat(j,b)
                                            + c_ss * (r_ijab - r_ijba) * Fov_hat(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
        }
        


        //transformed vector
        //arma::rowvec sigma (n_occ, fill::zeros);
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {

                // sigma_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        sigma_H(i) -= Y_bar[(k*n_aux+P)] * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }
                
                sigma(i) = sigma_0(i) + sigma_I(i) + sigma_H(i);
                // excit += (sigma(i)*r1(i)) / pow(norm(r1,"fro"),2);
            }
        }
        excit = as_scalar(sigma*r1.st()) / pow(norm(r1,"fro"),2);

        
        // update of the trial vector
        residual.zeros();
        arma::rowvec update (n_occ, fill::zeros);
        for(size_t i = 0; i < n_occ; i++) {
                
            double delta_i = e_orb(i);
            residual(i) = (sigma(i) - (excit*r1(i))) / norm(r1,"fro");
            update(i) = residual(i) / delta_i;
            r1(i) = (r1(i) + update(i)) / norm(r1,"fro");
                
        }
        
        exci = excit;
    }
}


#if 0
template<typename TC, typename TI>
void ri_eom_r<TC, TI>::restricted_energy(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, Mat<complex<double>> &BQho_a, 
    Mat<complex<double>> &BQoo_a, Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpv_a, 
    Mat<complex<double>> &BQpo_a, Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &V_Pab,  Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Mat<complex<double>> &residual, Col<complex<double>> &e_orb,
    array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3) {

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        
        complex<double> excit=(0.,0.);
        
        // intermediates
        arma::Mat<complex<double>> sigma_0 (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma_JG (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma_H (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma_I (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> E_vv (n_vir, n_vir, fill::zeros);
        arma::Mat<complex<double>> E_oo (n_occ, n_occ, fill::zeros);
        arma::Mat<complex<double>> Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::Mat<complex<double>> Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::Mat<complex<double>> Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        arma::Mat<complex<double>> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        arma::Mat<complex<double>> JG (n_orb, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> update (n_vir, n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::Col<complex<double>> iQ (n_aux, fill::zeros);
        arma::Col<complex<double>> iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * t1;
        iQ_bar += BQov_a * r1;

        // Fov_hat
        arma::Mat<complex<double>> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<complex<double>> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<complex<double>> Fov_hat1 = F11.st();
        arma::Mat<complex<double>> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<complex<double>> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<complex<double>> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<complex<double>> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<complex<double>> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<complex<double>> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<complex<double>> Fov_bar1 = F22.st();
        arma::Mat<complex<double>> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<complex<double>> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<complex<double>> Fov_bar = Fov_bar1 - Fov_bar2;

        // Fvv_hat
        arma::Mat<complex<double>> F3 = 2.0 * iQ.st() * BQpv_a;
        arma::Mat<complex<double>> F33(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<complex<double>> Fvv_hat1 = F33.st();
        arma::Mat<complex<double>> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<complex<double>> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<complex<double>> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<complex<double>> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<complex<double>> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<complex<double>> Foo_hat1 = F44.st();
        arma::Mat<complex<double>> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<complex<double>> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<complex<double>> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;


        /// step 4:
        for(size_t ij = 0; ij < npairs; ij++) {
            idx2 i2 = pairs[ij];
            size_t i = i2.i, j = i2.j;
            
            // for t2: 
            arma::Mat<complex<double>> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
            arma::Mat<complex<double>> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

            // for r2: 
            arma::Mat<complex<double>> Bhb_i(BQhb_a.colptr(i*n_vir), n_aux, n_vir, false, true);
            arma::Mat<complex<double>> Bhb_j(BQhb_a.colptr(j*n_vir), n_aux, n_vir, false, true);
            arma::Mat<complex<double>> Bbp_i(BQbp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
            arma::Mat<complex<double>> Bbp_j(BQbp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

            // integrals
            arma::Mat<complex<double>> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
            arma::Mat<complex<double>> W1 = Bhb_i.st() * Bhp_j; // r2:   aibj
            arma::Mat<complex<double>> W2 = Bhb_j.st() * Bhp_i; // r2:   bjai
            arma::Mat<complex<double>> W3 = Bbp_i.st() * Bhp_j; // r2:   aibj
            arma::Mat<complex<double>> W4 = Bbp_j.st() * Bhp_i; // r2:   bjai
            
            complex<double> delta_ij = e_orb(i) + e_orb(j);
            complex<double> t2ab(0.,0.), t2ba(0.,0.);
            complex<double> r2ab(0.,0.), r2ba(0.,0.);
            
            if(i == j) {
                const complex<double> *w0 = W0.memptr();
                const complex<double> *w1 = W1.memptr();
                const complex<double> *w2 = W2.memptr();
                const complex<double> *w3 = W3.memptr();
                const complex<double> *w4 = W4.memptr();

                for(size_t b = 0; b < n_vir; b++) {
                    
                    const complex<double> *w0b = w0 + b * n_vir;
                    const complex<double> *w1b = w1 + b * n_vir;
                    const complex<double> *w2b = w2 + b * n_vir;
                    const complex<double> *w3b = w3 + b * n_vir;
                    const complex<double> *w4b = w4 + b * n_vir;

                    complex<double> dijb = delta_ij - e_orb[n_occ+b];

                    for(size_t a = 0; a < n_vir; a++) {
                        complex<double> denom_t = dijb - e_orb[n_occ+a];
                        complex<double> denom_r = denom_t + exci;
                        t2ab = (conj(denom_t) * w0b[a]) / (conj(denom_t) * denom_t);
                        r2ab = (conj(denom_r) * (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b])) 
				                / (conj(denom_r) * denom_r);
                              
                        for(size_t Q = 0; Q < n_aux; Q++) {
	                        Yia[(a*n_occ*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
	                        Yai[(i*n_vir*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
	                        Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
	                    }

                        sigma_I(a,i) += (r2ab * Fov_hat(j,b)) + (t2ab * Fov_bar(j,b));
                        
                    }
                }

            } else {
                const complex<double> *w0 = W0.memptr();
                const complex<double> *w1 = W1.memptr();
                const complex<double> *w2 = W2.memptr();
                const complex<double> *w3 = W3.memptr();
                const complex<double> *w4 = W4.memptr();

                for(size_t b = 0; b < n_vir; b++) {
                    
                    const complex<double> *w0b = w0 + b * n_vir;
                    const complex<double> *w1b = w1 + b * n_vir;
                    const complex<double> *w2b = w2 + b * n_vir;
                    const complex<double> *w3b = w3 + b * n_vir;
                    const complex<double> *w4b = w4 + b * n_vir;

                    complex<double> dijb = delta_ij - e_orb[n_occ+b];

                    for(size_t a = 0; a < n_vir; a++) {
                        complex<double> denom_t = dijb - e_orb[n_occ+a];
                        complex<double> denom_r = denom_t + exci;
                        t2ab = (conj(denom_t) * w0b[a]) / (conj(denom_t) * denom_t);
                        t2ba = (conj(denom_t) * w0[a*n_vir+b]) / (conj(denom_t) * denom_t);

			            r2ab = (conj(denom_r) * (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b])) 
				                / (conj(denom_r) * denom_r);
                        r2ba = (conj(denom_r) * (w1[a*n_vir+b] + w2b[a] + w3[a*n_vir+b] + w4b[a])) 
				                / (conj(denom_r) * denom_r);
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {
	                        Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
	                        Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
	                        Yai[(i*n_vir*n_aux+a*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
	                        Yai[(j*n_vir*n_aux+b*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
	                        Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
	                        Y_bar[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
	                    }
                        
                        sigma_I(a,i) += ((2.0*r2ab-r2ba) * Fov_hat(j,b)) + ((2.0*t2ab-t2ba) * Fov_bar(j,b));
                        sigma_I(b,j) += ((2.0*r2ab-r2ba) * Fov_hat(i,a)) + ((2.0*t2ab-t2ba) * Fov_bar(i,a));
                    }
                }
            }
        }

        arma::Mat<complex<double>> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<complex<double>> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<complex<double>> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        sigma_0 += (E_vv*r1) - (r1*E_oo);


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<complex<double>> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<complex<double>> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<complex<double>> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<complex<double>> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<complex<double>> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<complex<double>> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<complex<double>> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<complex<double>> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<complex<double>> gamma_Q = gamma_G + gamma_J1 - gamma_J2;

        // V_PQ^(-1/2)
        arma::Mat<TI> PQinvhalf(arrays<TI>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        gamma_P = PQinvhalf * gamma_Q;
        //arma::Mat<double> gamma_P_re = (real(PQinvhalf) * real(gamma_Q)) - (imag(PQinvhalf) * imag(gamma_Q));
        //arma::Mat<double> gamma_P_im = (real(PQinvhalf) * imag(gamma_Q)) + (imag(PQinvhalf) * real(gamma_Q));
	
        //gamma_P.set_real(gamma_P_re);
        //gamma_P.set_imag(gamma_P_im);	

        for(size_t i = 0; i < n_occ; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                           * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }


        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        

	// c-product
	arma::Mat<double> r1_real = real(r1);
    arma::Mat<double> r1_imag = imag(r1);
    double cnorm_real = dot(r1_real,r1_real) - dot(r1_imag,r1_imag);
	double cnorm_imag = dot(r1_real,r1_imag) + dot(r1_real,r1_imag);
	complex<double> cnorm(cnorm_real, cnorm_imag);

	//transformed vector
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                
                // sigma_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        sigma_H(a,i) += Y_bar[(a*n_occ*n_aux+k*n_aux+P)]
                                            * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }
    
                sigma(a,i) = sigma_0(a,i) + sigma_JG(a,i) - sigma_H(a,i) + sigma_I(a,i);
                excit += ((conj(cnorm)*conj(cnorm)) * (sigma(a,i)*r1(a,i)))  / ((conj(cnorm)*conj(cnorm)) * (cnorm*cnorm));

            }
        }
        
        
        // update of the trial vector
        residual.zeros();
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                
                complex<double> delta_ia = e_orb(i) - e_orb[n_occ+a];
                residual(a,i) = (conj(cnorm) * (sigma(a,i) - (excit*r1(a,i)))) / (conj(cnorm) * cnorm);
		update(a,i) = (conj(delta_ia) * residual(a,i)) / (conj(delta_ia) * delta_ia);
		r1(a,i) = (conj(cnorm) * (r1(a,i) + update(a,i))) / (conj(cnorm) * cnorm);
            }
        }

	exci = excit;
    }

}

#endif


template class ri_eomip_r<double, double>;
template class ri_eomip_r<complex<double>, double>;
template class ri_eomip_r<complex<double>, complex<double>>;

}
