#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <armadillo>
#include <libposthf/motran/motran_2e3c.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/arrays/memory_pool.h>
#include <libqints/batch/bat_2e3c_shellpair_cgto.h>
#include <libqints/digestors/dig_passthru_2e3c_shellpair_cgto.h> // for (\mu\nu|Q)
#include <libqints/algorithms/gto/gto_pack.h>
#include <libgmbpt/util/dig_2e3c.h>
#include <libgmbpt/util/dig_2e3c_aux.h>
#include <libgmbpt/util/scr_2e3c.h>
#include "ri_eomee_sing_r.h"
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

/// GPP: RI-EOM-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: OPTIMIZE CODE
template<>
void ri_eomee_r<double,double>::ccs_restricted_energy_singlets(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &BQpv_a, Mat<double> &V_Pab,  
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Mat<double> &sigma) {
    

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        exci = 0; 
        double t2ab = 0.0, t2ba = 0.0;
        double r2ab = 0.0, r2ba = 0.0;
        // intermediates
        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);
        iQ_bar += BQov_a * vectorise(r1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = Fov_bar1 - Fov_bar2;

        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        // arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux, n_orb*n_occ, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        //arma::Mat<double> gamma_Q_new = gamma_G + gamma_J1 - gamma_J2;
        arma::Mat<double> gamma_Q = gamma_J1 - gamma_J2;
        // CUTK: if I use the above equation to get rid of gamma_Q the energy changes after this CCS calculation,
        // even though gamma_G (accu) is zero, next steps: check the norm of it and print the matrix to be sure. 

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        arma::vec iP (n_aux, fill::zeros);
        iP = PQinvhalf * iQ;

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
        arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;


        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        
        sigma_0 += (Fvv_hat*r1) - (r1*Foo_hat);

        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        #pragma omp parallel
        {
            arma::Mat<double> JG_local (n_orb, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];

                        }
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
        } // end parallel (2)

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        

        sigma.zeros();
        #pragma omp parallel
        {
            #pragma omp for	
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occ; k++) {
                            sigma_H(a,i) += Y_bar[(a*n_occ*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma(a,i) = sigma_0(a,i) + sigma_JG(a,i) - sigma_H(a,i);

                }
            }
	    } // end parallel (3)

    }
}


template<>
void ri_eomee_r<double,double>::davidson_restricted_energy_singlets(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &BQpv_a, Mat<double> &V_Pab,  
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Mat<double> &sigma) {
    

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
       

        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I (n_vir, n_occ, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);
        iQ_bar += BQov_a * vectorise(r1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = Fov_bar1 - Fov_bar2;



        /// step 4:
        #pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        #pragma omp parallel reduction(+:Yia,Yai,Y_bar,sigma_I)
        {	
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat sigma_I_local (n_vir, n_occ, fill::zeros);
            #pragma omp for schedule(dynamic)
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vir), n_aux, n_vir, false, true);


                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   bjai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   bjai
                
                double delta_ij = e_orb(i) + e_orb(j);
                double t2ab = 0.0, t2ba = 0.0;
                double r2ab = 0.0, r2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }

                            sigma_I_local(a,i) += (c_os * r2ab * Fov_hat(j,b)) + (c_os * t2ab * Fov_bar(j,b));
                            
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);
                            r2ba = (w1[a*n_vir+b] + w2b[a] + w3[a*n_vir+b] + w4b[a]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (r2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (r2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                            
                            sigma_I_local(a,i) += (c_os * t2ab * Fov_bar(j,b))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(j,b))
                                            + (c_os * r2ab * Fov_hat(j,b)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(j,b));
                            sigma_I_local(b,j) += (c_os * t2ab * Fov_bar(i,a))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(i,a))
                                            + (c_os * r2ab * Fov_hat(i,a)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(i,a));

                        }
                    }
                }
            }
            #pragma omp critical (Y)
            {
                Yai += Yai_local;
                Yia += Yia_local;
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
	    } // end parallel (1)


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );
        //for(size_t i = 1; i < n_occ; i++) {
        //}

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        // arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux, n_orb*n_occ, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G + gamma_J1 - gamma_J2;

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        arma::vec iP (n_aux, fill::zeros);
        iP = PQinvhalf * iQ;

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
        arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
        // arma::Mat<double> F33(F3_digestor.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;


        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        sigma_0 += (E_vv*r1) - (r1*E_oo);

        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        #pragma omp parallel reduction(+:JG)
	    {
            arma::Mat<double> JG_local (n_orb, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {
                            
                            JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                            
                        }
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
	    } // end parallel (2)

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        
        
        sigma.zeros();
        #pragma omp parallel
        {
            #pragma omp for	
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

                }
            }
	    } // end parallel (3)

    }
}



template<>
void ri_eomee_r<double,double>::diis_restricted_energy_singlets(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &BQpv_a, Mat<double> &V_Pab,  
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Mat<double> &sigma) {
    

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
        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I (n_vir, n_occ, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);
        iQ_bar += BQov_a * vectorise(r1);


        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = Fov_bar1 - Fov_bar2;


        /// step 4:
        // #pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        // #pragma omp parallel reduction(+:Yia,Yai,Y_bar,sigma_I)
        #pragma omp parallel
        {	
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat sigma_I_local (n_vir, n_occ, fill::zeros);
            // #pragma omp for schedule(dynamic)
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vir), n_aux, n_vir, false, true);


                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   bjai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   bjai
                
                double delta_ij = e_orb(i) + e_orb(j);
                double t2ab = 0.0, t2ba = 0.0;
                double r2ab = 0.0, r2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }

                            sigma_I_local(a,i) += (c_os * r2ab * Fov_hat(j,b)) + (c_os * t2ab * Fov_bar(j,b));
                            
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);
                            r2ba = (w1[a*n_vir+b] + w2b[a] + w3[a*n_vir+b] + w4b[a]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Yai[(i*n_vir*n_aux+a*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yai[(j*n_vir*n_aux+b*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (r2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Y_bar[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Y_bar_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (r2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                            
                            //sigma_I_local(a,i) += ((2.0*r2ab-r2ba) * Fov_hat(j,b)) + ((2.0*t2ab-t2ba) * Fov_bar(j,b));
                            //sigma_I_local(b,j) += ((2.0*r2ab-r2ba) * Fov_hat(i,a)) + ((2.0*t2ab-t2ba) * Fov_bar(i,a));
                            sigma_I_local(a,i) += (c_os * t2ab * Fov_bar(j,b))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(j,b))
                                            + (c_os * r2ab * Fov_hat(j,b)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(j,b));
                            sigma_I_local(b,j) += (c_os * t2ab * Fov_bar(i,a))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(i,a))
                                            + (c_os * r2ab * Fov_hat(i,a)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(i,a));

                        }
                    }
                }
            }
            #pragma omp critical (Y)
            {
                Yai += Yai_local;
                Yia += Yia_local;
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
	    } // end parallel (1)


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux, n_orb*n_occ, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G + gamma_J1 - gamma_J2;

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        arma::vec iP (n_aux, fill::zeros);
        iP = PQinvhalf * iQ;

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
        arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;

        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        sigma_0 += (E_vv*r1) - (r1*E_oo);

        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        #pragma omp parallel
	    {
            arma::Mat<double> JG_local (n_orb, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {
                            
                            JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                            
                        }
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
	    } // end parallel (2)


        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        
        
        //transformed vector
        //arma::mat sigma (n_vir, n_occ, fill::zeros);
        sigma.zeros();
        // #pragma omp parallel reduction(+:excit)
        #pragma omp parallel
        {
            double excit_local=0.0;
            #pragma omp for	
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
                    excit_local += (sigma(a,i)*r1(a,i)) / pow(norm(r1,"fro"),2);

                }
            }
            #pragma omp critical
            { 
                excit += excit_local; 
            }
	    } // end parallel (3)

        // update of the trial vector
        arma::mat residual(n_vir, n_occ, fill::zeros);
        arma::mat update (n_vir, n_occ, fill::zeros);
        #pragma omp parallel
	    {
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t a = 0; a < n_vir; a++) {
                    
                    double delta_ia = e_orb(i) - e_orb[n_occ+a];
                    residual(a,i) = (sigma(a,i) - (excit*r1(a,i))) / norm(r1,"fro");
                    update(a,i) = residual(a,i) / delta_ia;
                    r1(a,i) = (r1(a,i) + update(a,i)) / norm(r1,"fro");
                    
                }
            }
	    } // end parallel (4)

        exci = excit;
    }
}

template<>
void ri_eomee_r<double,double>::ccs_restricted_energy_singlets_digestor(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Mat<double> &sigma) {
    

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        exci = 0; 
        double t2ab = 0.0, t2ba = 0.0;
        double r2ab = 0.0, r2ba = 0.0;
        // intermediates
        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);
        iQ_bar += BQov_a * vectorise(r1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = Fov_bar1 - Fov_bar2;

        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        // arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux, n_orb*n_occ, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        //arma::Mat<double> gamma_Q_new = gamma_G + gamma_J1 - gamma_J2;
        arma::Mat<double> gamma_Q = gamma_J1 - gamma_J2;
        // CUTK: if I use the above equation to get rid of gamma_Q the energy changes after this CCS calculation,
        // even though gamma_G (accu) is zero, next steps: check the norm of it and print the matrix to be sure. 

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        arma::vec iP (n_aux, fill::zeros);
        iP = PQinvhalf * iQ;

        // digestor
        arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        {

            //  Step 1: Read libqints-type basis set from files and form shellpair basis.
            // libqints::basis_1e2c_shellpair_cgto<double> bsp;
            // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
            const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
            const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
            size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
            size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
            size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
            //  Construct the 2e3c shellpair basis and corresponding full basis range
            libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

            //  Step 2: prepare required input settings
            libqints::dev_omp dev;                  //  libqints-type device information.
            size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
            dev.init(1024);
            dev.nthreads = 1;
            dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
            libqints::deriv_code dc;
            dc.set(0);                //  Set integral derivative level
            libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
            libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
            qjob.begin(fbr);                                //  Start the libqints job for full basis range

            //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
            scr_2e3c scr(m_b3);

            //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
            dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
            if (dev.memory * dev.nthreads > mem_total) {
                std::cout << " Given memory is not enough for computing integrals." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

            //  Step 5:
            //  Memory available for thread-local result arrays:
            size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
            //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
            size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
            {  //  The code block here should be replaced by estimating # of munu basis function pairs
                //  and/or # of auxiliary basis function.
                nbsp_per_subrange = nbsp;
                naux_per_subrange = n_aux;
            }
            //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
            //  of each munu shell pair.
            size_t min_nbsp_per_subrange = 0;
            #pragma omp for 
            for (size_t isp = 0; isp < nsp; isp++) {
                size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
                min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
            }
            if (nbsp_per_subrange < min_nbsp_per_subrange) {
                std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
            //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
            //  of each auxiliary shell.
            size_t min_naux_per_subrange = 0;
            for (size_t is_q = 0; is_q < ns_q; is_q++) {
                size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
                min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
            }
            if (naux_per_subrange < min_naux_per_subrange) {
                std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

            //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
            arma::vec Fvec(nbsp);
            //  Result will be accumulated in the output arrays, so we need to zero out them
            Fvec.zeros();
            JG.zeros(); 
            dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occ, gamma_P, JG);
            // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

            //  Step 7: Loop over basis subranges and run libqints job
            libqints::batching_info<2> binfo;
            libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
            libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
            for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
                //  Current basis subrange
                libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
                    fbr, binfo.get_batch_window(biter.get_batch_number()));
                if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
                    std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
                    qjob.end();  //  End the libqints job before return
                    return;
                }
            }
            //  In case 2, we need to unpack F from vector form to matrix form with
            //  permutationally symmetric matrix elements are properly copied
            libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
            libaview::array_view<double> av_f(F.memptr(), F.n_elem);
            libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
            libaview::array_view<double> av_result(JG.memptr(), JG.n_elem);

        }

        arma::Mat<double> F3_digestor (n_vir, n_vir, fill::zeros);
        F3_digestor = 2.0 * CvirtA.st() * F * Lam_pA;

        // Fvv_hat
        arma::Mat<double> F33(F3_digestor.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;


        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        
        sigma_0 += (Fvv_hat*r1) - (r1*Foo_hat);

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        

        sigma.zeros();
        #pragma omp parallel
        {
            #pragma omp for	
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occ; k++) {
                            sigma_H(a,i) += Y_bar[(a*n_occ*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma(a,i) = sigma_0(a,i) + sigma_JG(a,i) - sigma_H(a,i);

                }
            }
	    } // end parallel (3)

    }
}


template<>
void ri_eomee_r<double,double>::davidson_restricted_energy_singlets_digestor(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Mat<double> &sigma) {
    

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
       

        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I (n_vir, n_occ, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);
        iQ_bar += BQov_a * vectorise(r1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = Fov_bar1 - Fov_bar2;



        /// step 4:
        #pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        #pragma omp parallel reduction(+:Yia,Yai,Y_bar,sigma_I)
        {	
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat sigma_I_local (n_vir, n_occ, fill::zeros);
            #pragma omp for schedule(dynamic)
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vir), n_aux, n_vir, false, true);


                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   bjai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   bjai
                
                double delta_ij = e_orb(i) + e_orb(j);
                double t2ab = 0.0, t2ba = 0.0;
                double r2ab = 0.0, r2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }

                            sigma_I_local(a,i) += (c_os * r2ab * Fov_hat(j,b)) + (c_os * t2ab * Fov_bar(j,b));
                            
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);
                            r2ba = (w1[a*n_vir+b] + w2b[a] + w3[a*n_vir+b] + w4b[a]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (r2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (r2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                            
                            sigma_I_local(a,i) += (c_os * t2ab * Fov_bar(j,b))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(j,b))
                                            + (c_os * r2ab * Fov_hat(j,b)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(j,b));
                            sigma_I_local(b,j) += (c_os * t2ab * Fov_bar(i,a))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(i,a))
                                            + (c_os * r2ab * Fov_hat(i,a)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(i,a));

                        }
                    }
                }
            }
            #pragma omp critical (Y)
            {
                Yai += Yai_local;
                Yia += Yia_local;
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
	    } // end parallel (1)


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux, n_orb*n_occ, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G + gamma_J1 - gamma_J2;

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        arma::vec iP (n_aux, fill::zeros);
        iP = PQinvhalf * iQ;

        // digestor
        arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        {

            //  Step 1: Read libqints-type basis set from files and form shellpair basis.
            // libqints::basis_1e2c_shellpair_cgto<double> bsp;
            // libqints::basis_1e1c_cgto<double> ;  //  1e1c auxiliary basis
            const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
            const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
            size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
            size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
            size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
            //  Construct the 2e3c shellpair basis and corresponding full basis range
            libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

            //  Step 2: prepare required input settings
            libqints::dev_omp dev;                  //  libqints-type device information.
            size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
            dev.init(1024);
            dev.nthreads = 1;
            dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
            libqints::deriv_code dc;
            dc.set(0);                //  Set integral derivative level
            libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
            libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
            qjob.begin(fbr);                                //  Start the libqints job for full basis range

            //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
            scr_2e3c scr(m_b3);

            //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
            dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
            if (dev.memory * dev.nthreads > mem_total) {
                std::cout << " Given memory is not enough for computing integrals." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

            //  Step 5:
            //  Memory available for thread-local result arrays:
            size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
            //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
            size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
            {  //  The code block here should be replaced by estimating # of munu basis function pairs
                //  and/or # of auxiliary basis function.
                nbsp_per_subrange = nbsp;
                naux_per_subrange = n_aux;
            }
            //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
            //  of each munu shell pair.
            size_t min_nbsp_per_subrange = 0;
            #pragma omp for 
            for (size_t isp = 0; isp < nsp; isp++) {
                size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
                min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
            }
            if (nbsp_per_subrange < min_nbsp_per_subrange) {
                std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
            //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
            //  of each auxiliary shell.
            size_t min_naux_per_subrange = 0;
            for (size_t is_q = 0; is_q < ns_q; is_q++) {
                size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
                min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
            }
            if (naux_per_subrange < min_naux_per_subrange) {
                std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

            //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
            arma::vec Fvec(nbsp);
            //  Result will be accumulated in the output arrays, so we need to zero out them
            Fvec.zeros();
            JG.zeros(); 
            dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occ, gamma_P, JG);
            // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

            //  Step 7: Loop over basis subranges and run libqints job
            libqints::batching_info<2> binfo;
            libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
            libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
            for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
                //  Current basis subrange
                libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
                    fbr, binfo.get_batch_window(biter.get_batch_number()));
                if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
                    std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
                    qjob.end();  //  End the libqints job before return
                    return;
                }
            }
            //  In case 2, we need to unpack F from vector form to matrix form with
            //  permutationally symmetric matrix elements are properly copied
            libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
            libaview::array_view<double> av_f(F.memptr(), F.n_elem);
            libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
            libaview::array_view<double> av_result(JG.memptr(), JG.n_elem);

        }

        arma::Mat<double> F3_digestor (n_vir, n_vir, fill::zeros);
        F3_digestor = 2.0 * CvirtA.st() * F * Lam_pA;


        // Fvv_hat
        arma::Mat<double> F33(F3_digestor.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;


        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        sigma_0 += (E_vv*r1) - (r1*E_oo);

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        
        
        sigma.zeros();
        #pragma omp parallel
        {
            #pragma omp for	
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

                }
            }
	    } // end parallel (3)

    }
}



template<>
void ri_eomee_r<double,double>::diis_restricted_energy_singlets_digestor(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Mat<double> &sigma) {
    

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
        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I (n_vir, n_occ, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);
        iQ_bar += BQov_a * vectorise(r1);


        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = Fov_bar1 - Fov_bar2;


        /// step 4:
        // #pragma omp declare reduction( + : arma::mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        // #pragma omp parallel reduction(+:Yia,Yai,Y_bar,sigma_I)
        #pragma omp parallel
        {	
            arma::mat Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::mat sigma_I_local (n_vir, n_occ, fill::zeros);
            // #pragma omp for schedule(dynamic)
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vir), n_aux, n_vir, false, true);


                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   bjai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   aibj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   bjai
                
                double delta_ij = e_orb(i) + e_orb(j);
                double t2ab = 0.0, t2ba = 0.0;
                double r2ab = 0.0, r2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }

                            sigma_I_local(a,i) += (c_os * r2ab * Fov_hat(j,b)) + (c_os * t2ab * Fov_bar(j,b));
                            
                        }
                    }

                } else {
                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();
                    const double *w2 = W2.memptr();
                    const double *w3 = W3.memptr();
                    const double *w4 = W4.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;
                        const double *w1b = w1 + b * n_vir;
                        const double *w2b = w2 + b * n_vir;
                        const double *w3b = w3 + b * n_vir;
                        const double *w4b = w4 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
                            t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

                            r2ab = (w1b[a] + w2[a*n_vir+b] + w3b[a] + w4[a*n_vir+b]) / (dijb - e_orb[n_occ+a] + exci);
                            r2ba = (w1[a*n_vir+b] + w2b[a] + w3[a*n_vir+b] + w4b[a]) / (dijb - e_orb[n_occ+a] + exci);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Yai[(i*n_vir*n_aux+a*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yai[(j*n_vir*n_aux+b*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (r2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Y_bar[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Y_bar_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (r2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                            
                            //sigma_I_local(a,i) += ((2.0*r2ab-r2ba) * Fov_hat(j,b)) + ((2.0*t2ab-t2ba) * Fov_bar(j,b));
                            //sigma_I_local(b,j) += ((2.0*r2ab-r2ba) * Fov_hat(i,a)) + ((2.0*t2ab-t2ba) * Fov_bar(i,a));
                            sigma_I_local(a,i) += (c_os * t2ab * Fov_bar(j,b))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(j,b))
                                            + (c_os * r2ab * Fov_hat(j,b)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(j,b));
                            sigma_I_local(b,j) += (c_os * t2ab * Fov_bar(i,a))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(i,a))
                                            + (c_os * r2ab * Fov_hat(i,a)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(i,a));

                        }
                    }
                }
            }
            #pragma omp critical (Y)
            {
                Yai += Yai_local;
                Yia += Yia_local;
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
	    } // end parallel (1)


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux, n_orb*n_occ, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G + gamma_J1 - gamma_J2;

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        arma::vec iP (n_aux, fill::zeros);
        iP = PQinvhalf * iQ;

        // digestor
        arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        {

            //  Step 1: Read libqints-type basis set from files and form shellpair basis.
            // libqints::basis_1e2c_shellpair_cgto<double> bsp;
            // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
            const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
            const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
            size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
            size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
            size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
            //  Construct the 2e3c shellpair basis and corresponding full basis range
            libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

            //  Step 2: prepare required input settings
            libqints::dev_omp dev;                  //  libqints-type device information.
            size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
            dev.init(1024);
            dev.nthreads = 1;
            dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
            libqints::deriv_code dc;
            dc.set(0);                //  Set integral derivative level
            libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
            libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
            qjob.begin(fbr);                                //  Start the libqints job for full basis range

            //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
            scr_2e3c scr(m_b3);

            //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
            dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
            if (dev.memory * dev.nthreads > mem_total) {
                std::cout << " Given memory is not enough for computing integrals." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

            //  Step 5:
            //  Memory available for thread-local result arrays:
            size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
            //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
            size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
            {  //  The code block here should be replaced by estimating # of munu basis function pairs
                //  and/or # of auxiliary basis function.
                nbsp_per_subrange = nbsp;
                naux_per_subrange = n_aux;
            }
            //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
            //  of each munu shell pair.
            size_t min_nbsp_per_subrange = 0;
            #pragma omp for 
            for (size_t isp = 0; isp < nsp; isp++) {
                size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
                min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
            }
            if (nbsp_per_subrange < min_nbsp_per_subrange) {
                std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
            //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
            //  of each auxiliary shell.
            size_t min_naux_per_subrange = 0;
            for (size_t is_q = 0; is_q < ns_q; is_q++) {
                size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
                min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
            }
            if (naux_per_subrange < min_naux_per_subrange) {
                std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

            //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
            arma::vec Fvec(nbsp);
            //  Result will be accumulated in the output arrays, so we need to zero out them
            Fvec.zeros();
            JG.zeros(); 
            dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occ, gamma_P, JG);
            // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

            //  Step 7: Loop over basis subranges and run libqints job
            libqints::batching_info<2> binfo;
            libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
            libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
            for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
                //  Current basis subrange
                libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
                    fbr, binfo.get_batch_window(biter.get_batch_number()));
                if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
                    std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
                    qjob.end();  //  End the libqints job before return
                    return;
                }
            }
            //  In case 2, we need to unpack F from vector form to matrix form with
            //  permutationally symmetric matrix elements are properly copied
            libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
            libaview::array_view<double> av_f(F.memptr(), F.n_elem);
            libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
            libaview::array_view<double> av_result(JG.memptr(), JG.n_elem);

        }

        arma::Mat<double> F3_digestor (n_vir, n_vir, fill::zeros);
        F3_digestor = 2.0 * CvirtA.st() * F * Lam_pA;

        // Fvv_hat
        arma::Mat<double> F33(F3_digestor.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;

        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<double> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        E_oo = Foo_hat + (YQai.st() * BQov).st(); // E_ji
        
        sigma_0 += (E_vv*r1) - (r1*E_oo);


        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        
        
        //transformed vector
        //arma::mat sigma (n_vir, n_occ, fill::zeros);
        sigma.zeros();
        // #pragma omp parallel reduction(+:excit)
        #pragma omp parallel
        {
            double excit_local=0.0;
            #pragma omp for	
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
                    excit_local += (sigma(a,i)*r1(a,i)) / pow(norm(r1,"fro"),2);

                }
            }
            #pragma omp critical
            { 
                excit += excit_local; 
            }
	    } // end parallel (3)

        // update of the trial vector
        arma::mat residual(n_vir, n_occ, fill::zeros);
        arma::mat update (n_vir, n_occ, fill::zeros);
        #pragma omp parallel
	    {
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t a = 0; a < n_vir; a++) {
                    
                    double delta_ia = e_orb(i) - e_orb[n_occ+a];
                    residual(a,i) = (sigma(a,i) - (excit*r1(a,i))) / norm(r1,"fro");
                    update(a,i) = residual(a,i) / delta_ia;
                    r1(a,i) = (r1(a,i) + update(a,i)) / norm(r1,"fro");
                    
                }
            }
	    } // end parallel (4)

        exci = excit;
    }
}


template<typename TC, typename TI>
void ri_eomee_r<TC, TI>::ccs_restricted_energy_singlets(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, 
    Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
    Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpo_a, 
    Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &BQpv_a, Mat<complex<double>> &V_Pab,  
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,  
    double c_os, double c_ss, Mat<complex<double>> &sigma) {


    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        exci = (0.0,0.0);
        complex<double> t2ab(0.,0.), t2ba(0.,0.);
        complex<double> r2ab(0.,0.), r2ba(0.,0.);
        
        // intermediates
        arma::Mat<complex<double>> sigma_0 (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma_JG (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma_H (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> JG (n_orb, n_occ, fill::zeros);
        arma::Mat<complex<double>> sigma_I (n_vir, n_occ, fill::zeros);
        //arma::Mat<complex<double>> E_vv (n_vir, n_vir, fill::zeros);
        //arma::Mat<complex<double>> E_oo (n_occ, n_occ, fill::zeros);
        arma::Mat<complex<double>> Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::Mat<complex<double>> Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::Mat<complex<double>> Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        arma::Mat<complex<double>> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        
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

        arma::Mat<complex<double>> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<complex<double>> YQai(Yai.memptr(), n_aux*n_vir, n_occ, false, true);
        arma::Mat<complex<double>> BQov(BQov_a.memptr(), n_aux*n_vir, n_occ, false, true);

        //Fvv_hat is E_vv, E_ab
        //Foo_hat is E_oo, E_ji
        
        sigma_0 += (Fvv_hat*r1) - (r1*Foo_hat);


        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<complex<double>> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<complex<double>> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<complex<double>> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );
        //for(size_t i = 1; i < n_occ; i++) {
        //}

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<complex<double>> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<complex<double>> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<complex<double>> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<complex<double>> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<complex<double>> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
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

        #pragma omp parallel
	{
        arma::Mat<complex<double>> JG_local (n_orb, n_occ, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                        * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG)
        {
            JG += JG_local;
        }
	} // end parallel (2)

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        

        sigma.zeros();
        #pragma omp parallel
        {
        #pragma omp for	
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

            }
        }
        } // end parallel (3)
        
        
    }

}


template<typename TC, typename TI>
void ri_eomee_r<TC, TI>::davidson_restricted_energy_singlets(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, 
    Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
    Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpo_a, 
    Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &BQpv_a, Mat<complex<double>> &V_Pab,  
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,  
    double c_os, double c_ss, Mat<complex<double>> &sigma) {


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
        #pragma omp declare reduction( + : complex<double> : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        #pragma omp declare reduction( + : arma::cx_mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        #pragma omp parallel reduction(+:Yia,Yai,Y_bar,sigma_I) 
        {
            arma::Mat<complex<double>> Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::Mat<complex<double>> Yia_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::Mat<complex<double>> Y_bar_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::Mat<complex<double>> sigma_I_local (n_vir, n_occ, fill::zeros);
        
            #pragma omp for schedule(dynamic)
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
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }

                            sigma_I_local(a,i) += (c_os * r2ab * Fov_hat(j,b)) + (c_os * t2ab * Fov_bar(j,b));
                            
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
                                //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Yai[(i*n_vir*n_aux+a*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yai[(j*n_vir*n_aux+b*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (r2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Y_bar[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Y_bar_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (r2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];

                            }
                            
                            //sigma_I(a,i) += ((2.0*r2ab-r2ba) * Fov_hat(j,b)) + ((2.0*t2ab-t2ba) * Fov_bar(j,b));
                            //sigma_I(b,j) += ((2.0*r2ab-r2ba) * Fov_hat(i,a)) + ((2.0*t2ab-t2ba) * Fov_bar(i,a));
                            sigma_I_local(a,i) += (c_os * t2ab * Fov_bar(j,b))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(j,b))
                                            + (c_os * r2ab * Fov_hat(j,b)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(j,b));
                            sigma_I_local(b,j) += (c_os * t2ab * Fov_bar(i,a))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(i,a))
                                            + (c_os * r2ab * Fov_hat(i,a)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(i,a));
                        }
                    }
                }
            }
            #pragma omp critical (Y)
            {
                Yai += Yai_local;
                Yia += Yia_local;
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
	    } // end parallel (1)

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
        //for(size_t i = 1; i < n_occ; i++) {
        //}

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<complex<double>> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<complex<double>> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);

        // omega_J2: third term of Γ(P,iβ)
        arma::Mat<complex<double>> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<complex<double>> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<complex<double>> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
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

        #pragma omp parallel
	{
        arma::Mat<complex<double>> JG_local (n_orb, n_occ, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                        * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG)
        {
            JG += JG_local;
        }
	} // end parallel (2)

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        

        sigma.zeros();
        #pragma omp parallel
        {
        #pragma omp for	
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

            }
        } // end parallel (3)
        }   
    }
    
}


template<typename TC, typename TI>
void ri_eomee_r<TC, TI>::diis_restricted_energy_singlets(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, 
    Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
    Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpo_a, 
    Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &BQpv_a, Mat<complex<double>> &V_Pab,  
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,  
    double c_os, double c_ss, Mat<complex<double>> &sigma) {


    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }
    
    {
        
    	TC excit=(0.,0.);
        
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
        #pragma omp declare reduction( + : complex<double> : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        #pragma omp declare reduction( + : arma::cx_mat : omp_out += omp_in ) initializer( omp_priv = omp_orig )
        #pragma omp parallel reduction(+:Yia,Yai,Y_bar,sigma_I) 
        {
            arma::Mat<complex<double>> Yai_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::Mat<complex<double>> Yia_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::Mat<complex<double>> Y_bar_local (n_aux, n_vir*n_occ, fill::zeros);
            arma::Mat<complex<double>> sigma_I_local (n_vir, n_occ, fill::zeros);
        
            #pragma omp for schedule(dynamic)
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
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * r2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            }

                            sigma_I_local(a,i) += (c_os * r2ab * Fov_hat(j,b)) + (c_os * t2ab * Fov_bar(j,b));
                            
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
                                //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Yai[(i*n_vir*n_aux+a*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yai_local[(i*n_vir*n_aux+a*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yai[(j*n_vir*n_aux+b*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yai_local[(j*n_vir*n_aux+b*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                //Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Y_bar_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (r2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Y_bar[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Y_bar_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (r2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                        + c_ss * (r2ab-r2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];

                            }
                            
                            //sigma_I(a,i) += ((2.0*r2ab-r2ba) * Fov_hat(j,b)) + ((2.0*t2ab-t2ba) * Fov_bar(j,b));
                            //sigma_I(b,j) += ((2.0*r2ab-r2ba) * Fov_hat(i,a)) + ((2.0*t2ab-t2ba) * Fov_bar(i,a));
                            sigma_I_local(a,i) += (c_os * t2ab * Fov_bar(j,b))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(j,b))
                                            + (c_os * r2ab * Fov_hat(j,b)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(j,b));
                            sigma_I_local(b,j) += (c_os * t2ab * Fov_bar(i,a))
                                            + (c_ss * (t2ab-t2ba) * Fov_bar(i,a))
                                            + (c_os * r2ab * Fov_hat(i,a)) 
                                            + (c_ss * (r2ab-r2ba) * Fov_hat(i,a));
                        }
                    }
                }
            }
            #pragma omp critical (Y)
            {
                Yai += Yai_local;
                Yia += Yia_local;
                Y_bar += Y_bar_local;
                sigma_I += sigma_I_local;
            }
	    } // end parallel (1)

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

        #pragma omp parallel reduction(+:JG)
	    {
            arma::Mat<complex<double>> JG_local (n_orb, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {
                            
                            JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)]
                                            * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                            
                        }
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
	    } // end parallel (2)

        /// step 6:
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;        

        // c-product
        arma::Mat<double> r1_real = real(r1);
        arma::Mat<double> r1_imag = imag(r1);
        arma::vec vecre =vectorise(r1_real);
        arma::vec vecim =vectorise(r1_imag);
        double dp_real = dot(vecre, vecre) - dot(vecim, vecim);
        double dp_imag = 2.0 * dot(vecre, vecim);
        TC dp (dp_real, dp_imag);
        TC cnorm = sqrt(dp);

	    //transformed vector
        sigma.zeros();
        #pragma omp parallel reduction(+:excit)
        {
            TC excit_local=0.0;
            #pragma omp for	
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
                    //excit_local += (sigma(a,i)*r1(a,i))  / abs(dp);
                    excit_local += (conj(dp) * (sigma(a,i)*r1(a,i)))  / (conj(dp) * dp);

                }
            }
            #pragma omp critical
            { 
                excit += excit_local; 
            }
	    } // end parallel (3)
        
        
        // update of the trial vector
        arma::Mat<complex<double>> residual (n_vir, n_occ, fill::zeros);
        arma::Mat<complex<double>> update (n_vir, n_occ, fill::zeros);
        #pragma omp parallel
	    {
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {
                    
                    TC delta_ia = e_orb(i) - e_orb[n_occ+a];

                    //residual(a,i) = (sigma(a,i) - (excit*r1(a,i))) / abs(cnorm);
                    residual(a,i) = (conj(cnorm) * (sigma(a,i) - (excit*r1(a,i)))) / (conj(cnorm) * cnorm);
                    update(a,i) = (conj(delta_ia) * residual(a,i)) / (conj(delta_ia) * delta_ia);
                    //r1(a,i) = (r1(a,i) + update(a,i)) / abs(cnorm);
                    r1(a,i) = (conj(cnorm) * (r1(a,i) + update(a,i))) / (conj(cnorm) * cnorm);
                }
            }
	    } // end parallel (4)
	    exci = excit;
    }
    
}


template<typename TC, typename TI>
void ri_eomee_r<TC, TI>::ccs_restricted_energy_singlets_digestor(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, 
    Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
    Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpo_a, 
    Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,  
    double c_os, double c_ss, Mat<complex<double>> &sigma) {

    // GPP: activate this with the digestor
    throw std::runtime_error("Digestor option for this algorithm is not yet implemented.");

}


template<typename TC, typename TI>
void ri_eomee_r<TC, TI>::davidson_restricted_energy_singlets_digestor(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, 
    Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
    Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpo_a, 
    Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,  
    double c_os, double c_ss, Mat<complex<double>> &sigma) {

    // GPP: activate this with the digestor
    throw std::runtime_error("Digestor option for this algorithm is not yet implemented.");

}


template<typename TC, typename TI>
void ri_eomee_r<TC, TI>::diis_restricted_energy_singlets_digestor(
    complex<double>& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, 
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoh_a, 
    Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
    Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpo_a, 
    Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, 
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
    Mat<complex<double>> &t1, Mat<complex<double>> &r1,
    Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,  
    double c_os, double c_ss, Mat<complex<double>> &sigma) {

    // GPP: activate this with the digestor
    throw std::runtime_error("Digestor option for this algorithm is not yet implemented.");

}


#if 0
/// GPP: RI-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: NOT THE OPTIMIZED CODE
template<>
void ri_eom_r<double>::restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQph_a, Mat<double> &BQoh_a, 
    Mat<double> &BQpv_a, Mat<double> &BQbh_a, Mat<double> &BQpb_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Mat<double> &r1,
    Mat<double> &residual, Col<double> &e_orb,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3) {
    
    memory_pool<double> mem(av_buff_ao);
    typename memory_pool<double>::checkpoint chkpt = mem.save_state();
    
    {
        
        double excit=0.0;
        
        // intermediates
        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I (n_vir, n_occ, fill::zeros);
        arma::mat Fvo_bar (n_vir, n_occ, fill::zeros);
        arma::mat Fov_hat (n_occ, n_vir, fill::zeros);
        arma::mat Fvv_hat (n_vir, n_vir, fill::zeros);
        arma::mat Fvv_hat2 (n_vir, n_vir, fill::zeros);
        arma::mat Foo_hat (n_occ, n_occ, fill::zeros);
        arma::mat Foo_hat2 (n_occ, n_occ, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Fov_bar (n_occ, n_vir, fill::zeros);
        arma::mat Fov_bar2 (n_occ, n_vir, fill::zeros);
        arma::mat Y (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        
        
        /// step 3: form i^Q and F_ia
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        for(size_t k = 0; k < n_occ; k++) {
            for(size_t c = 0; c < n_vir; c++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                    iQ(Q) += BQov_a[(k*n_vir*n_aux+c*n_aux+Q)] * t1(c,k);
                    iQ_bar(Q) += BQov_a[(k*n_vir*n_aux+c*n_aux+Q)] * r1(c,k);
                }
            }
        }
       
        // ovov
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                   Fov_hat(i,a) += 2.0 * iQ(Q) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                   Fov_bar(i,a) += 2.0 * iQ_bar(Q) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                }
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                       for(size_t Q = 0; Q < n_aux; Q++) {
                           Fov_hat(i,a) -= BQov_a[(j*n_vir*n_aux+a*n_aux+Q)]
                                           * BQov_a[(i*n_vir*n_aux+b*n_aux+Q)]*t1(b,j);
                           Fov_bar(i,a) -= BQov_a[(j*n_vir*n_aux+a*n_aux+Q)]
                                           * BQov_a[(i*n_vir*n_aux+b*n_aux+Q)]*r1(b,j);
                        }
                    }
                }
            }
        }

        // ovvv
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t b = 0; b < n_vir; b++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                    Fvv_hat(a,b) += 2.0 * iQ(Q) * BQpv_a[(a*n_vir*n_aux+b*n_aux+Q)];//kcab
                    // Fvv_hat2(a,b) += 2.0 * iQ(Q) * BQvv_a[(b*n_vir*n_aux+a*n_aux+Q)];//1
                }
                    
                // for(size_t i = 0; i < n_occ; i++) {
                //     for(size_t Q = 0; Q < n_aux; Q++) {
                //             Fvv_hat2(a,b) -= 2.0 * iQ(Q) * BQov_a[(i*n_vir*n_aux+b*n_aux+Q)] * t1(a,i);//2
                //     }
                // }

                for(size_t k = 0; k < n_occ; k++) {
                    for(size_t c = 0; c < n_vir; c++) {
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            Fvv_hat(a,b) -= BQpv_a[(a*n_vir*n_aux+c*n_aux+Q)]
                                                * BQov_a[(k*n_vir*n_aux+b*n_aux+Q)] * t1(c,k);//ackb

                            // Fvv_hat2(a,b) -= BQov_a[(k*n_vir*n_aux+b*n_aux+Q)]
                            //                     * BQvv_a[(a*n_vir*n_aux+c*n_aux+Q)] * t1(c,k);//3

                            // for(size_t l = 0; l < n_occ; l++) {
                            //     Fvv_hat2(a,b) += BQov_a[(k*n_vir*n_aux+b*n_aux+Q)] 
                            //                     * BQov_a[(l*n_vir*n_aux+c*n_aux+Q)]*t1(c,k)*t1(a,l);//4
                            // }
                        }
                    }
                }
                Fvv_hat(a,b) += f_vv(a,b);
            }
        }
        
        // ovoo
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t j = 0; j < n_occ; j++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                    Foo_hat(i,j) += 2.0 * iQ(Q) * BQoh_a[(i*n_occ*n_aux+j*n_aux+Q)];
                    // Foo_hat2(i,j) += 2.0 * iQ(Q) * BQoo_a[(i*n_occ*n_aux+j*n_aux+Q)];//1
                }

                // for(size_t c = 0; c < n_vir; c++) {
                //     for(size_t Q = 0; Q < n_aux; Q++) {
                //         Foo_hat2(i,j) += 2.0 * iQ(Q) * BQov_a[(i*n_vir*n_aux+c*n_aux+Q)] * t1(c,j);//2
                //     }
                // }

                for(size_t k = 0; k < n_occ; k++) {
                    for(size_t c = 0; c < n_vir; c++) {
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            
                            Foo_hat(i,j) -= BQov_a[(i*n_vir*n_aux+c*n_aux+Q)]*t1(c,k)
                                               * BQoh_a[(k*n_occ*n_aux+j*n_aux+Q)];

                            // Foo_hat2(i,j) -= BQov_a[(i*n_vir*n_aux+c*n_aux+Q)]*t1(c,k)
                            //                    * BQoo_a[(k*n_occ*n_aux+j*n_aux+Q)];//3
                        }
                        // for(size_t d = 0; d < n_vir; d++) {
                        //     for(size_t Q = 0; Q < n_aux; Q++) {
                        //         Foo_hat2(i,j) -= BQov_a[(i*n_vir*n_aux+c*n_aux+Q)] * t1(c,k)
                        //                             * BQov_a[(k*n_vir*n_aux+d*n_aux+Q)] * t1(d,j);//4
                        //     }
                        // }
                    }
                }
                Foo_hat(i,j) += f_oo(i,j);
            }
        }
        
        /// step 4:
        E_vv = Fvv_hat;
        E_oo = Foo_hat;
        
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
                        double t_ijab = 0.0;
                        double t_ijba = 0.0;
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            t_ijab += BQph_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)];
                            t_ijba += BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)];
                            
                            r_ijab += BQbh_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)]
                                    + BQbh_a[(b*n_occ*n_aux+j*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+i*n_aux+Q)]
                                    + BQpb_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)]
                                    + BQpb_a[(b*n_occ*n_aux+j*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+i*n_aux+Q)];
                            r_ijba += BQbh_a[(a*n_occ*n_aux+j*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]
                                    + BQbh_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)]
                                    + BQpb_a[(a*n_occ*n_aux+j*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]
                                    + BQpb_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)];

                        }
                        
                        t_ijab = t_ijab / delta_ijab;
                        t_ijba = t_ijba / delta_ijab;
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        // sigma_G - (jb|ca) ovvv permute 2,4
                        for(size_t c = 0; c < n_vir; c++) {
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                E_vv(a,c) -= (2.0 * t_ijab - t_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vir*n_aux+c*n_aux+Q)];
                            }
                        }
                            
                        // sigma_H - (jb|ik) ovoo permute 1,3
                        for(size_t k = 0; k < n_occ; k++) {
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                E_oo(k,i) += (2.0 * t_ijab - t_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar[(a*n_occ*n_aux+i*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I(a,i) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b))
                                            + ((2.0 * t_ijab - t_ijba) * Fov_bar(j,b))
                        ;
                        
                    }
                }
            }
        }
        
        
        
        sigma_0 += (E_vv*r1) - (r1*E_oo);
        
        
        /// step 5:
        arma::mat gamma_Q (n_aux, n_orb*n_occ, fill::zeros);
        for(size_t beta = 0; beta < n_orb; beta++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                    for(size_t a = 0; a < n_vir; a++) {
                        
                        // sigma_G
                        gamma_Q[(i*n_orb*n_aux+beta*n_aux+Q)] += Y_bar[(a*n_occ*n_aux+i*n_aux+Q)] * CvirtA(beta,a);

                    }

                    // sigma_J
                    gamma_Q[(i*n_orb*n_aux+beta*n_aux+Q)] += 2.0 * Lam_hA(beta,i) * iQ_bar(Q);
                    
                    for(size_t k = 0; k < n_occ; k++) {
                        gamma_Q[(i*n_orb*n_aux+beta*n_aux+Q)] -= Lam_hA_bar(beta,k) * BQoh_a[(k*n_occ*n_aux+i*n_aux+Q)];

                    }
                }
            }
        }
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        
        // (P|ab)
        arma::Mat<double> Unit(n_orb, n_orb, fill::eye);
        std::vector<size_t> vblst(1);
        idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
        blst.populate();
        op_coulomb op;
        {
            motran_2e3c_incore_result_container<double> buf(av_buff_ao);
            scr_null<bat_2e3c_shellpair_cgto<double>> scr;
            motran_2e3c<double, double> mot(op, m_b3, scr, m_dev);
            mot.set_trn(Unit);
            mot.run(m_dev, blst, buf);
        }
        arma::Mat<double> V_Pab(arrays<double>::ptr(av_buff_ao), Unit.n_cols * Unit.n_cols, n_aux, false, true);
        mem.load_state(chkpt);
        
        
        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
        arma::Mat<double> G (n_orb, n_occ, fill::zeros);
        arma::Mat<double> J (n_orb, n_occ, fill::zeros);
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
        
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;
        
        
        //transformed vector
        arma::mat sigma (n_vir, n_occ, fill::zeros);
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {
                
                // sigma_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        sigma_H(a,i) += Y_bar[(a*n_occ*n_aux+k*n_aux+P)]
                                            * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }
    
                sigma(a,i) = sigma_0(a,i) + sigma_JG(a,i) - sigma_H(a,i) + sigma_I(a,i);
                excit += (sigma(a,i)*r1(a,i)) / pow(norm(r1,"fro"),2);

            }
        }
        
        
        // update of the trial vector
        residual.zeros();
        arma::mat update (n_vir, n_occ, fill::zeros);
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {
                
                double delta_ia = e_orb(i) - e_orb[n_occ+a];
                residual(a,i) = (sigma(a,i) - (excit*r1(a,i))) / norm(r1,"fro");
                update(a,i) = residual(a,i) / delta_ia;
                r1(a,i) = (r1(a,i) + update(a,i)) / norm(r1,"fro");
                
            }
        }
        
        exci = excit;
        
    }
}

template<>
void ri_eom_r<double>::restricted_energy_triplets(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, Mat<double> &BQph_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, Mat<double> &BQho_a, 
    Mat<double> &BQoo_a, Mat<double> &BQob_a, Mat<double> &BQpv_a, 
    Mat<double> &BQpo_a, Mat<double> &BQhb_a, Mat<double> &BQbp_a, 
    Mat<double> &BQbh_a, Mat<double> &BQpb_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo, 
    Mat<double> &f_ov,  Mat<double> &f_vo,
    Mat<double> &t1, Mat<double> &r1,
    Mat<double> &residual, Col<double> &e_orb,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3) {

    memory_pool<double> mem(av_buff_ao);
    typename memory_pool<double>::checkpoint chkpt = mem.save_state();

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
        arma::mat sigma_0 (n_vir, n_occ, fill::zeros);
        arma::mat sigma_JG (n_vir, n_occ, fill::zeros);
        arma::mat sigma_H (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I (n_vir, n_occ, fill::zeros);
        arma::mat sigma_I2 (n_vir, n_occ, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat E_oo (n_occ, n_occ, fill::zeros);
        arma::mat Yai (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar2 (n_aux, n_vir*n_occ, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        arma::vec iQ_bar (n_aux, fill::zeros);
        iQ += BQov_a * t1;
        iQ_bar += BQov_a * r1;

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fov_bar
        arma::Mat<double> F2 = 2.0 * iQ_bar.st() * BQov_a;
        arma::Mat<double> F22(F2.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_bar1 = F22.st();
        arma::Mat<double> BQob(BQob_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_bar2 = BQob.st() * BQvo;
        arma::Mat<double> Fov_bar = - Fov_bar2; //triplet

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
        arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> Fvv_hat1 = F33.st();
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        // Foo_hat
        arma::Mat<double> F4 = 2.0 * iQ.st() * BQoh_a;
        arma::Mat<double> F44(F4.memptr(), n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat1 = F44.st();
        arma::Mat<double> BQho(BQho_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Foo_hat2 = BQoo.st() * BQho;
        arma::Mat<double> Foo_hat = f_oo + Foo_hat1 - Foo_hat2;


        /// step 4:

        E_vv = Fvv_hat;
        E_oo = Foo_hat;

        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
                        double t_ijab = 0.0;
                        double t_ijba = 0.0;
                        double r_ijab = 0.0;
                        // double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            t_ijab += BQph_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)];
                            t_ijba += BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)];

                            r_ijab += 2.0 * (BQbh_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)]    //(ai|bj)
                                            + BQpb_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)])  
                                            - BQbh_a[(a*n_occ*n_aux+j*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]   //(aj|bi)
                                            - BQpb_a[(a*n_occ*n_aux+j*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]   
                                            - BQbh_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)]   //(bi|aj)
                                            - BQpb_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)];  

                        }
                        
                        t_ijab = t_ijab / delta_ijab;
                        t_ijba = t_ijba / delta_ijab;
                        r_ijab = r_ijab / (delta_ijab + exci);
                        
                        // sigma_G - (jb|ca) ovvv permute 2,4
                        for(size_t c = 0; c < n_vir; c++) {
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                E_vv(a,c) -= (2.0 * t_ijab - t_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vir*n_aux+c*n_aux+Q)];
                            }
                        }
                            
                        // sigma_H - (jb|ik) ovoo permute 1,3
                        for(size_t k = 0; k < n_occ; k++) {
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                E_oo(k,i) += (2.0 * t_ijab - t_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar[(a*n_occ*n_aux+i*n_aux+P)] += r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I(a,i) += (r_ijab  * Fov_hat(j,b)) - (t_ijba * Fov_bar(j,b)); 
                        
                    }
                }
            }
        }

        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar(Y_bar.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G1 = YQia_bar * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G = gamma_G1.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_G.insert_cols(i*n_orb, gamma_G1.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11 = 2.0 * iQ_bar * vectorise(Lam_hA).st();
        arma::Mat<double> gamma_J1(gamma_J11.memptr(), n_aux*n_occ, n_orb, false, true);

        // / omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_J22 = BQoh * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2 = gamma_J22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_J2.insert_cols(i*n_orb, gamma_J22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G - gamma_J2; // triplet

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
        gamma_P = PQinvhalf * gamma_Q;

        
        // (P|ab)
        arma::Mat<double> Unit(n_orb, n_orb, fill::eye);
        std::vector<size_t> vblst(1);
        idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
        blst.populate();
        op_coulomb op;
        {
            motran_2e3c_incore_result_container<double> buf(av_buff_ao);
            scr_null<bat_2e3c_shellpair_cgto<double>> scr;
            motran_2e3c<double, double> mot(op, m_b3, scr, m_dev);
            mot.set_trn(Unit);
            mot.run(m_dev, blst, buf);
        }
        arma::Mat<double> V_Pab(arrays<double>::ptr(av_buff_ao), Unit.n_cols * Unit.n_cols, n_aux, false, true);
        mem.load_state(chkpt);
        
        
        arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
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
        
        // sigma_JG
        sigma_JG += Lam_pA.st() * JG;
        
        
        //transformed vector
        arma::mat sigma (n_vir, n_occ, fill::zeros);
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {
                
                // sigma_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        sigma_H(a,i) -= Y_bar[(a*n_occ*n_aux+k*n_aux+P)]
                                            * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }
    
                sigma(a,i) = sigma_0(a,i) + sigma_JG(a,i) + sigma_H(a,i) + sigma_I(a,i);
                excit += (sigma(a,i)*r1(a,i)) / pow(norm(r1,"fro"),2);

            }
        }
        
        
        // update of the trial vector
        residual.zeros();
        arma::mat update (n_vir, n_occ, fill::zeros);
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {
                
                double delta_ia = e_orb(i) - e_orb[n_occ+a];
                residual(a,i) = (sigma(a,i) - (excit*r1(a,i))) / norm(r1,"fro");
                update(a,i) = residual(a,i) / delta_ia;
                r1(a,i) = (r1(a,i) + update(a,i)) / norm(r1,"fro");
                
            }
        }
       
        exci = excit;
        
    }

}
#endif





////template class ri_eomee_r<double, double>;
template class ri_eomee_r<complex<double>, double>;
template class ri_eomee_r<complex<double>, complex<double>>;

}
