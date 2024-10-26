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
#include "ri_eomea_unr_r.h"
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

/// GPP: RI-EOMEA-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: OPTIMIZED CODE
template<>
void ri_eomea_unr_r<double,double>::css_unrestricted_energy(
    double& exci, const size_t& n_occa, const size_t& n_vira,
    const size_t& n_occb, const size_t& n_virb,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQpo_a, Mat<double> &BQvp_a, 
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoo_b,
    Mat<double> &BQpo_b, Mat<double> &BQvp_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a,
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b,
    Col<double> &r1a, Col<double> &r1b,
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Col<double> &sigma_a, Col<double> &sigma_b) {
    
    // memory_pool<double> mem(av_buff_ao);
    // typename memory_pool<double>::checkpoint chkpt = mem.save_state();

    // size_t npairs = (n_occ+1)*n_occ/2;
    // std::vector<size_t> occ_i2(npairs);
    // idx2_list pairs(n_occ, n_occ, npairs,
    //     array_view<size_t>(&occ_i2[0], occ_i2.size()));
    // for(size_t i = 0, ij = 0; i < n_occ; i++) {
    // for(size_t j = 0; j <= i; j++, ij++)
    //     pairs.set(ij, idx2(i, j));
    // }
    
    {
        // arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
        // arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ_a (n_aux, fill::zeros);
        iQ_a += BQov_a * vectorise(t1a);
        arma::vec iQ_b (n_aux, fill::zeros);
        iQ_b += BQov_b * vectorise(t1b);

        // Fvv_hat
        arma::Mat<double> F3a = (iQ_a.st() * BQvp_a) + (iQ_b.st() * BQvp_a);
        arma::Mat<double> Fvv_hat1_a(F3a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;
        arma::Mat<double> sigma_0_a = Fvv_hat_a*r1a;

        arma::Mat<double> F3b = (iQ_b.st() * BQvp_b) + (iQ_a.st() * BQvp_b);
        arma::Mat<double> Fvv_hat1_b(F3b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> BQvo(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQpo(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> Fvv_hat2_b = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;
        arma::Mat<double> sigma_0_b = Fvv_hat_b*r1b;
        
        //transformed vector
        #pragma omp parallel
        {
            // double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {

                sigma_a(a) = sigma_0_a(a);

            }
        }

        #pragma omp parallel
        {
            // double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {

                sigma_b(a) = sigma_0_b(a);

            }
        }
        // sigma_a.print("sigma_a in css_unrestricted_energy");
        // sigma_b.print("sigma_b in css_unrestricted_energy");
    }
}

template<>
void ri_eomea_unr_r<double,double>::davidson_unrestricted_energy(
    double& exci, const size_t& n_occa, const size_t& n_vira,
    const size_t& n_occb, const size_t& n_virb,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQpo_a, Mat<double> &BQvp_a, 
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoo_b,
    Mat<double> &BQpo_b, Mat<double> &BQvp_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a,
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b,
    Col<double> &r1a, Col<double> &r1b,
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Col<double> &sigma_a, Col<double> &sigma_b) {
    
    memory_pool<double> mem(av_buff_ao);
    typename memory_pool<double>::checkpoint chkpt = mem.save_state();
    
    {
       
         
        //double excit=0.0;
        //r1a.print("r1a");
        //r1b.print("r1b");

        arma::vec sigma_JG_a (n_vira, fill::zeros);
        arma::vec sigma_I_a (n_vira, fill::zeros);
        arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
        arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
        arma::mat Y_bar_a (n_aux, n_vira, fill::zeros);

        arma::vec sigma_JG_b (n_virb, fill::zeros);
        arma::vec sigma_I_b (n_virb, fill::zeros);
        arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
        arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
        arma::mat Y_bar_b (n_aux, n_virb, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ_a (n_aux, fill::zeros);
        iQ_a += BQov_a * vectorise(t1a);

        arma::vec iQ_b (n_aux, fill::zeros);
        iQ_b += BQov_b * vectorise(t1b);


        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);


        // Fov_hat
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;
        

        // Fvv_hat
        arma::Mat<double> F3a = (iQ_a.st() * BQvp_a) + (iQ_b.st() * BQvp_a);
        arma::Mat<double> Fvv_hat1_a(F3a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;

        arma::Mat<double> F3b = (iQ_b.st() * BQvp_b) + (iQ_a.st() * BQvp_b);
        arma::Mat<double> Fvv_hat1_b(F3b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;


        /// step 4:
        // (AA|BB)
        #pragma omp parallel
        {

            size_t npairs = n_occa*n_occb;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occa, n_occb, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occa; i++) {
            for(size_t j = 0; j < n_occb; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }
            
            arma::mat Yia_a_local (n_aux, n_vira*n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
            }
        } // end parallel (1)


        // (BB|AA)
        #pragma omp parallel
        {
            
            size_t npairs = n_occb*n_occa;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occb, n_occa, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occb; i++) {
            for(size_t j = 0; j < n_occa; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            arma::mat Yia_b_local (n_aux, n_virb*n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += c_os * t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
            }
        } // end parallel (2)
        
        //(AA|AA)
        #pragma omp parallel
        {

            size_t npairs = (n_occa+1)*n_occa/2;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occa, n_occa, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occa; i++) {
            for(size_t j = 0; j <= i; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            arma::mat Yia_a_local (n_aux, n_vira*n_occa, fill::zeros);

            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                
                double delta_ij = eA(i) + eA(j);
                double t2ab = 0.0;
                double t2ba = 0.0;
                
                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    for(size_t a = 0; a < n_vira; a++) {
                        t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        t2ba = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Yia)
            {
                Yia_a += Yia_a_local;
            }
        }

        //(BB|BB)
        #pragma omp parallel
        {

            size_t npairs = (n_occb+1)*n_occb/2;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occb, n_occb, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occb; i++) {
            for(size_t j = 0; j <= i; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }
        
            arma::mat Yia_b_local (n_aux, n_virb*n_occb, fill::zeros);

            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                
                double delta_ij = eB(i) + eB(j);
                double t2ab = 0.0;
                double t2ba = 0.0;

                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        t2ab = w0b[a] / (dijb - eB[n_occb+a]);
                        t2ba = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Yia)
            {
                Yia_b += Yia_b_local;
            }
        }

        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab

        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        
        arma::Mat<double> sigma_0_a = E_vv_a*r1a;
        arma::Mat<double> sigma_0_b = E_vv_b*r1b;


        // (AA|BB)
        #pragma omp parallel
        {
            arma::Mat<double> BQvpA(BQvp_a.memptr(), n_aux*n_vira, n_vira, false, true);
            arma::Mat<double> BQaA = BQvpA * r1a; // (n_aux*n_vir, n_vir)*n_vir
            BQaA.reshape(n_aux,n_vira);

            arma::vec sigma_I_a_local (n_vira, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        
                        //denominator
                        double delta_ijab = eB(j) - eA[n_occa+a] - eB[n_occb+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaA[(a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // r_ijba += BQaA[(b*n_aux+Q)]*BQhp_a[(j*n_virb*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        // r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_a_local[(a*n_aux+P)] += c_os * r_ijab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_a_local(a) += c_os * r_ijab * Fov_hat_b(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        }

        // (BB|AA)
        #pragma omp parallel
        {

            arma::Mat<double> BQvpB(BQvp_b.memptr(), n_aux*n_virb, n_virb, false, true);
            arma::Mat<double> BQaB = BQvpB * r1b; // (n_aux*n_vir, n_vir)*n_vir
            BQaB.reshape(n_aux,n_virb);

            arma::vec sigma_I_b_local (n_virb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        
                        //denominator
                        double delta_ijab = eA(j) - eB[n_occb+a] - eA[n_occa+b];
                        double r_ijab = 0.0;
                        // double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaB[(a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // r_ijba += BQaB[(b*n_aux+Q)]*BQhp_b[(j*n_vira*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        // r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_b_local[(a*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_b_local(a) += c_os * r_ijab * Fov_hat_a(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        }



        //(AA|AA)
        #pragma omp parallel
        {
            arma::Mat<double> BQvpA(BQvp_a.memptr(), n_aux*n_vira, n_vira, false, true);
            arma::Mat<double> BQaA = BQvpA * r1a; // (n_aux*n_vir, n_vir)*n_vir
            BQaA.reshape(n_aux,n_vira);

            arma::vec sigma_I_a_local (n_vira, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        
                        //denominator
                        double delta_ijab = eA(j) - eA[n_occa+a] - eA[n_occa+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaA[(a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            r_ijba += BQaA[(b*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_a_local[(a*n_aux+P)] += c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_a_local(a) += c_ss * (r_ijab - r_ijba) * Fov_hat_a(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        }

        //(BB|BB)
        #pragma omp parallel
        {

            arma::Mat<double> BQvpB(BQvp_b.memptr(), n_aux*n_virb, n_virb, false, true);
            arma::Mat<double> BQaB = BQvpB * r1b; // (n_aux*n_vir, n_vir)*n_vir
            BQaB.reshape(n_aux,n_virb);

            arma::vec sigma_I_b_local (n_virb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        
                        //denominator
                        double delta_ijab = eB(j) - eB[n_occb+a] - eB[n_occb+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaB[(a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            r_ijba += BQaB[(b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_b_local[(a*n_aux+P)] += c_ss * (r_ijab - r_ijba) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_b_local(a) += c_ss * (r_ijab - r_ijba) * Fov_hat_b(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        }



        /// step 5:
        arma::Mat<double> gamma_G_a = Y_bar_a * CvirtA.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)
        arma::Mat<double> gamma_G_b = Y_bar_b * CvirtB.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P_a (n_aux, n_orb, fill::zeros);
        arma::Mat<double> gamma_P_b (n_aux, n_orb, fill::zeros);
        gamma_P_a = PQinvhalf * gamma_G_a; // (n_aux,n_aux)*(n_aux,n_orb)
        gamma_P_b = PQinvhalf * gamma_G_b; // (n_aux,n_aux)*(n_aux,n_orb)

        
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

        arma::vec JG_a (n_orb, fill::zeros);  
        #pragma omp parallel
        {
            arma::vec JG_a_local (n_orb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_a_local(alpha) += gamma_P_a[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                                    // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG_a += JG_a_local;
            }
        }

        arma::vec JG_b (n_orb, fill::zeros);  
        #pragma omp parallel
        {
            arma::vec JG_b_local (n_orb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_b_local(alpha) += gamma_P_b[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                                    // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG_b += JG_b_local;
            }
        }


        /// step 6:
        sigma_JG_a += Lam_pA.st() * JG_a; //(orb,virt).t * orb = virt
        sigma_JG_b += Lam_pB.st() * JG_b; //(orb,virt).t * orb = virt
        
        //transformed vector
        // arma::vec sigma (n_vir, fill::zeros);
        #pragma omp parallel
        {
            // double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {

                sigma_a(a) = sigma_0_a(a) + sigma_I_a(a) + sigma_JG_a(a);

            }
        }

        #pragma omp parallel
        {
            // double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {

                sigma_b(a) = sigma_0_b(a) + sigma_I_b(a) + sigma_JG_b(a);

            }
        }
    }
}


template<>
void ri_eomea_unr_r<double,double>::diis_unrestricted_energy(
    double& exci, const size_t& n_occa, const size_t& n_vira,
    const size_t& n_occb, const size_t& n_virb,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQpo_a, Mat<double> &BQvp_a, 
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoo_b,
    Mat<double> &BQpo_b, Mat<double> &BQvp_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a,
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b,
    Col<double> &r1a, Col<double> &r1b,
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Col<double> &sigma_a, Col<double> &sigma_b) {
    
    memory_pool<double> mem(av_buff_ao);
    typename memory_pool<double>::checkpoint chkpt = mem.save_state();

    
    {
        
        //cout << "exci beginning: " << exci << endl;
        //double excit=0.0;

        arma::vec sigma_JG_a (n_vira, fill::zeros);
        arma::vec sigma_I_a (n_vira, fill::zeros);
        arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
        arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
        arma::mat Y_bar_a (n_aux, n_vira, fill::zeros);

        arma::vec sigma_JG_b (n_virb, fill::zeros);
        arma::vec sigma_I_b (n_virb, fill::zeros);
        arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
        arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
        arma::mat Y_bar_b (n_aux, n_virb, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ_a (n_aux, fill::zeros);
        iQ_a += BQov_a * vectorise(t1a);

        arma::vec iQ_b (n_aux, fill::zeros);
        iQ_b += BQov_b * vectorise(t1b);


        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);


        // Fov_hat
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;
        

        // Fvv_hat
        arma::Mat<double> F3a = (iQ_a.st() * BQvp_a) + (iQ_b.st() * BQvp_a);
        arma::Mat<double> Fvv_hat1_a(F3a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;

        arma::Mat<double> F3b = (iQ_b.st() * BQvp_b) + (iQ_a.st() * BQvp_b);
        arma::Mat<double> Fvv_hat1_b(F3b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;


        /// step 4:

        // (AA|BB)
        #pragma omp parallel
        {

            size_t npairs = n_occa*n_occb;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occa, n_occb, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occa; i++) {
            for(size_t j = 0; j < n_occb; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }
            
            arma::mat Yia_a_local (n_aux, n_vira*n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
            }
        } // end parallel (1)


        // (BB|AA)
        #pragma omp parallel
        {
            
            size_t npairs = n_occb*n_occa;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occb, n_occa, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occb; i++) {
            for(size_t j = 0; j < n_occa; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            arma::mat Yia_b_local (n_aux, n_virb*n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += c_os * t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
            }
        } // end parallel (2)
        
        //(AA|AA)
        #pragma omp parallel
        {

            size_t npairs = (n_occa+1)*n_occa/2;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occa, n_occa, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occa; i++) {
            for(size_t j = 0; j <= i; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            arma::mat Yia_a_local (n_aux, n_vira*n_occa, fill::zeros);

            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                
                double delta_ij = eA(i) + eA(j);
                double t2ab = 0.0;
                double t2ba = 0.0;
                
                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    for(size_t a = 0; a < n_vira; a++) {
                        t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        t2ba = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Yia_a)
            {
                Yia_a += Yia_a_local;
            }
        }

        //(BB|BB)
        #pragma omp parallel
        {

            size_t npairs = (n_occb+1)*n_occb/2;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occb, n_occb, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occb; i++) {
            for(size_t j = 0; j <= i; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }
        
            arma::mat Yia_b_local (n_aux, n_virb*n_occb, fill::zeros);

            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2: 
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                
                double delta_ij = eB(i) + eB(j);
                double t2ab = 0.0;
                double t2ba = 0.0;

                const double *w0 = W0.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        t2ab = w0b[a] / (dijb - eB[n_occb+a]);
                        t2ba = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += c_ss * (t2ab-t2ba) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }
                    }
                }
            }
            #pragma omp critical (Yia_b)
            {
                Yia_b += Yia_b_local;
            }
        }


        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab

        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        
        arma::Mat<double> sigma_0_a = E_vv_a*r1a;
        arma::Mat<double> sigma_0_b = E_vv_b*r1b;


        // (AA|BB)
        #pragma omp parallel
        {
            arma::Mat<double> BQvpA(BQvp_a.memptr(), n_aux*n_vira, n_vira, false, true);
            arma::Mat<double> BQaA = BQvpA * r1a; // (n_aux*n_vir, n_vir)*n_vir
            BQaA.reshape(n_aux,n_vira);

            arma::vec sigma_I_a_local (n_vira, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        
                        //denominator
                        double delta_ijab = eB(j) - eA[n_occa+a] - eB[n_occb+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaA[(a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // r_ijba += BQaA[(b*n_aux+Q)]*BQhp_a[(j*n_virb*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        // r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_a_local[(a*n_aux+P)] += c_os * r_ijab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_a_local(a) += c_os * r_ijab * Fov_hat_b(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        }

        // (BB|AA)
        #pragma omp parallel
        {

            arma::Mat<double> BQvpB(BQvp_b.memptr(), n_aux*n_virb, n_virb, false, true);
            arma::Mat<double> BQaB = BQvpB * r1b; // (n_aux*n_vir, n_vir)*n_vir
            BQaB.reshape(n_aux,n_virb);

            arma::vec sigma_I_b_local (n_virb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        
                        //denominator
                        double delta_ijab = eA(j) - eB[n_occb+a] - eA[n_occa+b];
                        double r_ijab = 0.0;
                        // double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaB[(a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // r_ijba += BQaB[(b*n_aux+Q)]*BQhp_b[(j*n_vira*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        // r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_b_local[(a*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_b_local(a) += c_os * r_ijab * Fov_hat_a(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        }



        //(AA|AA)
        #pragma omp parallel
        {
            arma::Mat<double> BQvpA(BQvp_a.memptr(), n_aux*n_vira, n_vira, false, true);
            arma::Mat<double> BQaA = BQvpA * r1a; // (n_aux*n_vir, n_vir)*n_vir
            BQaA.reshape(n_aux,n_vira);

            arma::vec sigma_I_a_local (n_vira, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        
                        //denominator
                        double delta_ijab = eA(j) - eA[n_occa+a] - eA[n_occa+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaA[(a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            r_ijba += BQaA[(b*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_a_local[(a*n_aux+P)] += c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_a_local(a) += c_ss * (r_ijab - r_ijba) * Fov_hat_a(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        }

        //(BB|BB)
        #pragma omp parallel
        {

            arma::Mat<double> BQvpB(BQvp_b.memptr(), n_aux*n_virb, n_virb, false, true);
            arma::Mat<double> BQaB = BQvpB * r1b; // (n_aux*n_vir, n_vir)*n_vir
            BQaB.reshape(n_aux,n_virb);

            arma::vec sigma_I_b_local (n_virb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        
                        //denominator
                        double delta_ijab = eB(j) - eB[n_occb+a] - eB[n_occb+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQaB[(a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            r_ijba += BQaB[(b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            Y_bar_b_local[(a*n_aux+P)] += c_ss * (r_ijab - r_ijba) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        sigma_I_b_local(a) += c_ss * (r_ijab - r_ijba) * Fov_hat_b(j,b);
                        
                    }
                }
            }
            #pragma omp critical (YI)
            {
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        }



        /// step 5:
        arma::Mat<double> gamma_G_a = Y_bar_a * CvirtA.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)
        arma::Mat<double> gamma_G_b = Y_bar_b * CvirtB.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P_a (n_aux, n_orb, fill::zeros);
        arma::Mat<double> gamma_P_b (n_aux, n_orb, fill::zeros);
        gamma_P_a = PQinvhalf * gamma_G_a; // (n_aux,n_aux)*(n_aux,n_orb)
        gamma_P_b = PQinvhalf * gamma_G_b; // (n_aux,n_aux)*(n_aux,n_orb)


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

        arma::vec JG_a (n_orb, fill::zeros);  
        #pragma omp parallel
        {
            arma::vec JG_a_local (n_orb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_a_local(alpha) += gamma_P_a[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                                    // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG_a += JG_a_local;
            }
        }

        arma::vec JG_b (n_orb, fill::zeros);  
        #pragma omp parallel
        {
            arma::vec JG_b_local (n_orb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_b_local(alpha) += gamma_P_b[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                                    // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG_b += JG_b_local;
            }
        }

        /// step 6:
        // vec a = vectorise(r1a);
        // vec b = vectorise(r1b);
        vec c = join_cols(r1a,r1b);
        // c.print("c");

        sigma_JG_a += Lam_pA.st() * JG_a; //(orb,virt).t * orb = virt
        sigma_JG_b += Lam_pB.st() * JG_b; //(orb,virt).t * orb = virt
        
        //transformed vector
        #pragma omp parallel
        {
            // double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {

                sigma_a(a) = sigma_0_a(a) + sigma_I_a(a) + sigma_JG_a(a);
                // sigma_a(a) = sigma_0_a(a);

            }
        }

        #pragma omp parallel
        {
            // double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {

                sigma_b(a) = sigma_0_b(a) + sigma_I_b(a) + sigma_JG_b(a);
                // sigma_b(a) = sigma_0_b(a);

            }
        }
        
        // sigma_a.print("sigma_a");
        // sigma_b.print("sigma_b");
        
        exci = (accu(sigma_a % r1a) + accu(sigma_b % r1b)) / pow(norm(c,"fro"),2);
        // cout << "exci: " << exci << endl;
        
        // update of the trial vector
        #pragma omp parallel
        {
            arma::vec residual_a (n_vira, fill::zeros);
            arma::vec update_a (n_vira, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                
                double delta_a = -eA[n_occa+a];
                residual_a(a) = (sigma_a(a) - (exci*r1a(a))) / norm(c,"fro");
                update_a(a) = residual_a(a) / delta_a;
                r1a(a) = (r1a(a) + update_a(a)) / norm(c,"fro");
                
            }
            // residual_a.print("residual_a");
            // update_a.print("update_a");
            // r1a.print("r1a");
        }
        
        #pragma omp parallel
        {
            arma::vec residual_b (n_virb, fill::zeros);
            arma::vec update_b (n_virb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                
                double delta_b = -eB[n_occb+a];
                residual_b(a) = (sigma_b(a) - (exci*r1b(a))) / norm(c,"fro");
                update_b(a) = residual_b(a) / delta_b;
                r1b(a) = (r1b(a) + update_b(a)) / norm(c,"fro");
                
            }
            // residual_b.print("residual_b");
            // update_b.print("update_b");
            // r1b.print("r1b");
        }

        //cout << "exci final: " << exci << endl;
        // exci = excit;
    }
}

template class ri_eomea_unr_r<double, double>;
template class ri_eomea_unr_r<complex<double>, double>;
template class ri_eomea_unr_r<complex<double>, complex<double>>;

}
