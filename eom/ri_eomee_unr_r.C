#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <armadillo>
#include <libposthf/motran/motran_2e3c.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/arrays/memory_pool.h>
#include "ri_eomee_unr_r.h"
#include <libgmbpt/util/dig_2e3c.h>
#include <libgmbpt/util/dig_2e3c_aux.h>
#include <libgmbpt/util/scr_2e3c.h>
#include <complex>

namespace libgmbpt{
using namespace libposthf;
using namespace libqints;
using namespace arma;
using namespace std;


template<>
void ri_eomee_unr_r<double,double>::ccs_unrestricted_energy(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a,
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQob_b, Mat<double> &BQpo_b, 
    Mat<double> &BQhb_b, Mat<double> &BQbp_b,
    Mat<double> &BQpv_a, Mat<double> &BQpv_b, 
    Mat<double> &V_Pab,  
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar, 
    Mat<double> &Lam_hB_bar, Mat<double> &Lam_pB_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a, 
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b, 
    Mat<double> &r1a, Mat<double> &r1b,  
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_a, Mat<double> &sigma_b) {

    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::vec iQ_bar_a (n_aux, fill::zeros);
    arma::mat sigma_0_a (n_vira, n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat sigma_JG_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_H_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_I_a (n_vira, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Y_bar_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::vec iQ_bar_b (n_aux, fill::zeros);
    arma::mat sigma_0_b (n_virb, n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat sigma_JG_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_H_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_I_b (n_virb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Y_bar_b (n_aux, n_virb*n_occb, fill::zeros);
    
    {   
        exci = 0; 
        double t2ab = 0.0, t2ba = 0.0, t2aa = 0.0, t2bb = 0.0, t2aa_2 = 0.0, t2bb_2 = 0.0;
        double r2ab = 0.0, r2ba = 0.0, r2aa = 0.0, r2bb = 0.0, r2aa_2 = 0.0, r2bb_2 = 0.0;


        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        // iQ, iQ_bar,
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);
        iQ_bar_a += BQov_a * vectorise(r1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);
        iQ_bar_b += BQov_b * vectorise(r1b);


        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobA(BQob_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQobB(BQob_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);


        // Fov_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|AA), (BB|AA)
        arma::Mat<double> F2a = (iQ_bar_a.st() * BQov_a) + (iQ_bar_b.st() * BQov_a);
        arma::Mat<double> F22a(F2a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_bar1_a = F22a.st();
        arma::Mat<double> Fov_bar2_a = BQobA.st() * BQvoA;
        arma::Mat<double> Fov_bar_a = Fov_bar1_a - Fov_bar2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F2b = (iQ_bar_b.st() * BQov_b) + (iQ_bar_a.st() * BQov_b);
        arma::Mat<double> F22b(F2b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_bar1_b = F22b.st();
        arma::Mat<double> Fov_bar2_b = BQobB.st() * BQvoB;
        arma::Mat<double> Fov_bar_b = Fov_bar1_b - Fov_bar2_b;

        // Fvv_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F3a = (iQ_a.st() * BQpv_a) + (iQ_b.st() * BQpv_a);
        arma::Mat<double> F33a(F3a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat1_a = F33a.st();
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F3b = (iQ_b.st() * BQpv_b) + (iQ_a.st() * BQpv_b);
        arma::Mat<double> F33b(F3b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat1_b = F33b.st();
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;

        // Foo_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F4a = (iQ_a.st() * BQoh_a) + (iQ_b.st() * BQoh_a);
        arma::Mat<double> F44a(F4a.memptr(), n_occa, n_occa, false, true);
        arma::Mat<double> Foo_hat1_a = F44a.st();
        arma::Mat<double> Foo_hat2_a = BQooA.st() * BQhoA;
        arma::Mat<double> Foo_hat_a = f_oo_a + Foo_hat1_a - Foo_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F4b = (iQ_b.st() * BQoh_b) + (iQ_a.st() * BQoh_b);
        arma::Mat<double> F44b(F4b.memptr(), n_occb, n_occb, false, true);
        arma::Mat<double> Foo_hat1_b = F44b.st();
        arma::Mat<double> Foo_hat2_b = BQooB.st() * BQhoB;
        arma::Mat<double> Foo_hat_b = f_oo_b + Foo_hat1_b - Foo_hat2_b;


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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: BJai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: BJai
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        //double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        //double r2ab = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eA[n_occa+a] + exci);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                 

                        // sigma_I_a(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);
                        sigma_I_a_local(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: bjAI
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: bjAI
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        //double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        //double r2ba = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eB[n_occb+a] + exci);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                                            
                        // sigma_I_b(a,i) += r2ab * Fov_hat_a(j,b) + t2ab * Fov_bar_a(j,b);
                        sigma_I_b_local(a,i) += r2ba * Fov_hat_a(j,b) + t2ba * Fov_bar_a(j,b);
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        //double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        //double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        //double r2aa = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                        //double r2aa_2 = (w1[a*n_vira+b] + w2b[a] + w3[a*n_vira+b] + w4b[a]) / (dijb - eA[n_occa+a] + exci);


                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_a(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        // sigma_I_a(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));
                        sigma_I_a_local(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        sigma_I_a_local(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end parallel (3)


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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        //double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        //double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        
                        //double r2bb = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                        //double r2bb_2 = (w1[a*n_virb+b] + w2b[a] + w3[a*n_virb+b] + w4b[a]) / (dijb - eB[n_occb+a] + exci);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_b(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        // sigma_I_b(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));
                        sigma_I_b_local(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        sigma_I_b_local(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)


        /// step 5:
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);


        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar_a(Y_bar_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> gamma_G1a = YQia_bar_a * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occa; i++) {
            gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11a = (iQ_bar_a * vectorise(Lam_hA).st()) + (iQ_bar_b * vectorise(Lam_hA).st());
        // arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux*n_occa, n_orb, false, true);
        arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux, n_orb*n_occa, false, true);

        // / omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occa; i++) {
            gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Qa = gamma_Ga + gamma_J1a - gamma_J2a;

        arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
        gamma_Pa = PQinvhalf * gamma_Qa;


        // arma::vec iP (n_aux, fill::zeros);
        // iP = (PQinvhalf * iQ_a) + (PQinvhalf * iQ_b);
        // (AA|AA), (BB|AA)
        // #pragma omp parallel
        // arma::Mat<double> F3_digestor_a (n_vira, n_vira, fill::zeros);
        // {

        //     // digestor
        //     arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        //     // arma::Mat<double> JG (n_orb, n_occa, fill::zeros);
        //     {

        //         //  Step 1: Read libqints-type basis set from files and form shellpair basis.
        //         // libqints::basis_1e2c_shellpair_cgto<double> bsp;
        //         // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
        //         const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
        //         const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
        //         size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
        //         size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
        //         size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
        //         //  Construct the 2e3c shellpair basis and corresponding full basis range
        //         libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

        //         //  Step 2: prepare required input settings
        //         libqints::dev_omp dev;                  //  libqints-type device information.
        //         size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
        //         dev.init(1024);
        //         dev.nthreads = 1;
        //         dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
        //         libqints::deriv_code dc;
        //         dc.set(0);                //  Set integral derivative level
        //         libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
        //         libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
        //         qjob.begin(fbr);                                //  Start the libqints job for full basis range

        //         //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
        //         scr_2e3c scr(m_b3);

        //         //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
        //         dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
        //         if (dev.memory * dev.nthreads > mem_total) {
        //             std::cout << " Given memory is not enough for computing integrals." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

        //         //  Step 5:
        //         //  Memory available for thread-local result arrays:
        //         size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
        //         //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
        //         size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
        //         {  //  The code block here should be replaced by estimating # of munu basis function pairs
        //             //  and/or # of auxiliary basis function.
        //             nbsp_per_subrange = nbsp;
        //             naux_per_subrange = n_aux;
        //         }
        //         //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
        //         //  of each munu shell pair.
        //         size_t min_nbsp_per_subrange = 0;
        //         #pragma omp for 
        //         for (size_t isp = 0; isp < nsp; isp++) {
        //             size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
        //             min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
        //         }
        //         if (nbsp_per_subrange < min_nbsp_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
        //         //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
        //         //  of each auxiliary shell.
        //         size_t min_naux_per_subrange = 0;
        //         for (size_t is_q = 0; is_q < ns_q; is_q++) {
        //             size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
        //             min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
        //         }
        //         if (naux_per_subrange < min_naux_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

        //         //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
        //         arma::vec Fvec(nbsp);
        //         //  Result will be accumulated in the output arrays, so we need to zero out them
        //         Fvec.zeros();
        //         JG_a.zeros(); 
        //         dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occa, gamma_Pa, JG_a);
        //         // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

        //         //  Step 7: Loop over basis subranges and run libqints job
        //         libqints::batching_info<2> binfo;
        //         libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
        //         libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
        //         for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
        //             //  Current basis subrange
        //             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
        //                 fbr, binfo.get_batch_window(biter.get_batch_number()));
        //             if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
        //                 std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
        //                 qjob.end();  //  End the libqints job before return
        //                 return;
        //             }
        //         }
        //         //  In case 2, we need to unpack F from vector form to matrix form with
        //         //  permutationally symmetric matrix elements are properly copied
        //         libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
        //         libaview::array_view<double> av_f(F.memptr(), F.n_elem);
        //         libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
        //         libaview::array_view<double> av_result(JG_a.memptr(), JG_a.n_elem);
        //     }
        // } // end (AA|AA), (BB|AA)
        //     F3_digestor_a = CvirtA.st() * F * Lam_pA;
            
        arma::mat JG_a_local (n_orb, n_occa, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        // JG_a(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                        //                 * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        JG_a_local(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG_a)
        {
            JG_a += JG_a_local;
        }

        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);

        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab
        E_oo_a = Foo_hat_a + (YQaiA.st() * BQovA).st(); // E_ji

        sigma_0_a += (E_vv_a*r1a) - (r1a*E_oo_a);


        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar_b(Y_bar_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> gamma_G1b = YQia_bar_b * CvirtB.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occb; i++) {
            gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11b = (iQ_bar_b * vectorise(Lam_hB).st()) + (iQ_bar_a * vectorise(Lam_hB).st());
        arma::Mat<double> gamma_J1b(gamma_J11b.memptr(), n_aux*n_occb, n_orb, false, true);

        // / omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occb; i++) {
            gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Qb = gamma_Gb + gamma_J1b - gamma_J2b;

        arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
        gamma_Pb = PQinvhalf * gamma_Qb;

        // (BB|BB), (AA|BB)
        // #pragma omp parallel
        // arma::Mat<double> F3_digestor_b (n_virb, n_virb, fill::zeros);
        // {
   

        //     arma::vec iP (n_aux, fill::zeros);
        //     // iP = PQinvhalf * iQ_b;
        //     iP = (PQinvhalf * iQ_b) + (PQinvhalf * iQ_a);

        //     // digestor
        //     arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        //     // arma::Mat<double> JG (n_orb, n_occb, fill::zeros);
        //     {

        //         //  Step 1: Read libqints-type basis set from files and form shellpair basis.
        //         // libqints::basis_1e2c_shellpair_cgto<double> bsp;
        //         // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
        //         const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
        //         const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
        //         size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
        //         size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
        //         size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
        //         //  Construct the 2e3c shellpair basis and corresponding full basis range
        //         libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

        //         //  Step 2: prepare required input settings
        //         libqints::dev_omp dev;                  //  libqints-type device information.
        //         size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
        //         dev.init(1024);
        //         dev.nthreads = 1;
        //         dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
        //         libqints::deriv_code dc;
        //         dc.set(0);                //  Set integral derivative level
        //         libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
        //         libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
        //         qjob.begin(fbr);                                //  Start the libqints job for full basis range

        //         //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
        //         scr_2e3c scr(m_b3);

        //         //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
        //         dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
        //         if (dev.memory * dev.nthreads > mem_total) {
        //             std::cout << " Given memory is not enough for computing integrals." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

        //         //  Step 5:
        //         //  Memory available for thread-local result arrays:
        //         size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
        //         //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
        //         size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
        //         {  //  The code block here should be replaced by estimating # of munu basis function pairs
        //             //  and/or # of auxiliary basis function.
        //             nbsp_per_subrange = nbsp;
        //             naux_per_subrange = n_aux;
        //         }
        //         //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
        //         //  of each munu shell pair.
        //         size_t min_nbsp_per_subrange = 0;
        //         #pragma omp for 
        //         for (size_t isp = 0; isp < nsp; isp++) {
        //             size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
        //             min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
        //         }
        //         if (nbsp_per_subrange < min_nbsp_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
        //         //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
        //         //  of each auxiliary shell.
        //         size_t min_naux_per_subrange = 0;
        //         for (size_t is_q = 0; is_q < ns_q; is_q++) {
        //             size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
        //             min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
        //         }
        //         if (naux_per_subrange < min_naux_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

        //         //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
        //         arma::vec Fvec(nbsp);
        //         //  Result will be accumulated in the output arrays, so we need to zero out them
        //         Fvec.zeros();
        //         JG_b.zeros(); 
        //         dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occb, gamma_Pb, JG_b);
        //         // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

        //         //  Step 7: Loop over basis subranges and run libqints job
        //         libqints::batching_info<2> binfo;
        //         libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
        //         libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
        //         for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
        //             //  Current basis subrange
        //             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
        //                 fbr, binfo.get_batch_window(biter.get_batch_number()));
        //             if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
        //                 std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
        //                 qjob.end();  //  End the libqints job before return
        //                 return;
        //             }
        //         }
        //         //  In case 2, we need to unpack F from vector form to matrix form with
        //         //  permutationally symmetric matrix elements are properly copied
        //         libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
        //         libaview::array_view<double> av_f(F.memptr(), F.n_elem);
        //         libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
        //         libaview::array_view<double> av_result(JG_b.memptr(), JG_b.n_elem);

        //     }
        // } // end (BB|BB), (AA|BB)
        //     F3_digestor_b = CvirtB.st() * F * Lam_pB;
            
        arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        // JG_b(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                        //                 * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        JG_b_local(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG_b)
        {
            JG_b += JG_b_local;
        }


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        E_oo_b = Foo_hat_b + (YQaiB.st() * BQovB).st(); // E_ji

        sigma_0_b += (E_vv_b*r1b) - (r1b*E_oo_b);

        vec a = vectorise(r1a);
        vec b = vectorise(r1b);
        vec c = join_cols(a,b);

        /// step 6:

        // sigma_JG
        sigma_JG_a += Lam_pA.st() * JG_a;
        // cout << "sigma_JG_a: " << accu(sigma_JG_a) << endl;

        // (AA|AA)
        #pragma omp parallel
        {
       
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_a(a,i) -= Y_bar_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_a(a,i) = sigma_0_a(a,i) + sigma_JG_a(a,i) + sigma_H_a(a,i) + sigma_I_a(a,i);

                }
            }
        } // end (AA|AA)

        // sigma_JG
        sigma_JG_b += Lam_pB.st() * JG_b;

        
        // (BB|BB)
        #pragma omp parallel
        {
                 
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_b(a,i) -= Y_bar_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_b(a,i) = sigma_0_b(a,i) + sigma_JG_b(a,i) + sigma_H_b(a,i) + sigma_I_b(a,i);

                }
            }
        } // end (BB|BB)
        
    }
}


template<>
void ri_eomee_unr_r<double,double>::davidson_unrestricted_energy(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a,
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQob_b, Mat<double> &BQpo_b, 
    Mat<double> &BQhb_b, Mat<double> &BQbp_b,
    Mat<double> &BQpv_a, Mat<double> &BQpv_b, 
    Mat<double> &V_Pab,  
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar, 
    Mat<double> &Lam_hB_bar, Mat<double> &Lam_pB_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a, 
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b, 
    Mat<double> &r1a, Mat<double> &r1b,  
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_a, Mat<double> &sigma_b) {


    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::vec iQ_bar_a (n_aux, fill::zeros);
    arma::mat sigma_0_a (n_vira, n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat sigma_JG_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_H_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_I_a (n_vira, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Y_bar_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::vec iQ_bar_b (n_aux, fill::zeros);
    arma::mat sigma_0_b (n_virb, n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat sigma_JG_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_H_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_I_b (n_virb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Y_bar_b (n_aux, n_virb*n_occb, fill::zeros);
    
    {   

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);
        iQ_bar_a += BQov_a * vectorise(r1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);
        iQ_bar_b += BQov_b * vectorise(r1b);


        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobA(BQob_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQobB(BQob_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);


        // Fov_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|AA), (BB|AA)
        arma::Mat<double> F2a = (iQ_bar_a.st() * BQov_a) + (iQ_bar_b.st() * BQov_a);
        arma::Mat<double> F22a(F2a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_bar1_a = F22a.st();
        arma::Mat<double> Fov_bar2_a = BQobA.st() * BQvoA;
        arma::Mat<double> Fov_bar_a = Fov_bar1_a - Fov_bar2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F2b = (iQ_bar_b.st() * BQov_b) + (iQ_bar_a.st() * BQov_b);
        arma::Mat<double> F22b(F2b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_bar1_b = F22b.st();
        arma::Mat<double> Fov_bar2_b = BQobB.st() * BQvoB;
        arma::Mat<double> Fov_bar_b = Fov_bar1_b - Fov_bar2_b;


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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: BJai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: BJai
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        double r2ab = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eA[n_occa+a] + exci);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                 

                        // sigma_I_a(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);
                        sigma_I_a_local(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: bjAI
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: bjAI
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        double r2ba = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eB[n_occb+a] + exci);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                                            
                        // sigma_I_b(a,i) += r2ab * Fov_hat_a(j,b) + t2ab * Fov_bar_a(j,b);
                        sigma_I_b_local(a,i) += r2ba * Fov_hat_a(j,b) + t2ba * Fov_bar_a(j,b);
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        double r2aa = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                        double r2aa_2 = (w1[a*n_vira+b] + w2b[a] + w3[a*n_vira+b] + w4b[a]) / (dijb - eA[n_occa+a] + exci);


                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_a(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        // sigma_I_a(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));
                        sigma_I_a_local(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        sigma_I_a_local(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end parallel (3)


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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        
                        double r2bb = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                        double r2bb_2 = (w1[a*n_virb+b] + w2b[a] + w3[a*n_virb+b] + w4b[a]) / (dijb - eB[n_occb+a] + exci);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_b(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        // sigma_I_b(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));
                        sigma_I_b_local(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        sigma_I_b_local(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)


        /// step 5:
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar_a(Y_bar_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> gamma_G1a = YQia_bar_a * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occa; i++) {
            gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11a = (iQ_bar_a * vectorise(Lam_hA).st()) + (iQ_bar_b * vectorise(Lam_hA).st());
        arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux*n_occa, n_orb, false, true);

        // / omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occa; i++) {
            gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Qa = gamma_Ga + gamma_J1a - gamma_J2a;

        arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
        gamma_Pa = PQinvhalf * gamma_Qa;

        // (AA|AA), (BB|AA)
        // #pragma omp parallel
        // arma::Mat<double> F3_digestor_a (n_vira, n_vira, fill::zeros);
        // {
        //     arma::vec iP (n_aux, fill::zeros);
        //     // iP = PQinvhalf * iQ_a;
        //     iP = (PQinvhalf * iQ_a) + (PQinvhalf * iQ_b);

        //     // digestor
        //     arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        //     // arma::Mat<double> JG (n_orb, n_occa, fill::zeros);
        //     {

        //         //  Step 1: Read libqints-type basis set from files and form shellpair basis.
        //         // libqints::basis_1e2c_shellpair_cgto<double> bsp;
        //         // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
        //         const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
        //         const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
        //         size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
        //         size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
        //         size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
        //         //  Construct the 2e3c shellpair basis and corresponding full basis range
        //         libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

        //         //  Step 2: prepare required input settings
        //         libqints::dev_omp dev;                  //  libqints-type device information.
        //         size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
        //         dev.init(1024);
        //         dev.nthreads = 1;
        //         dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
        //         libqints::deriv_code dc;
        //         dc.set(0);                //  Set integral derivative level
        //         libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
        //         libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
        //         qjob.begin(fbr);                                //  Start the libqints job for full basis range

        //         //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
        //         scr_2e3c scr(m_b3);

        //         //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
        //         dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
        //         if (dev.memory * dev.nthreads > mem_total) {
        //             std::cout << " Given memory is not enough for computing integrals." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

        //         //  Step 5:
        //         //  Memory available for thread-local result arrays:
        //         size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
        //         //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
        //         size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
        //         {  //  The code block here should be replaced by estimating # of munu basis function pairs
        //             //  and/or # of auxiliary basis function.
        //             nbsp_per_subrange = nbsp;
        //             naux_per_subrange = n_aux;
        //         }
        //         //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
        //         //  of each munu shell pair.
        //         size_t min_nbsp_per_subrange = 0;
        //         #pragma omp for 
        //         for (size_t isp = 0; isp < nsp; isp++) {
        //             size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
        //             min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
        //         }
        //         if (nbsp_per_subrange < min_nbsp_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
        //         //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
        //         //  of each auxiliary shell.
        //         size_t min_naux_per_subrange = 0;
        //         for (size_t is_q = 0; is_q < ns_q; is_q++) {
        //             size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
        //             min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
        //         }
        //         if (naux_per_subrange < min_naux_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

        //         //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
        //         arma::vec Fvec(nbsp);
        //         //  Result will be accumulated in the output arrays, so we need to zero out them
        //         Fvec.zeros();
        //         JG_a.zeros(); 
        //         dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occa, gamma_Pa, JG_a);
        //         // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

        //         //  Step 7: Loop over basis subranges and run libqints job
        //         libqints::batching_info<2> binfo;
        //         libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
        //         libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
        //         for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
        //             //  Current basis subrange
        //             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
        //                 fbr, binfo.get_batch_window(biter.get_batch_number()));
        //             if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
        //                 std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
        //                 qjob.end();  //  End the libqints job before return
        //                 return;
        //             }
        //         }
        //         //  In case 2, we need to unpack F from vector form to matrix form with
        //         //  permutationally symmetric matrix elements are properly copied
        //         libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
        //         libaview::array_view<double> av_f(F.memptr(), F.n_elem);
        //         libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
        //         libaview::array_view<double> av_result(JG_a.memptr(), JG_a.n_elem);
        //     }
        // } // end (AA|AA), (BB|AA)
        //     F3_digestor_a = CvirtA.st() * F * Lam_pA;
            
        arma::mat JG_a_local (n_orb, n_occa, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        // JG_a(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                        //                 * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        JG_a_local(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG_a)
        {
            JG_a += JG_a_local;
        }


        // Fvv_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F3a = (iQ_a.st() * BQpv_a) + (iQ_b.st() * BQpv_a);
        arma::Mat<double> F33a(F3a.memptr(), n_vira, n_vira, false, true);
        // arma::Mat<double> F33a(F3_digestor_a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat1_a = F33a.st();
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;


        // Foo_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F4a = (iQ_a.st() * BQoh_a) + (iQ_b.st() * BQoh_a);
        arma::Mat<double> F44a(F4a.memptr(), n_occa, n_occa, false, true);
        arma::Mat<double> Foo_hat1_a = F44a.st();
        arma::Mat<double> Foo_hat2_a = BQooA.st() * BQhoA;
        arma::Mat<double> Foo_hat_a = f_oo_a + Foo_hat1_a - Foo_hat2_a;


        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);

        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab
        E_oo_a = Foo_hat_a + (YQaiA.st() * BQovA).st(); // E_ji

        sigma_0_a += (E_vv_a*r1a) - (r1a*E_oo_a);

        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar_b(Y_bar_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> gamma_G1b = YQia_bar_b * CvirtB.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occb; i++) {
            gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11b = (iQ_bar_b * vectorise(Lam_hB).st()) + (iQ_bar_a * vectorise(Lam_hB).st());
        arma::Mat<double> gamma_J1b(gamma_J11b.memptr(), n_aux*n_occb, n_orb, false, true);

        // / omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occb; i++) {
            gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Qb = gamma_Gb + gamma_J1b - gamma_J2b;

        arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
        gamma_Pb = PQinvhalf * gamma_Qb;


        // (BB|BB), (AA|BB)
        // #pragma omp parallel
        // arma::Mat<double> F3_digestor_b (n_virb, n_virb, fill::zeros);
        // {


        //     arma::vec iP (n_aux, fill::zeros);
        //     // iP = PQinvhalf * iQ_b;
        //     iP = (PQinvhalf * iQ_b) + (PQinvhalf * iQ_a);

        //     // digestor
        //     arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        //     // arma::Mat<double> JG (n_orb, n_occb, fill::zeros);
        //     {

        //         //  Step 1: Read libqints-type basis set from files and form shellpair basis.
        //         // libqints::basis_1e2c_shellpair_cgto<double> bsp;
        //         // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
        //         const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
        //         const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
        //         size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
        //         size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
        //         size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
        //         //  Construct the 2e3c shellpair basis and corresponding full basis range
        //         libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

        //         //  Step 2: prepare required input settings
        //         libqints::dev_omp dev;                  //  libqints-type device information.
        //         size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
        //         dev.init(1024);
        //         dev.nthreads = 1;
        //         dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
        //         libqints::deriv_code dc;
        //         dc.set(0);                //  Set integral derivative level
        //         libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
        //         libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
        //         qjob.begin(fbr);                                //  Start the libqints job for full basis range

        //         //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
        //         scr_2e3c scr(m_b3);

        //         //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
        //         dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
        //         if (dev.memory * dev.nthreads > mem_total) {
        //             std::cout << " Given memory is not enough for computing integrals." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

        //         //  Step 5:
        //         //  Memory available for thread-local result arrays:
        //         size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
        //         //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
        //         size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
        //         {  //  The code block here should be replaced by estimating # of munu basis function pairs
        //             //  and/or # of auxiliary basis function.
        //             nbsp_per_subrange = nbsp;
        //             naux_per_subrange = n_aux;
        //         }
        //         //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
        //         //  of each munu shell pair.
        //         size_t min_nbsp_per_subrange = 0;
        //         #pragma omp for 
        //         for (size_t isp = 0; isp < nsp; isp++) {
        //             size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
        //             min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
        //         }
        //         if (nbsp_per_subrange < min_nbsp_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
        //         //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
        //         //  of each auxiliary shell.
        //         size_t min_naux_per_subrange = 0;
        //         for (size_t is_q = 0; is_q < ns_q; is_q++) {
        //             size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
        //             min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
        //         }
        //         if (naux_per_subrange < min_naux_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

        //         //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
        //         arma::vec Fvec(nbsp);
        //         //  Result will be accumulated in the output arrays, so we need to zero out them
        //         Fvec.zeros();
        //         JG_b.zeros(); 
        //         dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occb, gamma_Pb, JG_b);
        //         // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

        //         //  Step 7: Loop over basis subranges and run libqints job
        //         libqints::batching_info<2> binfo;
        //         libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
        //         libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
        //         for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
        //             //  Current basis subrange
        //             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
        //                 fbr, binfo.get_batch_window(biter.get_batch_number()));
        //             if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
        //                 std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
        //                 qjob.end();  //  End the libqints job before return
        //                 return;
        //             }
        //         }
        //         //  In case 2, we need to unpack F from vector form to matrix form with
        //         //  permutationally symmetric matrix elements are properly copied
        //         libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
        //         libaview::array_view<double> av_f(F.memptr(), F.n_elem);
        //         libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
        //         libaview::array_view<double> av_result(JG_b.memptr(), JG_b.n_elem);
        //     }
        // } // end (BB|BB), (AA|BB)
        //     F3_digestor_b = CvirtB.st() * F * Lam_pB;
            
        arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        // JG_b(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                        //                 * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        JG_b_local(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG_b)
        {
            JG_b += JG_b_local;
        }



        // (BB|BB), (AA|BB)
        arma::Mat<double> F3b = (iQ_b.st() * BQpv_b) + (iQ_a.st() * BQpv_b);
        arma::Mat<double> F33b(F3b.memptr(), n_virb, n_virb, false, true);
        // arma::Mat<double> F33b(F3_digestor_b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat1_b = F33b.st();
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;


        // (BB|BB), (AA|BB)
        arma::Mat<double> F4b = (iQ_b.st() * BQoh_b) + (iQ_a.st() * BQoh_b);
        arma::Mat<double> F44b(F4b.memptr(), n_occb, n_occb, false, true);
        arma::Mat<double> Foo_hat1_b = F44b.st();
        arma::Mat<double> Foo_hat2_b = BQooB.st() * BQhoB;
        arma::Mat<double> Foo_hat_b = f_oo_b + Foo_hat1_b - Foo_hat2_b;


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        E_oo_b = Foo_hat_b + (YQaiB.st() * BQovB).st(); // E_ji

        sigma_0_b += (E_vv_b*r1b) - (r1b*E_oo_b);

        vec a = vectorise(r1a);
        vec b = vectorise(r1b);
        vec c = join_cols(a,b);

        /// step 6:

        // sigma_JG
        sigma_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
       
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_a(a,i) -= Y_bar_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_a(a,i) = sigma_0_a(a,i) + sigma_JG_a(a,i) + sigma_H_a(a,i) + sigma_I_a(a,i);

                }
            }
        } // end (AA|AA)


        // sigma_JG
        sigma_JG_b += Lam_pB.st() * JG_b;
        
        // (BB|BB)
        #pragma omp parallel
        {
                 
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_b(a,i) -= Y_bar_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_b(a,i) = sigma_0_b(a,i) + sigma_JG_b(a,i) + sigma_H_b(a,i) + sigma_I_b(a,i);
                }
            }
        } // end (BB|BB)
        
    }
}


template<>
void ri_eomee_unr_r<double,double>::diis_unrestricted_energy(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a,
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQob_b, Mat<double> &BQpo_b, 
    Mat<double> &BQhb_b, Mat<double> &BQbp_b,
    Mat<double> &BQpv_a, Mat<double> &BQpv_b, 
    Mat<double> &V_Pab,  
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar, 
    Mat<double> &Lam_hB_bar, Mat<double> &Lam_pB_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a, 
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b, 
    Mat<double> &r1a, Mat<double> &r1b,  
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_a, Mat<double> &sigma_b) {


    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::vec iQ_bar_a (n_aux, fill::zeros);
    arma::mat sigma_0_a (n_vira, n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat sigma_JG_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_H_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_I_a (n_vira, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Y_bar_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::vec iQ_bar_b (n_aux, fill::zeros);
    arma::mat sigma_0_b (n_virb, n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat sigma_JG_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_H_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_I_b (n_virb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Y_bar_b (n_aux, n_virb*n_occb, fill::zeros);
    
    {   

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);
        iQ_bar_a += BQov_a * vectorise(r1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);
        iQ_bar_b += BQov_b * vectorise(r1b);


        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobA(BQob_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQobB(BQob_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);


        // Fov_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|AA), (BB|AA)
        arma::Mat<double> F2a = (iQ_bar_a.st() * BQov_a) + (iQ_bar_b.st() * BQov_a);
        arma::Mat<double> F22a(F2a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_bar1_a = F22a.st();
        arma::Mat<double> Fov_bar2_a = BQobA.st() * BQvoA;
        arma::Mat<double> Fov_bar_a = Fov_bar1_a - Fov_bar2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F2b = (iQ_bar_b.st() * BQov_b) + (iQ_bar_a.st() * BQov_b);
        arma::Mat<double> F22b(F2b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_bar1_b = F22b.st();
        arma::Mat<double> Fov_bar2_b = BQobB.st() * BQvoB;
        arma::Mat<double> Fov_bar_b = Fov_bar1_b - Fov_bar2_b;


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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: BJai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: BJai
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        double r2ab = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eA[n_occa+a] + exci);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                 

                        // sigma_I_a(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);
                        sigma_I_a_local(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: bjAI
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: bjAI
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        double r2ba = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eB[n_occb+a] + exci);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                                            
                        // sigma_I_b(a,i) += r2ab * Fov_hat_a(j,b) + t2ab * Fov_bar_a(j,b);
                        sigma_I_b_local(a,i) += r2ba * Fov_hat_a(j,b) + t2ba * Fov_bar_a(j,b);
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        double r2aa = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                        double r2aa_2 = (w1[a*n_vira+b] + w2b[a] + w3[a*n_vira+b] + w4b[a]) / (dijb - eA[n_occa+a] + exci);


                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_a(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        // sigma_I_a(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));
                        sigma_I_a_local(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        sigma_I_a_local(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end parallel (3)


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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        
                        double r2bb = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                        double r2bb_2 = (w1[a*n_virb+b] + w2b[a] + w3[a*n_virb+b] + w4b[a]) / (dijb - eB[n_occb+a] + exci);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_b(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        // sigma_I_b(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));
                        sigma_I_b_local(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        sigma_I_b_local(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)


        
/*
        //GPP: this is the nested for loop used before
        E_vv_a = Fvv_hat_a; // E_ab
        E_oo_a = Foo_hat_a; // E_ji
        #pragma omp parallel
        {
            
            arma::mat E_vv_a_local (n_vira, n_vira, fill::zeros);
            arma::mat E_oo_a_local (n_occa, n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {
                            
                            //denominator
                            double delta_AB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];
                            double t2ab = 0.0;
                            double r2ab = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                t2ab += BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                                
                                r2ab += BQhb_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]
                                        + BQhb_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)]
                                        + BQbp_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]
                                        + BQbp_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)];

                            }
                            
                            t2ab = t2ab / delta_AB;
                            r2ab = r2ab / (delta_AB + exci);
                            
                            for(size_t c = 0; c < n_vira; c++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_vv_a(a,c) -= t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vira*n_aux+c*n_aux+Q)];
                                    E_vv_a_local(a,c) -= t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vira*n_aux+c*n_aux+Q)];
                                }
                            }
                                
                            for(size_t k = 0; k < n_occa; k++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_oo_a(k,i) += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vira*n_aux+a*n_aux+Q)];
                                    E_oo_a_local(k,i) += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vira*n_aux+a*n_aux+Q)];
                                }
                            }
                            
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_a[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                                Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_a(a,i) += (r2ab * Fov_hat_b(j,b)) + (t2ab * Fov_bar_b(j,b));
                            sigma_I_a_local(a,i) += (r2ab * Fov_hat_b(j,b)) + (t2ab * Fov_bar_b(j,b));
                            
                        }
                    }
                }
            }
            #pragma omp critical (E_a)
            {
                E_vv_a += E_vv_a_local;
                E_oo_a += E_oo_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end (AA|BB)


        #pragma omp parallel
        {
            
            arma::mat E_vv_a_local (n_vira, n_vira, fill::zeros);
            arma::mat E_oo_a_local (n_occa, n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {
                            
                            //denominator
                            double delta_AA = eA(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];
                            double t2aa = 0.0;
                            double t2aa_2 = 0.0;
                            double r2aa = 0.0;
                            double r2aa_2 = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                t2aa += BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                                t2aa_2 += BQhp_a[(i*n_vira*n_aux+b*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)];
                                
                                r2aa += BQhb_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]
                                        + BQhb_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)]
                                        + BQbp_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]
                                        + BQbp_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)];
                                r2aa_2 += BQhb_a[(a*n_vira*n_aux+j*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+b*n_aux+Q)]
                                        + BQhb_a[(b*n_vira*n_aux+i*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)]
                                        + BQbp_a[(a*n_vira*n_aux+j*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+b*n_aux+Q)]
                                        + BQbp_a[(b*n_vira*n_aux+i*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)];

                            }
                            
                            t2aa = t2aa / delta_AA;
                            t2aa_2 = t2aa_2 / delta_AA;
                            r2aa = r2aa / (delta_AA + exci);
                            r2aa_2 = r2aa_2 / (delta_AA + exci);
                            
                            for(size_t c = 0; c < n_vira; c++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_vv_a(a,c) -= (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vira*n_aux+c*n_aux+Q)];
                                    E_vv_a_local(a,c) -= (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vira*n_aux+c*n_aux+Q)];
                                }
                            }
                                
                            for(size_t k = 0; k < n_occa; k++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_oo_a(k,i) += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vira*n_aux+a*n_aux+Q)];
                                    E_oo_a_local(k,i) += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vira*n_aux+a*n_aux+Q)];
                                }
                            }
                            
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_a[(a*n_occa*n_aux+i*n_aux+P)] += (r2aa - r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+P)] += (r2aa - r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_a(a,i) += ((r2aa - r2aa_2) * Fov_hat_a(j,b)) + ((t2aa - t2aa_2) * Fov_bar_a(j,b));
                            sigma_I_a_local(a,i) += ((r2aa - r2aa_2) * Fov_hat_a(j,b)) + ((t2aa - t2aa_2) * Fov_bar_a(j,b));
                            
                        }
                    }
                }
            }
            #pragma omp critical (E_a)
            {
                E_vv_a += E_vv_a_local;
                E_oo_a += E_oo_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end (AA|AA)
        

        E_vv_b = Fvv_hat_b; // E_ab
        E_oo_b = Foo_hat_b; // E_ji

        #pragma omp parallel
        {
            
            arma::mat E_vv_b_local (n_virb, n_virb, fill::zeros);
            arma::mat E_oo_b_local (n_occb, n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {
                            
                            //denominator
                            double delta_BA = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];
                            double t2ba = 0.0;
                            double r2ba = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                t2ba += BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                                
                                r2ba += BQhb_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]
                                        + BQhb_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)]
                                        + BQbp_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]
                                        + BQbp_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)];

                            }
                            
                            t2ba = t2ba / delta_BA;
                            r2ba = r2ba / (delta_BA + exci);

                            for(size_t c = 0; c < n_virb; c++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_vv_b(a,c) -= t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_b[(i*n_virb*n_aux+c*n_aux+Q)];
                                    E_vv_b_local(a,c) -= t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_b[(i*n_virb*n_aux+c*n_aux+Q)];
                                }
                            }
                                
                            for(size_t k = 0; k < n_occb; k++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_oo_b(k,i) += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_b[(k*n_virb*n_aux+a*n_aux+Q)];
                                    E_oo_b_local(k,i) += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_b[(k*n_virb*n_aux+a*n_aux+Q)];
                                }
                            }
                            
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_b[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_b(a,i) += (r2ba * Fov_hat_a(j,b)) + (t2ba * Fov_bar_a(j,b));
                            sigma_I_b_local(a,i) += (r2ba * Fov_hat_a(j,b)) + (t2ba * Fov_bar_a(j,b));
                            
                        }
                    }
                }
            }
            #pragma omp critical (E_b)
            {
                E_vv_b += E_vv_b_local;
                E_oo_b += E_oo_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|AA)


        #pragma omp parallel
        {
            
            arma::mat E_vv_b_local (n_virb, n_virb, fill::zeros);
            arma::mat E_oo_b_local (n_occb, n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {
                            
                            //denominator
                            double delta_BB = eB(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];
                            double t2bb = 0.0;
                            double t2bb_2 = 0.0;
                            double r2bb = 0.0;
                            double r2bb_2 = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                t2bb += BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                                t2bb_2 += BQhp_b[(i*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)];
                                
                                r2bb += BQhb_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]
                                        + BQhb_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)]
                                        + BQbp_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]
                                        + BQbp_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)];
                                r2bb_2 += BQhb_b[(j*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+b*n_aux+Q)]
                                        + BQhb_b[(i*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)]
                                        + BQbp_b[(j*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+b*n_aux+Q)]
                                        + BQbp_b[(i*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)];

                            }
                            
                            t2bb = t2bb / delta_BB;
                            t2bb_2 = t2bb_2 / delta_BB;
                            r2bb = r2bb / (delta_BB + exci);
                            r2bb_2 = r2bb_2 / (delta_BB + exci);
                            
                            // sigma_G - (jb|ca) ovvv permute 2,4
                            for(size_t c = 0; c < n_virb; c++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_vv_b(a,c) -= (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(i*n_virb*n_aux+c*n_aux+Q)];
                                    E_vv_b_local(a,c) -= (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(i*n_virb*n_aux+c*n_aux+Q)];
                                }
                            }
                                
                            // sigma_H - (jb|ik) ovoo permute 1,3
                            for(size_t k = 0; k < n_occb; k++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_oo_b(k,i) += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(k*n_virb*n_aux+a*n_aux+Q)];
                                    E_oo_b_local(k,i) += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(k*n_virb*n_aux+a*n_aux+Q)];
                                }
                            }
                            
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_b[(a*n_occb*n_aux+i*n_aux+P)] += (r2bb - r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                                Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+P)] += (r2bb - r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_b(a,i) += ((r2bb - r2bb_2) * Fov_hat_b(j,b)) + ((t2bb - t2bb_2) * Fov_bar_b(j,b)); 
                            sigma_I_b_local(a,i) += ((r2bb - r2bb_2) * Fov_hat_b(j,b)) + ((t2bb - t2bb_2) * Fov_bar_b(j,b)); 
                            
                        }
                    }
                }
            }
            #pragma omp critical (E_b)
            {
                E_vv_b += E_vv_b_local;
                E_oo_b += E_oo_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)
*/


        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (AA|AA), (BB|AA)
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_a(Y_bar_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<double> gamma_G1a = YQia_bar_a * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11a = (iQ_bar_a * vectorise(Lam_hA).st()) + (iQ_bar_b * vectorise(Lam_hA).st());
            arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux*n_occa, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga + gamma_J1a - gamma_J2a;

            arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;



            // // #pragma omp parallel
            // arma::Mat<double> F3_digestor_a (n_vira, n_vira, fill::zeros);
            // {
            // digestor
            // arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
            // {
            // arma::vec iP (n_aux, fill::zeros);
            // iP = (PQinvhalf * iQ_a) + (PQinvhalf * iQ_b);

            //     //  Step 1: Read libqints-type basis set from files and form shellpair basis.
            //     // libqints::basis_1e2c_shellpair_cgto<double> bsp;
            //     // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
            //     const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
            //     const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
            //     size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
            //     size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
            //     size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
            //     //  Construct the 2e3c shellpair basis and corresponding full basis range
            //     libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
            //     libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
            //     libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

            //     //  Step 2: prepare required input settings
            //     libqints::dev_omp dev;                  //  libqints-type device information.
            //     size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
            //     dev.init(1024);
            //     dev.nthreads = 1;
            //     dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
            //     libqints::deriv_code dc;
            //     dc.set(0);                //  Set integral derivative level
            //     libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
            //     libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
            //     qjob.begin(fbr);                                //  Start the libqints job for full basis range

            //     //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
            //     scr_2e3c scr(m_b3);

            //     //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
            //     dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
            //     if (dev.memory * dev.nthreads > mem_total) {
            //         std::cout << " Given memory is not enough for computing integrals." << std::endl;
            //         qjob.end();  //  End the libqints job before return
            //         return;
            //     }
            //     size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

            //     //  Step 5:
            //     //  Memory available for thread-local result arrays:
            //     size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
            //     //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
            //     size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
            //     {  //  The code block here should be replaced by estimating # of munu basis function pairs
            //         //  and/or # of auxiliary basis function.
            //         nbsp_per_subrange = nbsp;
            //         naux_per_subrange = n_aux;
            //     }
            //     //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
            //     //  of each munu shell pair.
            //     size_t min_nbsp_per_subrange = 0;
            //     #pragma omp for 
            //     for (size_t isp = 0; isp < nsp; isp++) {
            //         size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
            //         min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
            //     }
            //     if (nbsp_per_subrange < min_nbsp_per_subrange) {
            //         std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
            //         qjob.end();  //  End the libqints job before return
            //         return;
            //     }
            //     nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
            //     //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
            //     //  of each auxiliary shell.
            //     size_t min_naux_per_subrange = 0;
            //     for (size_t is_q = 0; is_q < ns_q; is_q++) {
            //         size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
            //         min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
            //     }
            //     if (naux_per_subrange < min_naux_per_subrange) {
            //         std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
            //         qjob.end();  //  End the libqints job before return
            //         return;
            //     }
            //     naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

            //     //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
            //     arma::vec Fvec(nbsp);
            //     //  Result will be accumulated in the output arrays, so we need to zero out them
            //     Fvec.zeros();
            //     JG_a.zeros(); 
            //     dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occa, gamma_Pa, JG_a);
            //     // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

            //     //  Step 7: Loop over basis subranges and run libqints job
            //     libqints::batching_info<2> binfo;
            //     libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
            //     libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
            //     for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
            //         //  Current basis subrange
            //         libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
            //             fbr, binfo.get_batch_window(biter.get_batch_number()));
            //         if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
            //             std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
            //             qjob.end();  //  End the libqints job before return
            //             return;
            //         }
            //     }
            //     //  In case 2, we need to unpack F from vector form to matrix form with
            //     //  permutationally symmetric matrix elements are properly copied
            //     libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
            //     libaview::array_view<double> av_f(F.memptr(), F.n_elem);
            //     libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
            //     libaview::array_view<double> av_result(JG_a.memptr(), JG_a.n_elem);

            // }
            // } // end (AA|AA), (BB|AA)
            // F3_digestor_a = CvirtA.st() * F * Lam_pA;
            
        arma::mat JG_a_local (n_orb, n_occa, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        // JG_a(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                        //                 * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        JG_a_local(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG_a)
        {
            JG_a += JG_a_local;
        }


        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia_bar_b(Y_bar_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> gamma_G1b = YQia_bar_b * CvirtB.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occb; i++) {
            gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J1: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J11b = (iQ_bar_b * vectorise(Lam_hB).st()) + (iQ_bar_a * vectorise(Lam_hB).st());
        arma::Mat<double> gamma_J1b(gamma_J11b.memptr(), n_aux*n_occb, n_orb, false, true);

        // / omega_J2: third term of Γ(P,iβ)
        arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occb; i++) {
            gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Qb = gamma_Gb + gamma_J1b - gamma_J2b;

        arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
        gamma_Pb = PQinvhalf * gamma_Qb;

        // (BB|BB), (AA|BB)
        // #pragma omp parallel
        // arma::Mat<double> F3_digestor_b (n_virb, n_virb, fill::zeros);
        // {
        //     arma::vec iP (n_aux, fill::zeros);
        //     // iP = PQinvhalf * iQ_b;
        //     iP = (PQinvhalf * iQ_b) + (PQinvhalf * iQ_a);

        //     // digestor
        //     arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
        //     // arma::Mat<double> JG (n_orb, n_occb, fill::zeros);
        //     {

        //         //  Step 1: Read libqints-type basis set from files and form shellpair basis.
        //         // libqints::basis_1e2c_shellpair_cgto<double> bsp;
        //         // libqints::basis_1e1c_cgto<double> \;  //  1e1c auxiliary basis
        //         const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
        //         const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
        //         size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
        //         size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
        //         size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
        //         //  Construct the 2e3c shellpair basis and corresponding full basis range
        //         libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
        //         libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

        //         //  Step 2: prepare required input settings
        //         libqints::dev_omp dev;                  //  libqints-type device information.
        //         size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
        //         dev.init(1024);
        //         dev.nthreads = 1;
        //         dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
        //         libqints::deriv_code dc;
        //         dc.set(0);                //  Set integral derivative level
        //         libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
        //         libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
        //         qjob.begin(fbr);                                //  Start the libqints job for full basis range

        //         //  Step 3: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
        //         scr_2e3c scr(m_b3);

        //         //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
        //         dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
        //         if (dev.memory * dev.nthreads > mem_total) {
        //             std::cout << " Given memory is not enough for computing integrals." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity

        //         //  Step 5:
        //         //  Memory available for thread-local result arrays:
        //         size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
        //         //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
        //         size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
        //         {  //  The code block here should be replaced by estimating # of munu basis function pairs
        //             //  and/or # of auxiliary basis function.
        //             nbsp_per_subrange = nbsp;
        //             naux_per_subrange = n_aux;
        //         }
        //         //  Get the minimum # of munu basis function pairs per subrange, which is the maximum # of munu basis function pars
        //         //  of each munu shell pair.
        //         size_t min_nbsp_per_subrange = 0;
        //         #pragma omp for 
        //         for (size_t isp = 0; isp < nsp; isp++) {
        //             size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
        //             min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
        //         }
        //         if (nbsp_per_subrange < min_nbsp_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity
        //         //  Get the minimum # of auxiliary basis functions per subrange, which is the maximum # of auxiliary basis functions
        //         //  of each auxiliary shell.
        //         size_t min_naux_per_subrange = 0;
        //         for (size_t is_q = 0; is_q < ns_q; is_q++) {
        //             size_t naux_is = b1x[is_q].get_num_comp();  //  # of basis functions of this shell
        //             min_naux_per_subrange = std::max(naux_is, min_naux_per_subrange);
        //         }
        //         if (naux_per_subrange < min_naux_per_subrange) {
        //             std::cout << " Given memory is not enough for holding thread-local result arrays." << std::endl;
        //             qjob.end();  //  End the libqints job before return
        //             return;
        //         }
        //         naux_per_subrange = min_naux_per_subrange;  //  Use minimum subrange for simplicity

        //         //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
        //         arma::vec Fvec(nbsp);
        //         //  Result will be accumulated in the output arrays, so we need to zero out them
        //         Fvec.zeros();
        //         JG_b.zeros(); 
        //         dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occb, gamma_Pb, JG_b);
        //         // dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

        //         //  Step 7: Loop over basis subranges and run libqints job
        //         libqints::batching_info<2> binfo;
        //         libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
        //         libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
        //         for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
        //             //  Current basis subrange
        //             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
        //                 fbr, binfo.get_batch_window(biter.get_batch_number()));
        //             if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
        //                 std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
        //                 qjob.end();  //  End the libqints job before return
        //                 return;
        //             }
        //         }
        //         //  In case 2, we need to unpack F from vector form to matrix form with
        //         //  permutationally symmetric matrix elements are properly copied
        //         libaview::array_view<double> av_fvec(Fvec.memptr(), Fvec.n_elem);
        //         libaview::array_view<double> av_f(F.memptr(), F.n_elem);
        //         libqints::gto::unpack(bsp, av_fvec, n_orb, n_orb, av_f);
        //         libaview::array_view<double> av_result(JG_b.memptr(), JG_b.n_elem);
        //     }
        // } // end (BB|BB), (AA|BB)
        //     F3_digestor_b = CvirtB.st() * F * Lam_pB;
            

        arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        // JG_b(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                        //                 * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        JG_b_local(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        
                    }
                }
            }
        }
        #pragma omp critical (JG_b)
        {
            JG_b += JG_b_local;
        }


        // Fvv_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F3a = (iQ_a.st() * BQpv_a) + (iQ_b.st() * BQpv_a);
        arma::Mat<double> F33a(F3a.memptr(), n_vira, n_vira, false, true);
        // arma::Mat<double> F33a(F3_digestor_a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat1_a = F33a.st();
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F3b = (iQ_b.st() * BQpv_b) + (iQ_a.st() * BQpv_b);
        arma::Mat<double> F33b(F3b.memptr(), n_virb, n_virb, false, true);
        // arma::Mat<double> F33b(F3_digestor_b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat1_b = F33b.st();
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;

        // Foo_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F4a = (iQ_a.st() * BQoh_a) + (iQ_b.st() * BQoh_a);
        arma::Mat<double> F44a(F4a.memptr(), n_occa, n_occa, false, true);
        arma::Mat<double> Foo_hat1_a = F44a.st();
        arma::Mat<double> Foo_hat2_a = BQooA.st() * BQhoA;
        arma::Mat<double> Foo_hat_a = f_oo_a + Foo_hat1_a - Foo_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F4b = (iQ_b.st() * BQoh_b) + (iQ_a.st() * BQoh_b);
        arma::Mat<double> F44b(F4b.memptr(), n_occb, n_occb, false, true);
        arma::Mat<double> Foo_hat1_b = F44b.st();
        arma::Mat<double> Foo_hat2_b = BQooB.st() * BQhoB;
        arma::Mat<double> Foo_hat_b = f_oo_b + Foo_hat1_b - Foo_hat2_b;


        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);

        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab
        E_oo_a = Foo_hat_a + (YQaiA.st() * BQovA).st(); // E_ji

        sigma_0_a += (E_vv_a*r1a) - (r1a*E_oo_a);


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        E_oo_b = Foo_hat_b + (YQaiB.st() * BQovB).st(); // E_ji

        sigma_0_b += (E_vv_b*r1b) - (r1b*E_oo_b);

        vec a = vectorise(r1a);
        vec b = vectorise(r1b);
        vec c = join_cols(a,b);

        /// step 6:

        // sigma_JG
        sigma_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
       
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_a(a,i) -= Y_bar_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_a(a,i) = sigma_0_a(a,i) + sigma_JG_a(a,i) + sigma_H_a(a,i) + sigma_I_a(a,i);

                }
            }
        } // end (AA|AA)


        // sigma_JG
        sigma_JG_b += Lam_pB.st() * JG_b;
        
        // (BB|BB)
        #pragma omp parallel
        {
                 
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_b(a,i) -= Y_bar_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_b(a,i) = sigma_0_b(a,i) + sigma_JG_b(a,i) + sigma_H_b(a,i) + sigma_I_b(a,i);

                }
            }
        } // end (BB|BB)

        exci = (accu(sigma_a % r1a) + accu(sigma_b % r1b)) / pow(norm(c,"fro"),2);

        // (AA|AA)
        #pragma omp parallel
        {
            // update of the trial vector
            arma::mat res_a (n_vira, n_occa, fill::zeros);
            arma::Mat<double> update_a (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                        
                    double delta_A = eA(i) - eA[n_occa+a];
                    res_a(a,i) = (sigma_a(a,i) - (exci*r1a(a,i))) / norm(c,"fro");
                    update_a(a,i) = res_a(a,i) / delta_A;
                    r1a(a,i) = (r1a(a,i) + update_a(a,i)) / norm(c,"fro");
                        
                }
            }
        } // end (AA|AA)

        // (BB|BB)
        #pragma omp parallel
        {
            // update of the trial vector
            arma::mat res_b (n_virb, n_occb, fill::zeros);
            arma::mat update_b (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                        
                    double delta_B = eB(i) - eB[n_occb+a];
                    res_b(a,i) = (sigma_b(a,i) - (exci*r1b(a,i))) / norm(c,"fro");
                    update_b(a,i) = res_b(a,i) / delta_B;
                    r1b(a,i) = (r1b(a,i) + update_b(a,i)) / norm(c,"fro");
                        
                }
            }
        } // end (BB|BB)

    }
}


template<>
void ri_eomee_unr_r<double,double>::ccs_unrestricted_energy_digestor(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a,
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQob_b, Mat<double> &BQpo_b, 
    Mat<double> &BQhb_b, Mat<double> &BQbp_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar, 
    Mat<double> &Lam_hB_bar, Mat<double> &Lam_pB_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a, 
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b, 
    Mat<double> &r1a, Mat<double> &r1b,  
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_a, Mat<double> &sigma_b) {


    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::vec iQ_bar_a (n_aux, fill::zeros);
    arma::mat sigma_0_a (n_vira, n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat sigma_JG_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_H_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_I_a (n_vira, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Y_bar_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::vec iQ_bar_b (n_aux, fill::zeros);
    arma::mat sigma_0_b (n_virb, n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat sigma_JG_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_H_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_I_b (n_virb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Y_bar_b (n_aux, n_virb*n_occb, fill::zeros);
    
    {   
         exci = 0; 
         double t2ab = 0.0, t2ba = 0.0, t2aa = 0.0, t2bb = 0.0, t2aa_2 = 0.0, t2bb_2 = 0.0;
         double r2ab = 0.0, r2ba = 0.0, r2aa = 0.0, r2bb = 0.0, r2aa_2 = 0.0, r2bb_2 = 0.0;

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);
        iQ_bar_a += BQov_a * vectorise(r1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);
        iQ_bar_b += BQov_b * vectorise(r1b);


        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobA(BQob_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQobB(BQob_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);


        // Fov_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|AA), (BB|AA)
        arma::Mat<double> F2a = (iQ_bar_a.st() * BQov_a) + (iQ_bar_b.st() * BQov_a);
        arma::Mat<double> F22a(F2a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_bar1_a = F22a.st();
        arma::Mat<double> Fov_bar2_a = BQobA.st() * BQvoA;
        arma::Mat<double> Fov_bar_a = Fov_bar1_a - Fov_bar2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F2b = (iQ_bar_b.st() * BQov_b) + (iQ_bar_a.st() * BQov_b);
        arma::Mat<double> F22b(F2b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_bar1_b = F22b.st();
        arma::Mat<double> Fov_bar2_b = BQobB.st() * BQvoB;
        arma::Mat<double> Fov_bar_b = Fov_bar1_b - Fov_bar2_b;


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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: BJai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: BJai
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                 

                        // sigma_I_a(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);
                        sigma_I_a_local(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: bjAI
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: bjAI
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {

                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                                            
                        // sigma_I_b(a,i) += r2ab * Fov_hat_a(j,b) + t2ab * Fov_bar_a(j,b);
                        sigma_I_b_local(a,i) += r2ba * Fov_hat_a(j,b) + t2ba * Fov_bar_a(j,b);
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_a(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        // sigma_I_a(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));
                        sigma_I_a_local(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        sigma_I_a_local(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end parallel (3)


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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_b(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        // sigma_I_b(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));
                        sigma_I_b_local(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        sigma_I_b_local(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)


        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (AA|AA), (BB|AA)
        // #pragma omp parallel
        arma::Mat<double> F3_digestor_a (n_vira, n_vira, fill::zeros);
        {
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_a(Y_bar_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<double> gamma_G1a = YQia_bar_a * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11a = (iQ_bar_a * vectorise(Lam_hA).st()) + (iQ_bar_b * vectorise(Lam_hA).st());
            arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux*n_occa, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga + gamma_J1a - gamma_J2a;

            arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;

            arma::vec iP (n_aux, fill::zeros);
            iP = (PQinvhalf * iQ_a) + (PQinvhalf * iQ_b);

            // digestor
            arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
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
                JG_a.zeros(); 
                dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occa, gamma_Pa, JG_a);
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
                libaview::array_view<double> av_result(JG_a.memptr(), JG_a.n_elem);

            }

            F3_digestor_a = CvirtA.st() * F * Lam_pA;


        } // end (AA|AA), (BB|AA)


        // Fvv_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F33a(F3_digestor_a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat1_a = F33a.st();
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;

        // Foo_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F4a = (iQ_a.st() * BQoh_a) + (iQ_b.st() * BQoh_a);
        arma::Mat<double> F44a(F4a.memptr(), n_occa, n_occa, false, true);
        arma::Mat<double> Foo_hat1_a = F44a.st();
        arma::Mat<double> Foo_hat2_a = BQooA.st() * BQhoA;
        arma::Mat<double> Foo_hat_a = f_oo_a + Foo_hat1_a - Foo_hat2_a;


        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);

        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab
        E_oo_a = Foo_hat_a + (YQaiA.st() * BQovA).st(); // E_ji

        sigma_0_a += (E_vv_a*r1a) - (r1a*E_oo_a);


        // (BB|BB), (AA|BB)
        // #pragma omp parallel
        arma::Mat<double> F3_digestor_b (n_virb, n_virb, fill::zeros);
        {
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_b(Y_bar_b.memptr(), n_aux*n_occb, n_virb, false, true);
            arma::Mat<double> gamma_G1b = YQia_bar_b * CvirtB.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11b = (iQ_bar_b * vectorise(Lam_hB).st()) + (iQ_bar_a * vectorise(Lam_hB).st());
            arma::Mat<double> gamma_J1b(gamma_J11b.memptr(), n_aux*n_occb, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_Gb + gamma_J1b - gamma_J2b;

            arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
            gamma_Pb = PQinvhalf * gamma_Qb;

            arma::vec iP (n_aux, fill::zeros);
            iP = (PQinvhalf * iQ_b) + (PQinvhalf * iQ_a);

            // digestor
            arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
            // arma::Mat<double> JG (n_orb, n_occb, fill::zeros);
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
                JG_b.zeros(); 
                dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occb, gamma_Pb, JG_b);
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
                libaview::array_view<double> av_result(JG_b.memptr(), JG_b.n_elem);

            }

            F3_digestor_b = CvirtB.st() * F * Lam_pB;

        } // end (BB|BB), (AA|BB)

        // Fvv_hat
        // (BB|BB), (AA|BB)
        arma::Mat<double> F33b(F3_digestor_b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat1_b = F33b.st();
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;

        // Foo_hat
        // (BB|BB), (AA|BB)
        arma::Mat<double> F4b = (iQ_b.st() * BQoh_b) + (iQ_a.st() * BQoh_b);
        arma::Mat<double> F44b(F4b.memptr(), n_occb, n_occb, false, true);
        arma::Mat<double> Foo_hat1_b = F44b.st();
        arma::Mat<double> Foo_hat2_b = BQooB.st() * BQhoB;
        arma::Mat<double> Foo_hat_b = f_oo_b + Foo_hat1_b - Foo_hat2_b;

        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        E_oo_b = Foo_hat_b + (YQaiB.st() * BQovB).st(); // E_ji

        sigma_0_b += (E_vv_b*r1b) - (r1b*E_oo_b);

        vec a = vectorise(r1a);
        vec b = vectorise(r1b);
        vec c = join_cols(a,b);

        /// step 6:

        // sigma_JG
        sigma_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
       
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_a(a,i) -= Y_bar_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_a(a,i) = sigma_0_a(a,i) + sigma_JG_a(a,i) + sigma_H_a(a,i) + sigma_I_a(a,i);

                }
            }
        } // end (AA|AA)
        

        // sigma_JG
        sigma_JG_b += Lam_pB.st() * JG_b;

        
        // (BB|BB)
        #pragma omp parallel
        {
                 
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_b(a,i) -= Y_bar_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_b(a,i) = sigma_0_b(a,i) + sigma_JG_b(a,i) + sigma_H_b(a,i) + sigma_I_b(a,i);
                    
                }
            }
        } // end (BB|BB)

    }
}


template<>
void ri_eomee_unr_r<double,double>::davidson_unrestricted_energy_digestor(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a,
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQob_b, Mat<double> &BQpo_b, 
    Mat<double> &BQhb_b, Mat<double> &BQbp_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar, 
    Mat<double> &Lam_hB_bar, Mat<double> &Lam_pB_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a, 
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b, 
    Mat<double> &r1a, Mat<double> &r1b,  
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_a, Mat<double> &sigma_b) {


    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::vec iQ_bar_a (n_aux, fill::zeros);
    arma::mat sigma_0_a (n_vira, n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat sigma_JG_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_H_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_I_a (n_vira, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Y_bar_a (n_aux, n_vira*n_occa, fill::zeros);
    //arma::mat sigma_a (n_vira, n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::vec iQ_bar_b (n_aux, fill::zeros);
    arma::mat sigma_0_b (n_virb, n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat sigma_JG_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_H_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_I_b (n_virb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Y_bar_b (n_aux, n_virb*n_occb, fill::zeros);
    //arma::mat sigma_b (n_virb, n_occb, fill::zeros);
    
    {   

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);
        iQ_bar_a += BQov_a * vectorise(r1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);
        iQ_bar_b += BQov_b * vectorise(r1b);


        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobA(BQob_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQobB(BQob_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);


        // Fov_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|AA), (BB|AA)
        arma::Mat<double> F2a = (iQ_bar_a.st() * BQov_a) + (iQ_bar_b.st() * BQov_a);
        arma::Mat<double> F22a(F2a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_bar1_a = F22a.st();
        arma::Mat<double> Fov_bar2_a = BQobA.st() * BQvoA;
        arma::Mat<double> Fov_bar_a = Fov_bar1_a - Fov_bar2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F2b = (iQ_bar_b.st() * BQov_b) + (iQ_bar_a.st() * BQov_b);
        arma::Mat<double> F22b(F2b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_bar1_b = F22b.st();
        arma::Mat<double> Fov_bar2_b = BQobB.st() * BQvoB;
        arma::Mat<double> Fov_bar_b = Fov_bar1_b - Fov_bar2_b;


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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: BJai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: BJai
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        double r2ab = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eA[n_occa+a] + exci);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                 

                        // sigma_I_a(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);
                        sigma_I_a_local(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: bjAI
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: bjAI
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        double r2ba = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eB[n_occb+a] + exci);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                                            
                        // sigma_I_b(a,i) += r2ab * Fov_hat_a(j,b) + t2ab * Fov_bar_a(j,b);
                        sigma_I_b_local(a,i) += r2ba * Fov_hat_a(j,b) + t2ba * Fov_bar_a(j,b);
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        double r2aa = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                        double r2aa_2 = (w1[a*n_vira+b] + w2b[a] + w3[a*n_vira+b] + w4b[a]) / (dijb - eA[n_occa+a] + exci);


                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_a(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        // sigma_I_a(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));
                        sigma_I_a_local(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        sigma_I_a_local(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end parallel (3)


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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        
                        double r2bb = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                        double r2bb_2 = (w1[a*n_virb+b] + w2b[a] + w3[a*n_virb+b] + w4b[a]) / (dijb - eB[n_occb+a] + exci);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_b(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        // sigma_I_b(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));
                        sigma_I_b_local(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        sigma_I_b_local(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)

        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (AA|AA), (BB|AA)
        // #pragma omp parallel
        arma::Mat<double> F3_digestor_a (n_vira, n_vira, fill::zeros);
        {
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_a(Y_bar_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<double> gamma_G1a = YQia_bar_a * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11a = (iQ_bar_a * vectorise(Lam_hA).st()) + (iQ_bar_b * vectorise(Lam_hA).st());
            arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux*n_occa, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga + gamma_J1a - gamma_J2a;

            arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;

            arma::vec iP (n_aux, fill::zeros);
            iP = (PQinvhalf * iQ_a) + (PQinvhalf * iQ_b);

            // digestor
            arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
            // arma::Mat<double> JG (n_orb, n_occa, fill::zeros);
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
                JG_a.zeros(); 
                dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occa, gamma_Pa, JG_a);
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
                libaview::array_view<double> av_result(JG_a.memptr(), JG_a.n_elem);

            }

            F3_digestor_a = CvirtA.st() * F * Lam_pA;
            
        } // end (AA|AA), (BB|AA)

        // Fvv_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F33a(F3_digestor_a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat1_a = F33a.st();
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;


        // Foo_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F4a = (iQ_a.st() * BQoh_a) + (iQ_b.st() * BQoh_a);
        arma::Mat<double> F44a(F4a.memptr(), n_occa, n_occa, false, true);
        arma::Mat<double> Foo_hat1_a = F44a.st();
        arma::Mat<double> Foo_hat2_a = BQooA.st() * BQhoA;
        arma::Mat<double> Foo_hat_a = f_oo_a + Foo_hat1_a - Foo_hat2_a;


        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);

        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab
        E_oo_a = Foo_hat_a + (YQaiA.st() * BQovA).st(); // E_ji

        sigma_0_a += (E_vv_a*r1a) - (r1a*E_oo_a);


        // (BB|BB), (AA|BB)
        // #pragma omp parallel
        arma::Mat<double> F3_digestor_b (n_virb, n_virb, fill::zeros);
        {
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_b(Y_bar_b.memptr(), n_aux*n_occb, n_virb, false, true);
            arma::Mat<double> gamma_G1b = YQia_bar_b * CvirtB.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11b = (iQ_bar_b * vectorise(Lam_hB).st()) + (iQ_bar_a * vectorise(Lam_hB).st());
            arma::Mat<double> gamma_J1b(gamma_J11b.memptr(), n_aux*n_occb, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_Gb + gamma_J1b - gamma_J2b;

            arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
            gamma_Pb = PQinvhalf * gamma_Qb;

            arma::vec iP (n_aux, fill::zeros);
            iP = (PQinvhalf * iQ_b) + (PQinvhalf * iQ_a);

            // digestor
            arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
            // arma::Mat<double> JG (n_orb, n_occb, fill::zeros);
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
                JG_b.zeros(); 
                dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occb, gamma_Pb, JG_b);
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
                libaview::array_view<double> av_result(JG_b.memptr(), JG_b.n_elem);

            }

            F3_digestor_b = CvirtB.st() * F * Lam_pB;
            
        } // end (BB|BB), (AA|BB)


        // (BB|BB), (AA|BB)
        arma::Mat<double> F33b(F3_digestor_b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat1_b = F33b.st();
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;


        // (BB|BB), (AA|BB)
        arma::Mat<double> F4b = (iQ_b.st() * BQoh_b) + (iQ_a.st() * BQoh_b);
        arma::Mat<double> F44b(F4b.memptr(), n_occb, n_occb, false, true);
        arma::Mat<double> Foo_hat1_b = F44b.st();
        arma::Mat<double> Foo_hat2_b = BQooB.st() * BQhoB;
        arma::Mat<double> Foo_hat_b = f_oo_b + Foo_hat1_b - Foo_hat2_b;


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        E_oo_b = Foo_hat_b + (YQaiB.st() * BQovB).st(); // E_ji

        sigma_0_b += (E_vv_b*r1b) - (r1b*E_oo_b);

        vec a = vectorise(r1a);
        vec b = vectorise(r1b);
        vec c = join_cols(a,b);

        /// step 6:

        // sigma_JG
        sigma_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
       
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_a(a,i) -= Y_bar_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_a(a,i) = sigma_0_a(a,i) + sigma_JG_a(a,i) + sigma_H_a(a,i) + sigma_I_a(a,i);

                }
            }
        } // end (AA|AA)


        // sigma_JG
        sigma_JG_b += Lam_pB.st() * JG_b;
        
        // (BB|BB)
        #pragma omp parallel
        {
                 
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_b(a,i) -= Y_bar_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_b(a,i) = sigma_0_b(a,i) + sigma_JG_b(a,i) + sigma_H_b(a,i) + sigma_I_b(a,i);
                    // sigma_b(a,i) = sigma_0_b(a,i);
                }
            }
        } // end (BB|BB)

    }
}


template<>
void ri_eomee_unr_r<double,double>::diis_unrestricted_energy_digestor(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoh_a, 
    Mat<double> &BQho_a, Mat<double> &BQoo_a, 
    Mat<double> &BQob_a, Mat<double> &BQpo_a, 
    Mat<double> &BQhb_a, Mat<double> &BQbp_a,
    Mat<double> &BQov_b, Mat<double> &BQvo_b, 
    Mat<double> &BQhp_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQob_b, Mat<double> &BQpo_b, 
    Mat<double> &BQhb_b, Mat<double> &BQbp_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &Lam_hA_bar, Mat<double> &Lam_pA_bar, 
    Mat<double> &Lam_hB_bar, Mat<double> &Lam_pB_bar,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &f_vv_a, Mat<double> &f_oo_a, 
    Mat<double> &f_vv_b, Mat<double> &f_oo_b,
    Mat<double> &t1a, Mat<double> &t1b, 
    Mat<double> &r1a, Mat<double> &r1b,  
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_a, Mat<double> &sigma_b) {


    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::vec iQ_bar_a (n_aux, fill::zeros);
    arma::mat sigma_0_a (n_vira, n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat sigma_JG_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_H_a (n_vira, n_occa, fill::zeros);
    arma::mat sigma_I_a (n_vira, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Y_bar_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::vec iQ_bar_b (n_aux, fill::zeros);
    arma::mat sigma_0_b (n_virb, n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat sigma_JG_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_H_b (n_virb, n_occb, fill::zeros);
    arma::mat sigma_I_b (n_virb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Y_bar_b (n_aux, n_virb*n_occb, fill::zeros);
    
    {   

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);
        iQ_bar_a += BQov_a * vectorise(r1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);
        iQ_bar_b += BQov_b * vectorise(r1b);


        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);
        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobA(BQob_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQobB(BQob_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);


        // Fov_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|AA), (BB|AA)
        arma::Mat<double> F2a = (iQ_bar_a.st() * BQov_a) + (iQ_bar_b.st() * BQov_a);
        arma::Mat<double> F22a(F2a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_bar1_a = F22a.st();
        arma::Mat<double> Fov_bar2_a = BQobA.st() * BQvoA;
        arma::Mat<double> Fov_bar_a = Fov_bar1_a - Fov_bar2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F2b = (iQ_bar_b.st() * BQov_b) + (iQ_bar_a.st() * BQov_b);
        arma::Mat<double> F22b(F2b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_bar1_b = F22b.st();
        arma::Mat<double> Fov_bar2_b = BQobB.st() * BQvoB;
        arma::Mat<double> Fov_bar_b = Fov_bar1_b - Fov_bar2_b;


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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: BJai
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: aiBJ
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: BJai
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        double r2ab = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eA[n_occa+a] + exci);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }
                 

                        // sigma_I_a(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);
                        sigma_I_a_local(a,i) += r2ab * Fov_hat_b(j,b) + t2ab * Fov_bar_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2: AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2: bjAI
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2: AIbj
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2: bjAI
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w0b[a] / (dijb - eB[n_occb+a]);
                        double r2ba = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eB[n_occb+a] + exci);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }
                                            
                        // sigma_I_b(a,i) += r2ab * Fov_hat_a(j,b) + t2ab * Fov_bar_a(j,b);
                        sigma_I_b_local(a,i) += r2ba * Fov_hat_a(j,b) + t2ba * Fov_bar_a(j,b);
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
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
            arma::mat Yai_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat Y_bar_a_local (n_aux, n_vira*n_occa, fill::zeros);
            arma::mat sigma_I_a_local (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        double r2aa = (w1b[a] + w2[a*n_vira+b] + w3b[a] + w4[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                        double r2aa_2 = (w1[a*n_vira+b] + w2b[a] + w3[a*n_vira+b] + w4b[a]) / (dijb - eA[n_occa+a] + exci);


                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Y_bar_a[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Y_bar_a[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Y_bar_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Y_bar_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (r2aa-r2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_a(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        // sigma_I_a(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));
                        sigma_I_a_local(a,i) += ((r2aa-r2aa_2) * Fov_hat_a(j,b)) + ((t2aa-t2aa_2) * Fov_bar_a(j,b));
                        sigma_I_a_local(b,j) += ((r2aa-r2aa_2) * Fov_hat_a(i,a)) + ((t2aa-t2aa_2) * Fov_bar_a(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_a += Y_bar_a_local;
                sigma_I_a += sigma_I_a_local;
            }
        } // end parallel (3)



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
            arma::mat Yai_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat Y_bar_b_local (n_aux, n_virb*n_occb, fill::zeros);
            arma::mat sigma_I_b_local (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        
                        double r2bb = (w1b[a] + w2[a*n_virb+b] + w3b[a] + w4[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                        double r2bb_2 = (w1[a*n_virb+b] + w2b[a] + w3[a*n_virb+b] + w4b[a]) / (dijb - eB[n_occb+a] + exci);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Y_bar_b[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Y_bar_b[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Y_bar_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Y_bar_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (r2bb-r2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }

                        // sigma_I_b(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        // sigma_I_b(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));
                        sigma_I_b_local(a,i) += ((r2bb-r2bb_2) * Fov_hat_b(j,b)) + ((t2bb-t2bb_2) * Fov_bar_b(j,b));
                        sigma_I_b_local(b,j) += ((r2bb-r2bb_2) * Fov_hat_b(i,a)) + ((t2bb-t2bb_2) * Fov_bar_b(i,a));

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_b += Y_bar_b_local;
                sigma_I_b += sigma_I_b_local;
            }
        } // end (BB|BB)

  

        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (AA|AA), (BB|AA)
        // #pragma omp parallel
        arma::Mat<double> F3_digestor_a (n_vira, n_vira, fill::zeros);
        {
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_a(Y_bar_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<double> gamma_G1a = YQia_bar_a * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11a = (iQ_bar_a * vectorise(Lam_hA).st()) + (iQ_bar_b * vectorise(Lam_hA).st());
            arma::Mat<double> gamma_J1a(gamma_J11a.memptr(), n_aux*n_occa, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga + gamma_J1a - gamma_J2a;

            arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;

            arma::vec iP (n_aux, fill::zeros);
            iP = (PQinvhalf * iQ_a) + (PQinvhalf * iQ_b);

            // digestor
            arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
            // arma::Mat<double> JG (n_orb, n_occa, fill::zeros);
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
                JG_a.zeros(); 
                dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occa, gamma_Pa, JG_a);
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
                libaview::array_view<double> av_result(JG_a.memptr(), JG_a.n_elem);

            }

            F3_digestor_a = CvirtA.st() * F * Lam_pA;
            
        } // end (AA|AA), (BB|AA)

        // (BB|BB), (AA|BB)
        // #pragma omp parallel
        arma::Mat<double> F3_digestor_b (n_virb, n_virb, fill::zeros);
        {
            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia_bar_b(Y_bar_b.memptr(), n_aux*n_occb, n_virb, false, true);
            arma::Mat<double> gamma_G1b = YQia_bar_b * CvirtB.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J1: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J11b = (iQ_bar_b * vectorise(Lam_hB).st()) + (iQ_bar_a * vectorise(Lam_hB).st());
            arma::Mat<double> gamma_J1b(gamma_J11b.memptr(), n_aux*n_occb, n_orb, false, true);

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_Gb + gamma_J1b - gamma_J2b;

            arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
            gamma_Pb = PQinvhalf * gamma_Qb;

            arma::vec iP (n_aux, fill::zeros);
            iP = (PQinvhalf * iQ_b) + (PQinvhalf * iQ_a);

            // digestor
            arma::Mat<double> F(n_orb, n_orb, arma::fill::zeros);
            // arma::Mat<double> JG (n_orb, n_occb, fill::zeros);
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
                JG_b.zeros(); 
                dig_2e3c_aux<double> dig(m_b3, iP, Fvec, n_occb, gamma_Pb, JG_b);
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
                libaview::array_view<double> av_result(JG_b.memptr(), JG_b.n_elem);

            }

            F3_digestor_b = CvirtB.st() * F * Lam_pB;
            
        } // end (BB|BB), (AA|BB)

        // Fvv_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F33a(F3_digestor_a.memptr(), n_vira, n_vira, false, true);
        arma::Mat<double> Fvv_hat1_a = F33a.st();
        arma::Mat<double> Fvv_hat2_a = BQpoA.st() * BQvoA;
        arma::Mat<double> Fvv_hat_a = f_vv_a + Fvv_hat1_a - Fvv_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F33b(F3_digestor_b.memptr(), n_virb, n_virb, false, true);
        arma::Mat<double> Fvv_hat1_b = F33b.st();
        arma::Mat<double> Fvv_hat2_b = BQpoB.st() * BQvoB;
        arma::Mat<double> Fvv_hat_b = f_vv_b + Fvv_hat1_b - Fvv_hat2_b;

        // Foo_hat
        // (AA|AA), (BB|AA)
        arma::Mat<double> F4a = (iQ_a.st() * BQoh_a) + (iQ_b.st() * BQoh_a);
        arma::Mat<double> F44a(F4a.memptr(), n_occa, n_occa, false, true);
        arma::Mat<double> Foo_hat1_a = F44a.st();
        arma::Mat<double> Foo_hat2_a = BQooA.st() * BQhoA;
        arma::Mat<double> Foo_hat_a = f_oo_a + Foo_hat1_a - Foo_hat2_a;

        // (BB|BB), (AA|BB)
        arma::Mat<double> F4b = (iQ_b.st() * BQoh_b) + (iQ_a.st() * BQoh_b);
        arma::Mat<double> F44b(F4b.memptr(), n_occb, n_occb, false, true);
        arma::Mat<double> Foo_hat1_b = F44b.st();
        arma::Mat<double> Foo_hat2_b = BQooB.st() * BQhoB;
        arma::Mat<double> Foo_hat_b = f_oo_b + Foo_hat1_b - Foo_hat2_b;


        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);

        E_vv_a = Fvv_hat_a - YQiaA.st() * BQvoA; // E_ab
        E_oo_a = Foo_hat_a + (YQaiA.st() * BQovA).st(); // E_ji

        sigma_0_a += (E_vv_a*r1a) - (r1a*E_oo_a);


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b = Fvv_hat_b - YQiaB.st() * BQvoB; // E_ab
        E_oo_b = Foo_hat_b + (YQaiB.st() * BQovB).st(); // E_ji

        sigma_0_b += (E_vv_b*r1b) - (r1b*E_oo_b);

        vec a = vectorise(r1a);
        vec b = vectorise(r1b);
        vec c = join_cols(a,b);

        /// step 6:

        // sigma_JG
        sigma_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
       
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_a(a,i) -= Y_bar_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_a(a,i) = sigma_0_a(a,i) + sigma_JG_a(a,i) + sigma_H_a(a,i) + sigma_I_a(a,i);

                }
            }
        } // end (AA|AA)


        // sigma_JG
        sigma_JG_b += Lam_pB.st() * JG_b;
        
        // (BB|BB)
        #pragma omp parallel
        {
                 
            //transformed vector
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_b(a,i) -= Y_bar_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_b(a,i) = sigma_0_b(a,i) + sigma_JG_b(a,i) + sigma_H_b(a,i) + sigma_I_b(a,i);
                    // sigma_b(a,i) = sigma_0_b(a,i);
                }
            }
        } // end (BB|BB)
        exci = (accu(sigma_a % r1a) + accu(sigma_b % r1b)) / pow(norm(c,"fro"),2);

        // (AA|AA)
        #pragma omp parallel
        {
            // update of the trial vector
            arma::mat res_a (n_vira, n_occa, fill::zeros);
            arma::Mat<double> update_a (n_vira, n_occa, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                        
                    double delta_A = eA(i) - eA[n_occa+a];
                    res_a(a,i) = (sigma_a(a,i) - (exci*r1a(a,i))) / norm(c,"fro");
                    update_a(a,i) = res_a(a,i) / delta_A;
                    r1a(a,i) = (r1a(a,i) + update_a(a,i)) / norm(c,"fro");
                        
                }
            }
        } // end (AA|AA)

        // (BB|BB)
        #pragma omp parallel
        {
            // update of the trial vector
            arma::mat res_b (n_virb, n_occb, fill::zeros);
            arma::mat update_b (n_virb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                        
                    double delta_B = eB(i) - eB[n_occb+a];
                    res_b(a,i) = (sigma_b(a,i) - (exci*r1b(a,i))) / norm(c,"fro");
                    update_b(a,i) = res_b(a,i) / delta_B;
                    r1b(a,i) = (r1b(a,i) + update_b(a,i)) / norm(c,"fro");
                        
                }
            }
        } // end (BB|BB)

    }
}



#if 0
/// GPP: RI-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: NOT THE OPTIMIZED CODE
template<>
void ri_eomee_r<double>::restricted_energy(
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
            ri_motran_incore_buf<double> buf(av_buff_ao);
            motran_2e3c<double, double> mot(op, m_b3, 0.0, m_dev);
            mot.set_trn(Unit);
            mot.run(blst, buf, m_dev);
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
        arma::mat E_vv2 (n_vir, n_vir, fill::zeros);
        arma::mat E_oo2 (n_occ, n_occ, fill::zeros);
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
            ri_motran_incore_buf<double> buf(av_buff_ao);
            motran_2e3c<double, double> mot(op, m_b3, 0.0, m_dev);
            mot.set_trn(Unit);
            mot.run(blst, buf, m_dev);
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

// GPP: activate the complex later by doing templates
//template class ri_eomee_unr_r<double, double>;

}
