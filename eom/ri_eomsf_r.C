#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <armadillo>
#include <libposthf/motran/motran_2e3c.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/arrays/memory_pool.h>
#include "ri_eomsf_r.h"

namespace libgmbpt{
using namespace libposthf;
using namespace libqints;
using namespace arma;

namespace{
    double std_real(double x) { return x; }
    double std_real(const std::complex<double> &x) { return x.real(); }
    std::complex<double> std_conj(const std::complex<double> &x) 
    { 
        std::complex<double> z (x.real(), -x.imag());
        return z; 
    }
}

template<>
void ri_eomsf_r<double>::ccs_spinflip(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQoh_a, Mat<double> &BQho_a,
    Mat<double> &BQoo_a, Mat<double> &BQpv_a, 
    Mat<double> &BQpo_a, Mat<double> &BQov_b, 
    Mat<double> &BQvo_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQpv_b, Mat<double> &BQpo_b, 
    Mat<double> &BQob_ab, Mat<double> &BQob_ba, 
    Mat<double> &BQhp_a, Mat<double> &BQhp_b,
    Mat<double> &BQhb_ba, Mat<double> &BQhb_ab,
    Mat<double> &BQbp_ba, Mat<double> &BQbp_ab,
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
    Mat<double> &r1_Ai, Mat<double> &r1_aI,  
    Mat<double> &res_Ai, Mat<double> &res_aI, 
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_Ai, Mat<double> &sigma_aI) {

        
    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);

    arma::mat sigma_0_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_0_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_JG_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_JG_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_I_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_I_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_H_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_H_aI (n_vira, n_occb, fill::zeros);
    arma::mat Y_bar_Ai (n_aux, n_virb*n_occa, fill::zeros);
    arma::mat Y_bar_aI (n_aux, n_vira*n_occb, fill::zeros);
    
    {  
        exci = 0;
        double t2ab = 0.0, t2aa = 0.0, t2bb = 0.0, t2aa_2 = 0.0, t2bb_2 = 0.0;
        double r2ab = 0.0, r2ba = 0.0;

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij

        // iQ, iQ_bar,
        // (AA|AA)
        iQ_a += BQov_a * t1a;

        // (BB|BB)
        iQ_b += BQov_b * t1b;

        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobAB(BQob_ab.memptr(), n_aux*n_occb, n_occa, false, true);
        arma::Mat<double> BQobBA(BQob_ba.memptr(), n_aux*n_occa, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);

        // Fov_hat
        // (AA|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;
        // (BB|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|BB)
        arma::Mat<double> Fov_bar2_ab = BQobAB.st() * BQvoB;
        arma::Mat<double> Fov_bar_ab = - Fov_bar2_ab;
        // (BB|AA)
        arma::Mat<double> Fov_bar2_ba = BQobBA.st() * BQvoA;
        arma::Mat<double> Fov_bar_ba = - Fov_bar2_ba;

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
        
        E_vv_a = Fvv_hat_a;
        E_oo_a = Foo_hat_a;
        
        E_vv_b = Fvv_hat_b;
        E_oo_b = Foo_hat_b;


        /// step 4:         

        // aiBJ/BiaJ
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
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAJB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAJB
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }

                        // BiaJ
                        // sigma_I_Ai(b,i) += -t2ab * Fov_bar_ba(j,a);
                        sigma_I_Ai_local(b,i) += -t2ab * Fov_bar_ba(j,a);

                    }

                    // AiBJ
                    for(size_t a = 0; a < n_virb; a++) {
                    
                        // iAJB - iBJA + iAJB - iBJA
                        r2ab = (w1b[a] - w1[a*n_virb+b] + w3b[a] - w3[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_b(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_b(j,b);

                    }

                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        } // end parallel (1)


        // AIbj/bIaj

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
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   Iajb
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   Iajb
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        t2ab = w0b[a] / (dijb - eB[n_occb+a]);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }

                        // BiaJ
                        // sigma_I_aI(b,i) += -t2ab * Fov_bar_ab(j,a);
                        sigma_I_aI_local(b,i) += -t2ab * Fov_bar_ab(j,a);

                    }

                    // aIbj
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        // Iajb - Ibja + Iajb - Ibja
                        r2ab = (w1b[a] - w1[a*n_vira+b] + w3b[a] - w3[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_aI_local(a,i) += r2ab * Fov_hat_a(j,b);

                    }

                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                sigma_I_aI += sigma_I_aI_local;
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
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_ab.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_ab.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAjb
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   jAib
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAjb
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   jAib
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }
                    }

                    // Aibj
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        // iAjb - jAib + iAjb - jAib
                        r2ab = (w1b[a] - w2b[a] + w3b[a] - w4b[a]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+j*n_aux+P)] += -r2ab * BQov_a[(i*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_Ai_local(a,j) += -r2ab * Fov_hat_a(i,b);

                    }
                }
            }
            #pragma omp critical (sigma_I_Ai)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        } // end parallel (3)

        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);
        E_vv_a -= YQiaA.st() * BQvoA; // E_ab
        E_oo_a += (YQaiA.st() * BQovA).st(); // E_ji



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
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_ba.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_ba.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   AIBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   IaJB
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   JaIB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   IaJB
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   JaIB
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];

                    // AIBJ
                    for(size_t a = 0; a < n_virb; a++) {
                        t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }
                    }

                    // aIBJ
                    for(size_t a = 0; a < n_vira; a++) {

                        r2ba = (w1b[a] - w2b[a] + w3b[a] - w4b[a]) / (dijb - eA[n_occa+a] + exci);

                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+j*n_aux+P)] += -r2ba * BQov_b[(i*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ba * Fov_hat_b(j,b);
                        sigma_I_aI_local(a,i) += r2ba * Fov_hat_b(j,b);
                        sigma_I_aI_local(a,j) += -r2ba * Fov_hat_b(i,b);

                    }
                }
            }
            #pragma omp critical (sigma_I_aI)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_aI += Y_bar_aI_local;
                sigma_I_aI += sigma_I_aI_local;
            }
        } // end parallel (4)


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b -= YQiaB.st() * BQvoB; // E_ab
        E_oo_b += (YQaiB.st() * BQovB).st(); // E_ji


        sigma_0_Ai += (E_vv_b*r1_Ai) - (r1_Ai*E_oo_a);
        sigma_0_aI += (E_vv_a*r1_aI) - (r1_aI*E_oo_b);

        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
                        
        // (AA|AA), (BB|AA)
        #pragma omp parallel
        {

            // omega_G
            arma::Mat<double> YQiA_bar(Y_bar_Ai.memptr(), n_aux*n_occa, n_virb, false, true);
            arma::Mat<double> gamma_G1a = YQiA_bar * CvirtB.st(); // (n_aux*n_occa,n_virb)*(n_orb,n_virb)=(n_aux*n_occa,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occa, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga - gamma_J2a;
            arma::Mat<double> gamma_Pa = PQinvhalf * gamma_Qa;
            
            arma::Mat<double> JG_a_local (n_orb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occa; i++) {
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

        } // end (AA|AA), (BB|AA)


        // (BB|BB), (AA|BB)
        #pragma omp parallel
        {

            // omega_G
            arma::Mat<double> YQIa_bar(Y_bar_aI.memptr(), n_aux*n_occb, n_vira, false, true);
            arma::Mat<double> gamma_G1b = YQIa_bar * CvirtA.st(); // (n_aux*n_occb,n_vira)*(n_orb,n_vira)=(n_aux*n_occb,n_orb)
            arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occb, n_orb)
            arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_Gb - gamma_J2b;
            arma::Mat<double> gamma_Pb = PQinvhalf * gamma_Qb;
            

            arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occb; i++) {
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
        } // end (BB|BB), (AA|BB)

        vec a = vectorise(r1_Ai);
        vec b = vectorise(r1_aI);
        vec c = join_cols(a,b);


        /// step 6:

        // sigma_JG
        sigma_JG_Ai += Lam_pB.st() * JG_a; // (n_orb,n_virb)*(n_orb,n_occa)

        // (AA|AA) A->B
        #pragma omp parallel
        {
        
            //transformed vector
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_Ai(a,i) -= Y_bar_Ai[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_Ai(a,i) = sigma_0_Ai(a,i) + sigma_JG_Ai(a,i) + sigma_H_Ai(a,i) + sigma_I_Ai(a,i);

                }
            }
        } // end (AA|AA)

        // sigma_JG
        sigma_JG_aI += Lam_pA.st() * JG_b; // (n_orb,n_vira)*(n_orb,n_occb)

        // (BB|BB) B->A
        #pragma omp parallel
        {
                
            //transformed vector
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_aI(a,i) -= Y_bar_aI[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_aI(a,i) = sigma_0_aI(a,i) + sigma_JG_aI(a,i) + sigma_H_aI(a,i) + sigma_I_aI(a,i);

                }
            }
        } // end (BB|BB)

        // cout << "sigma_0_aI: " << accu(sigma_0_aI) << endl;
        // cout << "sigma_JG_aI: " << accu(sigma_JG_aI) << endl;
        // cout << "sigma_H_aI: " << accu(sigma_H_aI) << endl;
        // cout << "sigma_I_aI: " << accu(sigma_I_aI) << endl;

        //exci = (accu(sigma_Ai % r1_Ai) + accu(sigma_aI % r1_aI)) / pow(norm(c,"fro"),2);

        //// (AA|AA)
        //#pragma omp parallel
        //{
        //    // update of the trial vector
        //    res_Ai.zeros();
        //    arma::Mat<double> update_Ai (n_virb, n_occa, fill::zeros);
        //    #pragma omp for
        //    for(size_t a = 0; a < n_virb; a++) {
        //        for(size_t i = 0; i < n_occa; i++) {
        //                
        //            double delta_Ai = eA(i) - eB[n_occb+a];
        //            res_Ai(a,i) = (sigma_Ai(a,i) - (exci*r1_Ai(a,i))) / norm(c,"fro");
        //            update_Ai(a,i) = res_Ai(a,i) / delta_Ai;
        //            r1_Ai(a,i) = (r1_Ai(a,i) + update_Ai(a,i)) / norm(c,"fro");
        //                
        //        }
        //    }
        //} // end (AA|AA)
        //
        //// (BB|BB)
        //#pragma omp parallel
        //{
        //    // update of the trial vector
        //    res_aI.zeros();
        //    arma::mat update_aI (n_vira, n_occb, fill::zeros);
        //    #pragma omp for
        //    for(size_t a = 0; a < n_vira; a++) {
        //        for(size_t i = 0; i < n_occb; i++) {
        //                
        //            double delta_aI = eB(i) - eA[n_occa+a];
        //            res_aI(a,i) = (sigma_aI(a,i) - (exci*r1_aI(a,i))) / norm(c,"fro");
        //            update_aI(a,i) = res_aI(a,i) / delta_aI;
        //            r1_aI(a,i) = (r1_aI(a,i) + update_aI(a,i)) / norm(c,"fro");
        //                
        //        }
        //    }
        //} // end (BB|BB)

    }

}

template<>
void ri_eomsf_r<double>::davidson_spinflip(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQoh_a, Mat<double> &BQho_a,
    Mat<double> &BQoo_a, Mat<double> &BQpv_a, 
    Mat<double> &BQpo_a, Mat<double> &BQov_b, 
    Mat<double> &BQvo_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQpv_b, Mat<double> &BQpo_b, 
    Mat<double> &BQob_ab, Mat<double> &BQob_ba, 
    Mat<double> &BQhp_a, Mat<double> &BQhp_b,
    Mat<double> &BQhb_ba, Mat<double> &BQhb_ab,
    Mat<double> &BQbp_ba, Mat<double> &BQbp_ab,
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
    Mat<double> &r1_Ai, Mat<double> &r1_aI,  
    Mat<double> &res_Ai, Mat<double> &res_aI, 
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_Ai, Mat<double> &sigma_aI) {

        
    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);

    arma::mat sigma_0_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_0_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_JG_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_JG_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_I_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_I_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_H_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_H_aI (n_vira, n_occb, fill::zeros);
    arma::mat Y_bar_Ai (n_aux, n_virb*n_occa, fill::zeros);
    arma::mat Y_bar_aI (n_aux, n_vira*n_occb, fill::zeros);
    
    {  

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij

        // iQ, iQ_bar,
        // (AA|AA)
        iQ_a += BQov_a * t1a;

        // (BB|BB)
        iQ_b += BQov_b * t1b;

        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobAB(BQob_ab.memptr(), n_aux*n_occb, n_occa, false, true);
        arma::Mat<double> BQobBA(BQob_ba.memptr(), n_aux*n_occa, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);

        // Fov_hat
        // (AA|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;
        // (BB|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|BB)
        arma::Mat<double> Fov_bar2_ab = BQobAB.st() * BQvoB;
        arma::Mat<double> Fov_bar_ab = - Fov_bar2_ab;
        // (BB|AA)
        arma::Mat<double> Fov_bar2_ba = BQobBA.st() * BQvoA;
        arma::Mat<double> Fov_bar_ba = - Fov_bar2_ba;

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
        
        E_vv_a = Fvv_hat_a;
        E_oo_a = Foo_hat_a;
        
        E_vv_b = Fvv_hat_b;
        E_oo_b = Foo_hat_b;


        /// step 4:         

        // aiBJ/BiaJ
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
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAJB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAJB
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }

                        // BiaJ
                        // sigma_I_Ai(b,i) += -t2ab * Fov_bar_ba(j,a);
                        sigma_I_Ai_local(b,i) += -t2ab * Fov_bar_ba(j,a);

                    }

                    // AiBJ
                    for(size_t a = 0; a < n_virb; a++) {
                    
                        // iAJB - iBJA + iAJB - iBJA
                        double r2ab = (w1b[a] - w1[a*n_virb+b] + w3b[a] - w3[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_b(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_b(j,b);

                    }

                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        } // end parallel (1)


        // AIbj/bIaj

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
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   Iajb
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   Iajb
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eB[n_occb+a]);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }

                        // BiaJ
                        // sigma_I_aI(b,i) += -t2ab * Fov_bar_ab(j,a);
                        sigma_I_aI_local(b,i) += -t2ab * Fov_bar_ab(j,a);

                    }

                    // aIbj
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        // Iajb - Ibja + Iajb - Ibja
                        double r2ab = (w1b[a] - w1[a*n_vira+b] + w3b[a] - w3[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_aI_local(a,i) += r2ab * Fov_hat_a(j,b);

                    }

                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                sigma_I_aI += sigma_I_aI_local;
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
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_ab.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_ab.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAjb
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   jAib
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAjb
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   jAib
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }
                    }

                    // Aibj
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        // iAjb - jAib + iAjb - jAib
                        double r2ab = (w1b[a] - w2b[a] + w3b[a] - w4b[a]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+j*n_aux+P)] += -r2ab * BQov_a[(i*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_Ai_local(a,j) += -r2ab * Fov_hat_a(i,b);

                    }
                }
            }
            #pragma omp critical (sigma_I_Ai)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        } // end parallel (3)

        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);
        E_vv_a -= YQiaA.st() * BQvoA; // E_ab
        E_oo_a += (YQaiA.st() * BQovA).st(); // E_ji



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
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_ba.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_ba.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   AIBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   IaJB
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   JaIB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   IaJB
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   JaIB
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];

                    // AIBJ
                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }
                    }

                    // aIBJ
                    for(size_t a = 0; a < n_vira; a++) {

                        double r2ba = (w1b[a] - w2b[a] + w3b[a] - w4b[a]) / (dijb - eA[n_occa+a] + exci);

                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+j*n_aux+P)] += -r2ba * BQov_b[(i*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ba * Fov_hat_b(j,b);
                        sigma_I_aI_local(a,i) += r2ba * Fov_hat_b(j,b);
                        sigma_I_aI_local(a,j) += -r2ba * Fov_hat_b(i,b);

                    }
                }
            }
            #pragma omp critical (sigma_I_aI)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_aI += Y_bar_aI_local;
                sigma_I_aI += sigma_I_aI_local;
            }
        } // end parallel (4)


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b -= YQiaB.st() * BQvoB; // E_ab
        E_oo_b += (YQaiB.st() * BQovB).st(); // E_ji


        sigma_0_Ai += (E_vv_b*r1_Ai) - (r1_Ai*E_oo_a);
        sigma_0_aI += (E_vv_a*r1_aI) - (r1_aI*E_oo_b);

        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
                        
        // (AA|AA), (BB|AA)
        #pragma omp parallel
        {

            // omega_G
            arma::Mat<double> YQiA_bar(Y_bar_Ai.memptr(), n_aux*n_occa, n_virb, false, true);
            arma::Mat<double> gamma_G1a = YQiA_bar * CvirtB.st(); // (n_aux*n_occa,n_virb)*(n_orb,n_virb)=(n_aux*n_occa,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occa, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga - gamma_J2a;
            arma::Mat<double> gamma_Pa = PQinvhalf * gamma_Qa;
            
            arma::Mat<double> JG_a_local (n_orb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occa; i++) {
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

        } // end (AA|AA), (BB|AA)


        // (BB|BB), (AA|BB)
        #pragma omp parallel
        {

            // omega_G
            arma::Mat<double> YQIa_bar(Y_bar_aI.memptr(), n_aux*n_occb, n_vira, false, true);
            arma::Mat<double> gamma_G1b = YQIa_bar * CvirtA.st(); // (n_aux*n_occb,n_vira)*(n_orb,n_vira)=(n_aux*n_occb,n_orb)
            arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occb, n_orb)
            arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_Gb - gamma_J2b;
            arma::Mat<double> gamma_Pb = PQinvhalf * gamma_Qb;
            

            arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occb; i++) {
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
        } // end (BB|BB), (AA|BB)

        vec a = vectorise(r1_Ai);
        vec b = vectorise(r1_aI);
        vec c = join_cols(a,b);


        /// step 6:

        // sigma_JG
        sigma_JG_Ai += Lam_pB.st() * JG_a; // (n_orb,n_virb)*(n_orb,n_occa)

        // (AA|AA) A->B
        #pragma omp parallel
        {
        
            //transformed vector
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_Ai(a,i) -= Y_bar_Ai[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_Ai(a,i) = sigma_0_Ai(a,i) + sigma_JG_Ai(a,i) + sigma_H_Ai(a,i) + sigma_I_Ai(a,i);

                }
            }
        } // end (AA|AA)

        // sigma_JG
        sigma_JG_aI += Lam_pA.st() * JG_b; // (n_orb,n_vira)*(n_orb,n_occb)

        // (BB|BB) B->A
        #pragma omp parallel
        {
                
            //transformed vector
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_aI(a,i) -= Y_bar_aI[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_aI(a,i) = sigma_0_aI(a,i) + sigma_JG_aI(a,i) + sigma_H_aI(a,i) + sigma_I_aI(a,i);

                }
            }
        } // end (BB|BB)

        //exci = (accu(sigma_Ai % r1_Ai) + accu(sigma_aI % r1_aI)) / pow(norm(c,"fro"),2);
        //
        //// (AA|AA)
        //#pragma omp parallel
        //{
        //    // update of the trial vector
        //    res_Ai.zeros();
        //    arma::Mat<double> update_Ai (n_virb, n_occa, fill::zeros);
        //    #pragma omp for
        //    for(size_t a = 0; a < n_virb; a++) {
        //        for(size_t i = 0; i < n_occa; i++) {
        //                
        //            double delta_Ai = eA(i) - eB[n_occb+a];
        //            res_Ai(a,i) = (sigma_Ai(a,i) - (exci*r1_Ai(a,i))) / norm(c,"fro");
        //            update_Ai(a,i) = res_Ai(a,i) / delta_Ai;
        //            r1_Ai(a,i) = (r1_Ai(a,i) + update_Ai(a,i)) / norm(c,"fro");
        //                
        //        }
        //    }
        //} // end (AA|AA)
        //
        //// (BB|BB)
        //#pragma omp parallel
        //{
        //    // update of the trial vector
        //    res_aI.zeros();
        //    arma::mat update_aI (n_vira, n_occb, fill::zeros);
        //    #pragma omp for
        //    for(size_t a = 0; a < n_vira; a++) {
        //        for(size_t i = 0; i < n_occb; i++) {
        //                
        //            double delta_aI = eB(i) - eA[n_occa+a];
        //            res_aI(a,i) = (sigma_aI(a,i) - (exci*r1_aI(a,i))) / norm(c,"fro");
        //            update_aI(a,i) = res_aI(a,i) / delta_aI;
        //            r1_aI(a,i) = (r1_aI(a,i) + update_aI(a,i)) / norm(c,"fro");
        //                
        //        }
        //    }
        //} // end (BB|BB)

    }

}

template<>
void ri_eomsf_r<double>::diis_spinflip(
    double &exci, const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb, 
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQoh_a, Mat<double> &BQho_a,
    Mat<double> &BQoo_a, Mat<double> &BQpv_a, 
    Mat<double> &BQpo_a, Mat<double> &BQov_b, 
    Mat<double> &BQvo_b, Mat<double> &BQoh_b, 
    Mat<double> &BQho_b, Mat<double> &BQoo_b, 
    Mat<double> &BQpv_b, Mat<double> &BQpo_b, 
    Mat<double> &BQob_ab, Mat<double> &BQob_ba, 
    Mat<double> &BQhp_a, Mat<double> &BQhp_b,
    Mat<double> &BQhb_ba, Mat<double> &BQhb_ab,
    Mat<double> &BQbp_ba, Mat<double> &BQbp_ab,
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
    Mat<double> &r1_Ai, Mat<double> &r1_aI,  
    Mat<double> &res_Ai, Mat<double> &res_aI, 
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    Mat<double> &sigma_Ai, Mat<double> &sigma_aI) {

        
    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat E_vv_a (n_vira, n_vira, fill::zeros);
    arma::mat E_oo_a (n_occa, n_occa, fill::zeros);
    arma::mat Yia_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat Yai_a (n_aux, n_vira*n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat E_vv_b (n_virb, n_virb, fill::zeros);
    arma::mat E_oo_b (n_occb, n_occb, fill::zeros);
    arma::mat Yia_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat Yai_b (n_aux, n_virb*n_occb, fill::zeros);

    arma::mat sigma_0_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_0_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_JG_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_JG_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_I_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_I_aI (n_vira, n_occb, fill::zeros);
    arma::mat sigma_H_Ai (n_virb, n_occa, fill::zeros);
    arma::mat sigma_H_aI (n_vira, n_occb, fill::zeros);
    arma::mat Y_bar_Ai (n_aux, n_virb*n_occa, fill::zeros);
    arma::mat Y_bar_aI (n_aux, n_vira*n_occb, fill::zeros);
    
    {  

        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij

        // iQ, iQ_bar,
        // (AA|AA)
        iQ_a += BQov_a * t1a;

        // (BB|BB)
        iQ_b += BQov_b * t1b;

        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQobAB(BQob_ab.memptr(), n_aux*n_occb, n_occa, false, true);
        arma::Mat<double> BQobBA(BQob_ba.memptr(), n_aux*n_occa, n_occb, false, true);
        arma::Mat<double> BQpoA(BQpo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQpoB(BQpo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQhoA(BQho_a.memptr(), n_aux*n_occa, n_occa, false, true);
        arma::Mat<double> BQhoB(BQho_b.memptr(), n_aux*n_occb, n_occb, false, true);
        arma::Mat<double> BQovA(BQov_a.memptr(), n_aux*n_vira, n_occa, false, true);
        arma::Mat<double> BQovB(BQov_b.memptr(), n_aux*n_virb, n_occb, false, true);

        // Fov_hat
        // (AA|AA)
        arma::Mat<double> F1a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F11a(F1a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> Fov_hat1_a = F11a.st();
        arma::Mat<double> Fov_hat2_a = BQooA.st() * BQvoA;
        arma::Mat<double> Fov_hat_a = Fov_hat1_a - Fov_hat2_a;
        // (BB|BB)
        arma::Mat<double> F1b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F11b(F1b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> Fov_hat1_b = F11b.st();
        arma::Mat<double> Fov_hat2_b = BQooB.st() * BQvoB;
        arma::Mat<double> Fov_hat_b = Fov_hat1_b - Fov_hat2_b;

        // Fov_bar
        // (AA|BB)
        arma::Mat<double> Fov_bar2_ab = BQobAB.st() * BQvoB;
        arma::Mat<double> Fov_bar_ab = - Fov_bar2_ab;
        // (BB|AA)
        arma::Mat<double> Fov_bar2_ba = BQobBA.st() * BQvoA;
        arma::Mat<double> Fov_bar_ba = - Fov_bar2_ba;

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
        
        E_vv_a = Fvv_hat_a;
        E_oo_a = Foo_hat_a;
        
        E_vv_b = Fvv_hat_b;
        E_oo_b = Foo_hat_b;


        /// step 4:         

        // aiBJ/BiaJ
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
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aiBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAJB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAJB
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eA[n_occa+a]);
                        
                        // aiBJ
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                        }

                        // BiaJ
                        // sigma_I_Ai(b,i) += -t2ab * Fov_bar_ba(j,a);
                        sigma_I_Ai_local(b,i) += -t2ab * Fov_bar_ba(j,a);

                    }

                    // AiBJ
                    for(size_t a = 0; a < n_virb; a++) {
                    
                        // iAJB - iBJA + iAJB - iBJA
                        double r2ab = (w1b[a] - w1[a*n_virb+b] + w3b[a] - w3[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_b(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_b(j,b);

                    }

                }
            }
            #pragma omp critical (Y_a)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        } // end parallel (1)


        // AIbj/bIaj

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
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   AIbj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   Iajb
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   Iajb
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ab = w0b[a] / (dijb - eB[n_occb+a]);
                        
                        // AIbj
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += t2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                        }

                        // BiaJ
                        // sigma_I_aI(b,i) += -t2ab * Fov_bar_ab(j,a);
                        sigma_I_aI_local(b,i) += -t2ab * Fov_bar_ab(j,a);

                    }

                    // aIbj
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        // Iajb - Ibja + Iajb - Ibja
                        double r2ab = (w1b[a] - w1[a*n_vira+b] + w3b[a] - w3[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_aI_local(a,i) += r2ab * Fov_hat_a(j,b);

                    }

                }
            }
            #pragma omp critical (Y_b)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                sigma_I_aI += sigma_I_aI_local;
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
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_j(BQhb_ab.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_j(BQbp_ab.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAjb
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   jAib
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAjb
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   jAib
                
                double delta_ij = eA(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_virb;
                    const double *w2b = w2 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;
                    const double *w4b = w4 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aibj
                    for(size_t a = 0; a < n_vira; a++) {
                        double t2aa = w0b[a] / (dijb - eA[n_occa+a]);
                        double t2aa_2 = w0[a*n_vira+b] / (dijb - eA[n_occa+a]);

                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_a[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yia_a[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            // Yai_a[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            // Yai_a[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yia_a_local[(a*n_occa*n_aux+i*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yia_a_local[(b*n_occa*n_aux+j*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                            Yai_a_local[(i*n_vira*n_aux+a*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            Yai_a_local[(j*n_vira*n_aux+b*n_aux+Q)] += (t2aa-t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];
                        }
                    }

                    // Aibj
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        // iAjb - jAib + iAjb - jAib
                        double r2ab = (w1b[a] - w2b[a] + w3b[a] - w4b[a]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+j*n_aux+P)] += -r2ab * BQov_a[(i*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_Ai_local(a,j) += -r2ab * Fov_hat_a(i,b);

                    }
                }
            }
            #pragma omp critical (sigma_I_Ai)
            {
                Yia_a += Yia_a_local;
                Yai_a += Yai_a_local;
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        } // end parallel (3)

        arma::Mat<double> YQiaA(Yia_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> YQaiA(Yai_a.memptr(), n_aux*n_vira, n_occa, false, true);
        E_vv_a -= YQiaA.st() * BQvoA; // E_ab
        E_oo_a += (YQaiA.st() * BQovA).st(); // E_ji



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
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;
                
                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for r2: 
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_j(BQhb_ba.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_j(BQbp_ba.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   AIBJ
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   IaJB
                arma::Mat<double> W2 = Bhb_j.st() * Bhp_i; // r2:   JaIB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   IaJB
                arma::Mat<double> W4 = Bbp_j.st() * Bhp_i; // r2:   JaIB
                
                double delta_ij = eB(i)+eB(j);
                
                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();
                const double *w2 = W2.memptr();
                const double *w3 = W3.memptr();
                const double *w4 = W4.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                        
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_vira;
                    const double *w2b = w2 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;
                    const double *w4b = w4 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];

                    // AIBJ
                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w0b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w0[a*n_virb+b] / (dijb - eB[n_occb+a]);
                            
                        for(size_t Q = 0; Q < n_aux; Q++) {
                            // Yia_b[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yia_b[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            // Yai_b[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            // Yai_b[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yia_b_local[(a*n_occb*n_aux+i*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yia_b_local[(b*n_occb*n_aux+j*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                            Yai_b_local[(i*n_virb*n_aux+a*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            Yai_b_local[(j*n_virb*n_aux+b*n_aux+Q)] += (t2bb-t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];
                        }
                    }

                    // aIBJ
                    for(size_t a = 0; a < n_vira; a++) {

                        double r2ba = (w1b[a] - w2b[a] + w3b[a] - w4b[a]) / (dijb - eA[n_occa+a] + exci);

                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+j*n_aux+P)] += -r2ba * BQov_b[(i*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ba * Fov_hat_b(j,b);
                        sigma_I_aI_local(a,i) += r2ba * Fov_hat_b(j,b);
                        sigma_I_aI_local(a,j) += -r2ba * Fov_hat_b(i,b);

                    }
                }
            }
            #pragma omp critical (sigma_I_aI)
            {
                Yia_b += Yia_b_local;
                Yai_b += Yai_b_local;
                Y_bar_aI += Y_bar_aI_local;
                sigma_I_aI += sigma_I_aI_local;
            }
        } // end parallel (4)


        arma::Mat<double> YQiaB(Yia_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> YQaiB(Yai_b.memptr(), n_aux*n_virb, n_occb, false, true);
        E_vv_b -= YQiaB.st() * BQvoB; // E_ab
        E_oo_b += (YQaiB.st() * BQovB).st(); // E_ji


        sigma_0_Ai += (E_vv_b*r1_Ai) - (r1_Ai*E_oo_a);
        sigma_0_aI += (E_vv_a*r1_aI) - (r1_aI*E_oo_b);

        /// step 5:
        
        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
                        
        // (AA|AA), (BB|AA)
        #pragma omp parallel
        {

            // omega_G
            arma::Mat<double> YQiA_bar(Y_bar_Ai.memptr(), n_aux*n_occa, n_virb, false, true);
            arma::Mat<double> gamma_G1a = YQiA_bar * CvirtB.st(); // (n_aux*n_occa,n_virb)*(n_orb,n_virb)=(n_aux*n_occa,n_orb)
            arma::Mat<double> gamma_Ga = gamma_G1a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_Ga.insert_cols(i*n_orb, gamma_G1a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_J22a = BQohA * (Lam_hA_bar).st(); // (n_aux*n_occa, n_orb)
            arma::Mat<double> gamma_J2a = gamma_J22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_J2a.insert_cols(i*n_orb, gamma_J22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_Ga - gamma_J2a;
            arma::Mat<double> gamma_Pa = PQinvhalf * gamma_Qa;
            
            arma::Mat<double> JG_a_local (n_orb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occa; i++) {
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

        } // end (AA|AA), (BB|AA)


        // (BB|BB), (AA|BB)
        #pragma omp parallel
        {

            // omega_G
            arma::Mat<double> YQIa_bar(Y_bar_aI.memptr(), n_aux*n_occb, n_vira, false, true);
            arma::Mat<double> gamma_G1b = YQIa_bar * CvirtA.st(); // (n_aux*n_occb,n_vira)*(n_orb,n_vira)=(n_aux*n_occb,n_orb)
            arma::Mat<double> gamma_Gb = gamma_G1b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_Gb.insert_cols(i*n_orb, gamma_G1b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // / omega_J2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_J22b = BQohB * (Lam_hB_bar).st(); // (n_aux*n_occb, n_orb)
            arma::Mat<double> gamma_J2b = gamma_J22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_J2b.insert_cols(i*n_orb, gamma_J22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_Gb - gamma_J2b;
            arma::Mat<double> gamma_Pb = PQinvhalf * gamma_Qb;
            

            arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occb; i++) {
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
        } // end (BB|BB), (AA|BB)

        vec a = vectorise(r1_Ai);
        vec b = vectorise(r1_aI);
        vec c = join_cols(a,b);


        /// step 6:

        // sigma_JG
        sigma_JG_Ai += Lam_pB.st() * JG_a; // (n_orb,n_virb)*(n_orb,n_occa)

        // (AA|AA) A->B
        #pragma omp parallel
        {
        
            //transformed vector
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            sigma_H_Ai(a,i) -= Y_bar_Ai[(a*n_occa*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_Ai(a,i) = sigma_0_Ai(a,i) + sigma_JG_Ai(a,i) + sigma_H_Ai(a,i) + sigma_I_Ai(a,i);

                }
            }
        } // end (AA|AA)

        // sigma_JG
        sigma_JG_aI += Lam_pA.st() * JG_b; // (n_orb,n_vira)*(n_orb,n_occb)

        // (BB|BB) B->A
        #pragma omp parallel
        {
                
            //transformed vector
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    
                    // sigma_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            sigma_H_aI(a,i) -= Y_bar_aI[(a*n_occb*n_aux+k*n_aux+P)]
                                                * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }
        
                    sigma_aI(a,i) = sigma_0_aI(a,i) + sigma_JG_aI(a,i) + sigma_H_aI(a,i) + sigma_I_aI(a,i);

                }
            }
        } // end (BB|BB)

        exci = (accu(sigma_Ai % r1_Ai) + accu(sigma_aI % r1_aI)) / pow(norm(c,"fro"),2);

        // (AA|AA)
        #pragma omp parallel
        {
            // update of the trial vector
            res_Ai.zeros();
            arma::Mat<double> update_Ai (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                        
                    double delta_Ai = eA(i) - eB[n_occb+a];
                    res_Ai(a,i) = (sigma_Ai(a,i) - (exci*r1_Ai(a,i))) / norm(c,"fro");
                    update_Ai(a,i) = res_Ai(a,i) / delta_Ai;
                    r1_Ai(a,i) = (r1_Ai(a,i) + update_Ai(a,i)) / norm(c,"fro");
                        
                }
            }
        } // end (AA|AA)

        // (BB|BB)
        #pragma omp parallel
        {
            // update of the trial vector
            res_aI.zeros();
            arma::mat update_aI (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                        
                    double delta_aI = eB(i) - eA[n_occa+a];
                    res_aI(a,i) = (sigma_aI(a,i) - (exci*r1_aI(a,i))) / norm(c,"fro");
                    update_aI(a,i) = res_aI(a,i) / delta_aI;
                    r1_aI(a,i) = (r1_aI(a,i) + update_aI(a,i)) / norm(c,"fro");
                        
                }
            }
        } // end (BB|BB)

    }

}

// GPP: These are nested for loops used before for step 4

        /*

        // (AA|BB)
        #pragma omp parallel
        {
            
            // aiBJ
            arma::mat E_vv_a_local (n_vira, n_vira, fill::zeros);
            arma::mat E_oo_a_local (n_occa, n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t j = 0; j < n_occb; j++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t a = 0; a < n_vira; a++) {
                            
                            //denominator
                            double delta_iJaB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];
                            double t2ab = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // t2ab += BQph_a[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)];
                                t2ab += BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                            }
                            
                            t2ab = t2ab / delta_iJaB;
                            // cout << "i: " << i << " j: " << j << " b: " << b << " a: " << a << "  t2ab loop: " << t2ab << endl;
                            
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
                            
                            // sigma_I
                            // BiaJ
                            // sigma_I_Ai(b,i) += -t2ab * Fov_bar_ba(j,a);
                            sigma_I_Ai_local(b,i) += -t2ab * Fov_bar_ba(j,a);
                            // sigma_I_Ai.print("sigma_I_Ai");

                        }
                    }
                }
            }
            #pragma omp critical (E_a)
            {
                E_vv_a += E_vv_a_local;
                E_oo_a += E_oo_a_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        }

        // (BB|AA)
        #pragma omp parallel
        {
            
            // AIbj
            arma::mat E_vv_b_local (n_virb, n_virb, fill::zeros);
            arma::mat E_oo_b_local (n_occb, n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_vira; b++) {
                for(size_t a = 0; a < n_virb; a++) {
                    for(size_t i = 0; i < n_occb; i++) {
                        for(size_t j = 0; j < n_occa; j++) {
                            
                            //denominator
                            double delta_IjAb = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];
                            double t2ba = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // t2ba += BQph_b[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)];
                                t2ba += BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                            }
                            
                            t2ba = t2ba / delta_IjAb;
                            
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

                            // sigma_I
                            // sigma_I_aI(a,i) += -(t2ba * Fov_bar_ab(j,b));
                            sigma_I_aI_local(b,i) += -(t2ba * Fov_bar_ab(j,a));

                        }
                    }
                }
            }
            #pragma omp critical (E_b)
            {
                E_vv_b += E_vv_b_local;
                E_oo_b += E_oo_b_local;
                sigma_I_aI += sigma_I_aI_local;
            }
        }

        #pragma omp parallel
        {   
            // AibJ
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_vira; b++) {
                for(size_t a = 0; a < n_virb; a++) {
                    for(size_t i = 0; i < n_occa; i++) {
                        for(size_t j = 0; j < n_occb; j++) {
                            
                            //denominator
                            double delta_iJAb = eA(i) + eB(j) - eB[n_occb+a] - eA[n_occa+b];
                            double t2ab = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                // t2ab += BQph_a[(b*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(a*n_occb*n_aux+j*n_aux+Q)];      // biAJ 
                                t2ab += BQhp_a[(i*n_vira*n_aux+b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)];      // biJA
                                
                            }
                            
                            t2ab = t2ab / delta_iJAb;
                            
                            // sigma_I
                            // sigma_I_Ai(a,i) += -(t2ab * Fov_bar_ba(j,b));
                            sigma_I_Ai_local(a,i) += -(t2ab * Fov_bar_ba(j,b));
                            
                        }
                    }
                }
            }
            #pragma omp critical (I_a)
            {
                sigma_I_Ai += sigma_I_Ai_local;
                sigma_I_Ai_2 += sigma_I_Ai_local;
            }
        }


        #pragma omp parallel
        {
            // aIBj
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_virb; b++) {
                for(size_t a = 0; a < n_vira; a++) {
                    for(size_t i = 0; i < n_occb; i++) {
                        for(size_t j = 0; j < n_occa; j++) {
                            
                            //denominator
                            double delta_IjaB = eB(i) + eA(j) - eA[n_occa+a] - eB[n_occb+b];
                            double t2ba = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                // t2ba += BQph_b[(b*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occa*n_aux+j*n_aux+Q)];  // BIaj
                                t2ba += BQhp_b[(i*n_virb*n_aux+b*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)];  // BIaj

                            }
                            
                            t2ba = t2ba / delta_IjaB;


                            // sigma_I
                            // sigma_I_aI(a,i) += -(t2ba * Fov_bar_ab(j,b));
                            sigma_I_aI_local(a,i) += -(t2ba * Fov_bar_ab(j,b));
                            
                        }
                    }
                }
            }
            #pragma omp critical (I_b)
            {
                sigma_I_aI += sigma_I_aI_local;
            }
        }
           

        #pragma omp parallel
        {
            // AiBJ
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            #pragma omp for

            for(size_t i = 0; i < n_occa; i++) {
                for(size_t j = 0; j < n_occb; j++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t a = 0; a < n_virb; a++) {
                            
                            //denominator
                            double delta_iJAB = eA(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];
                            double r2ab = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                                                        
                                // r2ab += BQbh_ba[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)]      // AiBJ
                                //         - BQbh_ba[(b*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(a*n_occb*n_aux+j*n_aux+Q)]    // BiAJ
                                //         + BQpb_ba[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)]    // AiBJ
                                //         - BQpb_ba[(b*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(a*n_occb*n_aux+j*n_aux+Q)]    // BiAJ
                                //         ;

                                r2ab += BQhb_ab[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]      // iAJB
                                        - BQhb_ab[(i*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)]    // iBJA
                                        + BQbp_ab[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]    // iAJB
                                        - BQbp_ab[(i*n_virb*n_aux+b*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)]    // BiAJ
                                        ;
                                
                            }
                            
                            r2ab = r2ab / (delta_iJAB + exci);
                            // cout << "i: " << i << " j: " << j << " b: " << b << " a: " << a << "  r2ab loop: " << r2ab << endl;
               
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                                Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_Ai(a,i) += (r2ab * Fov_hat_b(j,b));
                            sigma_I_Ai_local(a,i) += (r2ab * Fov_hat_b(j,b));
                            
                        }
                    }
                }
            } 
            #pragma omp critical (Y_a)
            {
                sigma_I_Ai += sigma_I_Ai_local;
                Y_bar_Ai += Y_bar_Ai_local;
            }

        } // end (AA|BB)

        #pragma omp parallel
        {
            // aIbj
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_vira; b++) {
                for(size_t a = 0; a < n_vira; a++) {
                    for(size_t i = 0; i < n_occb; i++) {
                        for(size_t j = 0; j < n_occa; j++) {
                            
                            //denominator
                            double delta_Ijab = eB(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];
                            double r2ba = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                
                                // r2ba += BQbh_ab[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)]      // aIbj
                                //         - BQbh_ab[(b*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occa*n_aux+j*n_aux+Q)]    // bIaj
                                //         + BQpb_ab[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)]    // aIbj
                                //         - BQpb_ab[(b*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occa*n_aux+j*n_aux+Q)]    // bIaj
                                //         ;

                                
                                r2ba += BQhb_ba[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]      // aIbj
                                        - BQhb_ba[(i*n_vira*n_aux+b*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)]    // bIaj
                                        + BQbp_ba[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]    // aIbj
                                        - BQbp_ba[(i*n_vira*n_aux+b*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)]    // bIaj
                                        ;

                            }
                            
                            r2ba = r2ba / (delta_Ijab + exci);
                                                                     
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }
                            
                                                                       
                            // sigma_I
                            // sigma_I_aI(a,i) += (r2ba * Fov_hat_a(j,b));
                            sigma_I_aI_local(a,i) += (r2ba * Fov_hat_a(j,b));
                            
                        }
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                sigma_I_aI += sigma_I_aI_local;
                Y_bar_aI += Y_bar_aI_local;
            }
        } // end (BB|AA)
        

        #pragma omp parallel
        {
            // aibj
            arma::mat E_vv_a_local (n_vira, n_vira, fill::zeros);
            arma::mat E_oo_a_local (n_occa, n_occa, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_vira; b++) {
                for(size_t a = 0; a < n_vira; a++) {
                    for(size_t i = 0; i < n_occa; i++) {
                        for(size_t j = 0; j < n_occa; j++) {
                            
                            //denominator
                            double delta_AA = eA(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];
                            double t2aa = 0.0;
                            double t2aa_2 = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // t2aa += BQph_a[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)];
                                // t2aa_2 += BQph_a[(b*n_occa*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occa*n_aux+j*n_aux+Q)];
                                t2aa += BQph_a[(a*n_occa*n_aux+i*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];
                                t2aa_2 += BQph_a[(b*n_occa*n_aux+i*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+a*n_aux+Q)];
                            }
                            
                            t2aa = t2aa / delta_AA;
                            t2aa_2 = t2aa_2 / delta_AA;
                            
                            for(size_t c = 0; c < n_vira; c++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    E_vv_a(a,c) -= (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vira*n_aux+c*n_aux+Q)];
                                    // E_vv_a_local(a,c) -= (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(i*n_vira*n_aux+c*n_aux+Q)];
                                }
                            }
                                
                            for(size_t k = 0; k < n_occa; k++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    E_oo_a(k,i) += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vira*n_aux+a*n_aux+Q)];
                                    // E_oo_a_local(k,i) += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(k*n_vira*n_aux+a*n_aux+Q)];
                                }
                            }
                                                        
                        }
                    }
                }
            }
            #pragma omp critical (E_a)
            {
                E_vv_a += E_vv_a_local;
                E_oo_a += E_oo_a_local;
            }
        }
        
        #pragma omp parallel
        {
            // Aibj
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            #pragma omp for
            for(size_t j = 0; j < n_occa; j++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t a = 0; a < n_virb; a++) {
                            
                            //denominator
                            double delta_ijAb = eA(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];
                            double r2aa = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {

                                
                                // r2aa += BQbh_ba[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)]      // Aibj
                                //         - BQbh_ba[(a*n_occa*n_aux+j*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+i*n_aux+Q)]    // Ajbi
                                //         + BQpb_ba[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)]    // Aibj
                                //         - BQpb_ba[(a*n_occa*n_aux+j*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+i*n_aux+Q)]    // Ajbi
                                //         ;

                                
                                r2aa += BQhb_ab[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]      // iAjb
                                        - BQhb_ab[(j*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+b*n_aux+Q)]    // jAib
                                        + BQbp_ab[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)]    // iAjb
                                        - BQbp_ab[(j*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(i*n_vira*n_aux+b*n_aux+Q)]    // jAib
                                        ;

                            }
                            
                            r2aa = r2aa / (delta_ijAb + exci);
                            
                            // cout << "i: " << i << " j: " << j << " b: " << b << " a: " << a << "  r2aa: " << r2aa << endl;
                            
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2aa * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2aa * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_Ai(a,i) += r2aa * Fov_hat_a(j,b);
                            sigma_I_Ai_local(a,i) += r2aa * Fov_hat_a(j,b);
                            
                            // sigma_I_Ai_local.print("sigma_I_Ai_local");
                            // cout << "sigma_I_Ai_local: " << std::setprecision(10) << accu(sigma_I_Ai_local) << endl;

                        }
                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                // sigma_I_Ai += sigma_I_Ai_local;
                // Y_bar_Ai += Y_bar_Ai_local;
            }
        } // end (AA|AA)


        #pragma omp parallel
        {
            
            // AIBJ
            arma::mat E_vv_b_local (n_virb, n_virb, fill::zeros);
            arma::mat E_oo_b_local (n_occb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_virb; b++) {
                for(size_t a = 0; a < n_virb; a++) {
                    for(size_t i = 0; i < n_occb; i++) {
                        for(size_t j = 0; j < n_occb; j++) {
                            
                            //denominator
                            double delta_BB = eB(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];
                            double t2bb = 0.0;
                            double t2bb_2 = 0.0;

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // t2bb += BQph_b[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)];
                                // t2bb_2 += BQph_b[(b*n_occb*n_aux+i*n_aux+Q)]*BQph_b[(a*n_occb*n_aux+j*n_aux+Q)];
                                t2bb += BQph_b[(a*n_occb*n_aux+i*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];
                                t2bb_2 += BQph_b[(b*n_occb*n_aux+i*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+a*n_aux+Q)];
                            }
                            
                            t2bb = t2bb / delta_BB;
                            t2bb_2 = t2bb_2 / delta_BB;
                            
                            for(size_t c = 0; c < n_virb; c++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_vv_b(a,c) -= (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(i*n_virb*n_aux+c*n_aux+Q)];
                                    E_vv_b_local(a,c) -= (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(i*n_virb*n_aux+c*n_aux+Q)];
                                }
                            }
                            
                            for(size_t k = 0; k < n_occb; k++) {
                                for(size_t Q = 0; Q < n_aux; Q++) {
                                    // E_oo_b(k,i) += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(k*n_virb*n_aux+a*n_aux+Q)];
                                    E_oo_b_local(k,i) += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(k*n_virb*n_aux+a*n_aux+Q)];
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp critical (E_b)
            {
                E_vv_b += E_vv_b_local;
                E_oo_b += E_oo_b_local;
            }
        }

        #pragma omp parallel
        {
            // aIBJ
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            #pragma omp for
            for(size_t b = 0; b < n_virb; b++) {
                for(size_t a = 0; a < n_vira; a++) {
                    for(size_t i = 0; i < n_occb; i++) {
                        for(size_t j = 0; j < n_occb; j++) {
                            
                            //denominator
                            double delta_IJaB = eB(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];
                            double r2bb = 0.0;
                            
                            for(size_t Q = 0; Q < n_aux; Q++) {
                                
                                // r2bb += BQbh_ab[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)]     // aIBJ
                                //         - BQbh_ab[(a*n_occb*n_aux+j*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+i*n_aux+Q)]   // aJBI
                                //         + BQpb_ab[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)]   // aIBJ
                                //         - BQpb_ab[(a*n_occb*n_aux+j*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+i*n_aux+Q)]   // aJBI
                                //         ;
                                
                                r2bb += BQbh_ab[(a*n_occb*n_aux+i*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]     // aIBJ
                                        - BQbh_ab[(a*n_occb*n_aux+j*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+b*n_aux+Q)]   // aJBI
                                        + BQpb_ab[(a*n_occb*n_aux+i*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)]   // aIBJ
                                        - BQpb_ab[(a*n_occb*n_aux+j*n_aux+Q)]*BQhp_b[(i*n_virb*n_aux+b*n_aux+Q)]  // aJBI
                                        ;
                                        
                            }
                            r2bb = r2bb / (delta_IJaB + exci);
                                                        
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2bb * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                                Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2bb * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }
                            
                            // sigma_I
                            // sigma_I_aI(a,i) += r2bb * Fov_hat_b(j,b);
                            sigma_I_aI_local(a,i) += r2bb * Fov_hat_b(j,b);
                            
                        }
                    }
                }
            }
            #pragma omp critical (Y_b)
            {
                sigma_I_aI += sigma_I_aI_local;
                Y_bar_aI += Y_bar_aI_local;
            }
        } // end (BB|BB)


        // AiBJ
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

            arma::mat Y_bar_Ai_local (n_aux, n_virb*n_occa, fill::zeros);
            arma::mat sigma_I_Ai_local (n_virb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for r2: 
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhb_i(BQhb_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bbp_i(BQbp_ab.colptr(i*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   iAJB
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   iAJB
                
                double delta_ij = eA(i) + eB(j);

                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w1b = w1 + b * n_virb;
                    const double *w3b = w3 + b * n_virb;

                    double dijb = delta_ij - eB[n_occb+b];

                    // AiBJ
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        // iAJB - iBJA + iAJB - iBJA
                        double r2ab = (w1b[a] - w1[a*n_virb+b] + w3b[a] - w3[a*n_virb+b]) / (dijb - eB[n_occb+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_Ai[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_bar_Ai_local[(a*n_occa*n_aux+i*n_aux+P)] += r2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_Ai(a,i) += r2ab * Fov_hat_b(j,b);
                        sigma_I_Ai_local(a,i) += r2ab * Fov_hat_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_bar_a)
            {
                Y_bar_Ai += Y_bar_Ai_local;
                sigma_I_Ai += sigma_I_Ai_local;
            }
        }


        // aIbj
        #pragma omp parallel
        {

            size_t npairs = n_occa*n_occb;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occb, n_occa, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occb; i++) {
            for(size_t j = 0; j < n_occa; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            arma::mat Y_bar_aI_local (n_aux, n_vira*n_occb, fill::zeros);
            arma::mat sigma_I_aI_local (n_vira, n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // for r2: 
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhb_i(BQhb_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bbp_i(BQbp_ba.colptr(i*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W1 = Bhb_i.st() * Bhp_j; // r2:   Iajb
                arma::Mat<double> W3 = Bbp_i.st() * Bhp_j; // r2:   Iajb
                
                double delta_ij = eB(i) + eA(j);

                const double *w1 = W1.memptr();
                const double *w3 = W3.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w1b = w1 + b * n_vira;
                    const double *w3b = w3 + b * n_vira;

                    double dijb = delta_ij - eA[n_occa+b];

                    // aIbj
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        // Iajb - Ibja + Iajb - Ibja
                        double r2ab = (w1b[a] - w1[a*n_vira+b] + w3b[a] - w3[a*n_vira+b]) / (dijb - eA[n_occa+a] + exci);
                    
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar_aI[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            Y_bar_aI_local[(a*n_occb*n_aux+i*n_aux+P)] += r2ab * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                        }

                        // sigma_I_aI(a,i) += r2ab * Fov_hat_a(j,b);
                        sigma_I_aI_local(a,i) += r2ab * Fov_hat_a(j,b);

                    }
                }
            }
            #pragma omp critical (Y_bar_b)
            {
                Y_bar_aI += Y_bar_aI_local;
                sigma_I_aI += sigma_I_aI_local;
            }
        } 

        */
                            

template class ri_eomsf_r<double>;
template class ri_eomsf_r<std::complex<double> >;

}

