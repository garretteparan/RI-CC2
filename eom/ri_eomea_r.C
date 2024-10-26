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
#include "ri_eomea_r.h"
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
void ri_eomea_r<double,double>::css_restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQpo_a, Mat<double> &BQvp_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Col<double> &r1, Col<double> &e_orb,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Col<double> &sigma) {
    
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
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQvp_a;
        arma::Mat<double> Fvv_hat1(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

        arma::Mat<double> sigma_0 = Fvv_hat*r1;
        
        //transformed vector
        #pragma omp parallel
        {
            double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {

                sigma(a) = sigma_0(a);

            }
        }
    }
}

template<>
void ri_eomea_r<double,double>::davidson_restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQpo_a, Mat<double> &BQvp_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Col<double> &r1, Col<double> &e_orb,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Col<double> &sigma) {
    
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

        arma::vec sigma_JG (n_vir, fill::zeros);
        arma::vec sigma_I (n_vir, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir, fill::zeros);
        
        /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
        arma::vec iQ (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);

        // Fov_hat
        arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> Fov_hat1 = F11.st();
        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
        arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQvp_a;
        arma::Mat<double> Fvv_hat1(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;


        /// step 4:
        #pragma omp parallel
        {

            arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
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
                double r2ab = 0.0;
                double r2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
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
                                //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                // Yia[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                    }
                }
            }
            #pragma omp critical (Yia)
            {
                Yia += Yia_local;
            }
        }


        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        
        arma::Mat<double> sigma_0 = E_vv*r1;

        arma::Mat<double> BQvp(BQvp_a.memptr(), n_aux*n_vir, n_vir, false, true);
        arma::Mat<double> BQa = BQvp * r1; // (n_aux*n_vir, n_vir)*n_vir
        BQa.reshape(n_aux,n_vir);


        #pragma omp parallel
        {
            arma::vec sigma_I_local (n_vir, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_vir, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;
                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQa[(a*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            r_ijba += BQa[(b*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            // Y_bar_local[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            Y_bar_local[(a*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                        + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        // sigma_I(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        // sigma_I_local(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        sigma_I_local(a) += c_os * r_ijab * Fov_hat(j,b)
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



        /// step 5:
        arma::Mat<double> gamma_G = Y_bar * CvirtA.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb, fill::zeros);
        gamma_P = PQinvhalf * gamma_G; // (n_aux,n_aux)*(n_aux,n_orb)

        
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

        arma::vec JG (n_orb, fill::zeros);  
        #pragma omp parallel
        {
            arma::vec JG_local (n_orb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_local(alpha) += gamma_P[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                                    // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
        }


        /// step 6:
        sigma_JG += Lam_pA.st() * JG; //(orb,virt).t * orb = virt
        
        //transformed vector
        // arma::vec sigma (n_vir, fill::zeros);
        #pragma omp parallel
        {
            double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {

                sigma(a) = sigma_0(a) + sigma_I(a) + sigma_JG(a);

            }
        }
        
    }
}


template<>
void ri_eomea_r<double,double>::diis_restricted_energy(
    double& exci, const size_t& n_occ, const size_t& n_vir,
    const size_t& n_aux, const size_t& n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQpo_a, Mat<double> &BQvp_a, 
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &f_vv, Mat<double> &f_oo,
    Mat<double> &t1, Col<double> &r1, Col<double> &e_orb,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss, Col<double> &sigma) {
    
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

        arma::vec sigma_JG (n_vir, fill::zeros);
        arma::vec sigma_I (n_vir, fill::zeros);
        arma::mat E_vv (n_vir, n_vir, fill::zeros);
        arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
        arma::mat Y_bar (n_aux, n_vir, fill::zeros);
        
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

        // Fvv_hat
        arma::Mat<double> F3 = 2.0 * iQ.st() * BQvp_a;
        arma::Mat<double> Fvv_hat1(F3.memptr(), n_vir, n_vir, false, true);
        arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
        arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;


        /// step 4:
        #pragma omp parallel
        {

            arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
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
                double r2ab = 0.0;
                double r2ba = 0.0;
                
                if(i == j) {
                    const double *w0 = W0.memptr();

                    for(size_t b = 0; b < n_vir; b++) {
                        
                        const double *w0b = w0 + b * n_vir;

                        double dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

                            for(size_t Q = 0; Q < n_aux; Q++) {
                                // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
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
                                //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                // Yia[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                                Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
                                Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
                                                                + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
                            }
                        }
                    }
                }
            }
            #pragma omp critical (Yia)
            {
                Yia += Yia_local;
            }
        }


        arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
        E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        
        arma::Mat<double> sigma_0 = E_vv*r1;

        arma::Mat<double> BQvp(BQvp_a.memptr(), n_aux*n_vir, n_vir, false, true);
        arma::Mat<double> BQa = BQvp * r1; // (n_aux*n_vir, n_vir)*n_vir
        BQa.reshape(n_aux,n_vir);


        #pragma omp parallel
        {
            arma::vec sigma_I_local (n_vir, fill::zeros);
            arma::mat Y_bar_local (n_aux, n_vir, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        
                        //denominator
                        double delta_ijab = e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
                        double r_ijab = 0.0;
                        double r_ijba = 0.0;

                        
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            r_ijab += BQa[(a*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
                            r_ijba += BQa[(b*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+a*n_aux+Q)];

                        }
                        
                        r_ijab = r_ijab / (delta_ijab + exci);
                        r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_bar[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            // Y_bar_local[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            Y_bar_local[(a*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                        + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }
                        
                        // sigma_I
                        // sigma_I(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        // sigma_I_local(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
                        sigma_I_local(a) += c_os * r_ijab * Fov_hat(j,b)
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



        /// step 5:
        arma::Mat<double> gamma_G = Y_bar * CvirtA.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P (n_aux, n_orb, fill::zeros);
        gamma_P = PQinvhalf * gamma_G; // (n_aux,n_aux)*(n_aux,n_orb)


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

        arma::vec JG (n_orb, fill::zeros);
        #pragma omp parallel
        {
            arma::vec JG_local (n_orb, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t beta = 0; beta < n_orb; beta++) {
                    for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
                        JG_local(alpha) += gamma_P[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                                    // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
        }

        /// step 6:
        sigma_JG += Lam_pA.st() * JG; //(orb,virt).t * orb = virt
        
        //transformed vector
        // arma::vec sigma (n_vir, fill::zeros);
        #pragma omp parallel
        {
            double excit_local=0.0;
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {

                sigma(a) = sigma_0(a) + sigma_I(a) + sigma_JG(a);
                excit_local += (sigma(a)*r1(a)) / pow(norm(r1,"fro"),2);

            }
            #pragma omp critical
            { 
                excit += excit_local; 
            }
        }
        
        
        // update of the trial vector
        // residual.zeros();
        arma::vec residual (n_vir, fill::zeros);
        arma::vec update (n_vir, fill::zeros);
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                
                double delta_a = -e_orb[n_occ+a];
                residual(a) = (sigma(a) - (excit*r1(a))) / norm(r1,"fro");
                update(a) = residual(a) / delta_a;
                r1(a) = (r1(a) + update(a)) / norm(r1,"fro");
                
            }
        }

        exci = excit;
    }
}

// template<>
// void ri_eomea_r<double,double>::css_unrestricted_energy(
//     double &exci, const size_t& n_occa, const size_t& n_vira,
//     const size_t& n_occb, const size_t& n_virb,
//     const size_t& n_aux, const size_t& n_orb,
//     Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
//     Mat<TC> &BQoo_a, Mat<TC> &BQpo_a, Mat<TC> &BQvp_a,
//     Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
//     Mat<TC> &BQoo_b, Mat<TC> &BQpo_b, Mat<TC> &BQvp_b,
//     Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
//     Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
//     Mat<TC> &CoccA, Mat<TC> &CvirtA,
//     Mat<TC> &CoccB, Mat<TC> &CvirtB,
//     Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
//     Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
//     Mat<TC> &t1a, Mat<TC> &t1b, 
//     Col<TC> &r1, 
//     Col<TC> &e_orb,
//     array_view<TC> av_buff_ao,
//     array_view<TC> av_pqinvhalf,
//     const libqints::dev_omp &m_dev,
//     const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
//     double c_os, double c_ss, Col<TC> &sigma) {
    
//     memory_pool<double> mem(av_buff_ao);
//     typename memory_pool<double>::checkpoint chkpt = mem.save_state();

//     size_t npairs = (n_occ+1)*n_occ/2;
//     std::vector<size_t> occ_i2(npairs);
//     idx2_list pairs(n_occ, n_occ, npairs,
//         array_view<size_t>(&occ_i2[0], occ_i2.size()));
//     for(size_t i = 0, ij = 0; i < n_occ; i++) {
//     for(size_t j = 0; j <= i; j++, ij++)
//         pairs.set(ij, idx2(i, j));
//     }
    
//     {
//         arma::mat E_vv (n_vir, n_vir, fill::zeros);
        
//         /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
//         arma::vec iQ (n_aux, fill::zeros);
//         iQ += BQov_a * vectorise(t1);

//         // Fvv_hat
//         // arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
//         // arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
//         // arma::Mat<double> Fvv_hat1 = F33.st();
//         arma::Mat<double> F3 = 2.0 * iQ.st() * BQvp_a;
//         arma::Mat<double> Fvv_hat1(F3.memptr(), n_vir, n_vir, false, true);
//         arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
//         arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
//         arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
//         arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;

//         arma::Mat<double> sigma_0 = Fvv_hat*r1;
        
//         //transformed vector
//         #pragma omp parallel
//         {
//             double excit_local=0.0;
//             #pragma omp for
//             for(size_t a = 0; a < n_vir; a++) {

//                 sigma(a) = sigma_0(a);

//             }
//         }
//     }
// }

// template<>
// void ri_eomea_r<double,double>::davidson_unrestricted_energy(
//     double &exci, const size_t& n_occa, const size_t& n_vira,
//     const size_t& n_occb, const size_t& n_virb,
//     const size_t& n_aux, const size_t& n_orb,
//     Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
//     Mat<TC> &BQoo_a, Mat<TC> &BQpo_a, Mat<TC> &BQvp_a,
//     Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
//     Mat<TC> &BQoo_b, Mat<TC> &BQpo_b, Mat<TC> &BQvp_b,
//     Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
//     Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
//     Mat<TC> &CoccA, Mat<TC> &CvirtA,
//     Mat<TC> &CoccB, Mat<TC> &CvirtB,
//     Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
//     Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
//     Mat<TC> &t1a, Mat<TC> &t1b, 
//     Col<TC> &r1, 
//     Col<TC> &e_orb,
//     array_view<TC> av_buff_ao,
//     array_view<TC> av_pqinvhalf,
//     const libqints::dev_omp &m_dev,
//     const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
//     double c_os, double c_ss, Col<TC> &sigma) {
    
//     memory_pool<double> mem(av_buff_ao);
//     typename memory_pool<double>::checkpoint chkpt = mem.save_state();

//     size_t npairs = (n_occ+1)*n_occ/2;
//     std::vector<size_t> occ_i2(npairs);
//     idx2_list pairs(n_occ, n_occ, npairs,
//         array_view<size_t>(&occ_i2[0], occ_i2.size()));
//     for(size_t i = 0, ij = 0; i < n_occ; i++) {
//     for(size_t j = 0; j <= i; j++, ij++)
//         pairs.set(ij, idx2(i, j));
//     }
    
//     {
        
//         double excit=0.0;

//         arma::vec sigma_JG (n_vir, fill::zeros);
//         arma::vec sigma_I (n_vir, fill::zeros);
//         arma::mat E_vv (n_vir, n_vir, fill::zeros);
//         arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
//         arma::mat Y_bar (n_aux, n_vir, fill::zeros);
        
//         /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
//         arma::vec iQ (n_aux, fill::zeros);
//         iQ += BQov_a * vectorise(t1);

//         // Fov_hat
//         arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
//         arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
//         arma::Mat<double> Fov_hat1 = F11.st();
//         arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
//         arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
//         arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
//         arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

//         // Fvv_hat
//         // arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
//         // arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
//         // arma::Mat<double> Fvv_hat1 = F33.st();
//         arma::Mat<double> F3 = 2.0 * iQ.st() * BQvp_a;
//         arma::Mat<double> Fvv_hat1(F3.memptr(), n_vir, n_vir, false, true);
//         arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
//         arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
//         arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;


//         /// step 4:
//         #pragma omp parallel
//         {

//             arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
//             #pragma omp for
//             for(size_t ij = 0; ij < npairs; ij++) {
//                 idx2 i2 = pairs[ij];
//                 size_t i = i2.i, j = i2.j;
                
//                 // for t2: 
//                 arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
//                 arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

//                 // integrals
//                 arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                
//                 double delta_ij = e_orb(i) + e_orb(j);
//                 double t2ab = 0.0;
//                 double t2ba = 0.0;
//                 double r2ab = 0.0;
//                 double r2ba = 0.0;
                
//                 if(i == j) {
//                     const double *w0 = W0.memptr();

//                     for(size_t b = 0; b < n_vir; b++) {
                        
//                         const double *w0b = w0 + b * n_vir;

//                         double dijb = delta_ij - e_orb[n_occ+b];

//                         for(size_t a = 0; a < n_vir; a++) {
//                             t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

//                             for(size_t Q = 0; Q < n_aux; Q++) {
//                                 // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                             }
//                         }
//                     }

//                 } else {
//                     const double *w0 = W0.memptr();

//                     for(size_t b = 0; b < n_vir; b++) {
                        
//                         const double *w0b = w0 + b * n_vir;

//                         double dijb = delta_ij - e_orb[n_occ+b];

//                         for(size_t a = 0; a < n_vir; a++) {
//                             t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
//                             t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

//                             for(size_t Q = 0; Q < n_aux; Q++) {
//                                 //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
//                                 // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
//                                 //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 // Yia[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
//                                 //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
//                                 Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
//                                                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
//                                                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
//                             }
//                         }
//                     }
//                 }
//             }
//             #pragma omp critical (Yia)
//             {
//                 Yia += Yia_local;
//             }
//         }


//         arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
//         E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        
//         arma::Mat<double> sigma_0 = E_vv*r1;

//         arma::Mat<double> BQvp(BQvp_a.memptr(), n_aux*n_vir, n_vir, false, true);
//         arma::Mat<double> BQa = BQvp * r1; // (n_aux*n_vir, n_vir)*n_vir
//         BQa.reshape(n_aux,n_vir);


//         #pragma omp parallel
//         {
//             arma::vec sigma_I_local (n_vir, fill::zeros);
//             arma::mat Y_bar_local (n_aux, n_vir, fill::zeros);
//             #pragma omp for
//             for(size_t a = 0; a < n_vir; a++) {
//                 for(size_t b = 0; b < n_vir; b++) {
//                     for(size_t j = 0; j < n_occ; j++) {
                        
//                         //denominator
//                         double delta_ijab = e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
//                         double r_ijab = 0.0;
//                         double r_ijba = 0.0;
                        
//                         for(size_t Q = 0; Q < n_aux; Q++) {

//                             r_ijab += BQa[(a*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                             r_ijba += BQa[(b*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+a*n_aux+Q)];

//                         }
                        
//                         r_ijab = r_ijab / (delta_ijab + exci);
//                         r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
//                         for(size_t P = 0; P < n_aux; P++) {
//                             // Y_bar[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
//                             // Y_bar_local[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
//                             Y_bar_local[(a*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
//                                                         + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
//                         }
                        
//                         // sigma_I
//                         // sigma_I(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
//                         // sigma_I_local(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
//                         sigma_I_local(a) += c_os * r_ijab * Fov_hat(j,b)
//                                             + c_ss * (r_ijab - r_ijba) * Fov_hat(j,b);
                        
//                     }
//                 }
//             }
//             #pragma omp critical (YI)
//             {
//                 Y_bar += Y_bar_local;
//                 sigma_I += sigma_I_local;
//             }
//         }



//         /// step 5:
//         arma::Mat<double> gamma_G = Y_bar * CvirtA.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)

//         // V_PQ^(-1/2)
//         arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
//         arma::Mat<double> gamma_P (n_aux, n_orb, fill::zeros);
//         gamma_P = PQinvhalf * gamma_G; // (n_aux,n_aux)*(n_aux,n_orb)

        
//         // (P|ab)
//         arma::Mat<double> Unit(n_orb, n_orb, fill::eye);
//         std::vector<size_t> vblst(1);
//         idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
//         blst.populate();
//         op_coulomb op;
//         {
//             motran_2e3c_incore_result_container<double> buf(av_buff_ao);
//             scr_null<bat_2e3c_shellpair_cgto<double>> scr;
//             motran_2e3c<double, double> mot(op, m_b3, scr, m_dev);
//             mot.set_trn(Unit);
//             mot.run(m_dev, blst, buf);
//         }
//         arma::Mat<double> V_Pab(arrays<double>::ptr(av_buff_ao), Unit.n_cols * Unit.n_cols, n_aux, false, true);
//         mem.load_state(chkpt);

//         arma::vec JG (n_orb, fill::zeros);  
//         #pragma omp parallel
//         {
//             arma::vec JG_local (n_orb, fill::zeros);
//             #pragma omp for
//             for(size_t P = 0; P < n_aux; P++) {
//                 for(size_t beta = 0; beta < n_orb; beta++) {
//                     for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
//                         JG_local(alpha) += gamma_P[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
//                                     // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
//                     }
//                 }
//             }
//             #pragma omp critical (JG)
//             {
//                 JG += JG_local;
//             }
//         }

// /*
//         // GPP: this is the digestor that replaces the formation of JG
//         arma::mat JG (n_orb, 1, fill::zeros);
//         {

//             //  Step 1: Read libqints-type basis set from files and form shellpair basis.
//             libqints::basis_1e2c_shellpair_cgto<double> bsp;
//             size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
//             size_t nsp = bsp.get_nsp();    //  # of munu shell pairs

//             //  Step 2: Construct the 2e3c shellpair basis and corresponding full basis range
//             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
//             libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
//             libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

//             //  Step 3: prepare required input settings
//             libqints::dev_omp dev;                  //  libqints-type device information.
//             size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
//             dev.init(1024);
//             dev.nthreads = 1;
//             dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
//             libqints::deriv_code dc;
//             dc.set(0);                //  Set integral derivative level
//             libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
//             libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
//             qjob.begin(fbr);                                //  Start the libqints job for full basis range

//             //  Compute (\mu\nu|Q) * L(\nu|Q) = V(\mu)
//             // size_t ni = n_occ;
//             size_t ni = 1;
//             arma::mat L(n_aux, n_orb * ni, arma::fill::randn);

//             //  Step 4: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
//             scr_2e3c scr(m_b3);

//             //  Step 5:
//             //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
//             size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
//             size_t nmunu = 0, nP = 0;

//             #pragma omp for 
//             for(size_t i = 0; i < frbra.distance(); i++) 
//                 nmunu += frbra[i].get_num_comp();

//             #pragma omp for 
//             for(size_t i = 0; i < frket.distance(); i++) 
//                 nP += frket[i].get_num_comp();
//             {
//                 nbsp_per_subrange = nmunu;
//                 naux_per_subrange = nP;
//             }

//             size_t min_nbsp_per_subrange = 0;
//             #pragma omp for 
//             for (size_t isp = 0; isp < nsp; isp++) {
//                 size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
//                 min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
//             }
//             nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity

//             //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
//             // arma::vec Fvec(nbsp);
//             JG.zeros();  //  Result will be accumulated in the output arrays, so we need to zero out them
//             dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

//             //  Step 7: Loop over basis subranges and run libqints job
//             libqints::batching_info<2> binfo;
//             libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
//             libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
//             for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
//                 //  Current basis subrange
//                 libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
//                     fbr, binfo.get_batch_window(biter.get_batch_number()));

//                 if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
//                     std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
//                     qjob.end();  //  End the libqints job before return
//                     throw std::runtime_error("motran_2e3c: qints failure");
//                 }
//             }

//             libaview::array_view<double> av_result(JG.memptr(), JG.n_elem);

//         }
// */

//         /// step 6:
//         sigma_JG += Lam_pA.st() * JG; //(orb,virt).t * orb = virt
        
//         //transformed vector
//         // arma::vec sigma (n_vir, fill::zeros);
//         #pragma omp parallel
//         {
//             double excit_local=0.0;
//             #pragma omp for
//             for(size_t a = 0; a < n_vir; a++) {

//                 sigma(a) = sigma_0(a) + sigma_I(a) + sigma_JG(a);

//             }
//         }
        
//     }
// }


// template<>
// void ri_eomea_r<double,double>::diis_unrestricted_energy(
//     double &exci, const size_t& n_occa, const size_t& n_vira,
//     const size_t& n_occb, const size_t& n_virb,
//     const size_t& n_aux, const size_t& n_orb,
//     Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
//     Mat<TC> &BQoo_a, Mat<TC> &BQpo_a, Mat<TC> &BQvp_a,
//     Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
//     Mat<TC> &BQoo_b, Mat<TC> &BQpo_b, Mat<TC> &BQvp_b,
//     Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
//     Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
//     Mat<TC> &CoccA, Mat<TC> &CvirtA,
//     Mat<TC> &CoccB, Mat<TC> &CvirtB,
//     Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
//     Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
//     Mat<TC> &t1a, Mat<TC> &t1b, 
//     Col<TC> &r1, 
//     Col<TC> &e_orb,
//     array_view<TC> av_buff_ao,
//     array_view<TC> av_pqinvhalf,
//     const libqints::dev_omp &m_dev,
//     const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
//     double c_os, double c_ss, Col<TC> &sigma) {
    
//     memory_pool<double> mem(av_buff_ao);
//     typename memory_pool<double>::checkpoint chkpt = mem.save_state();

//     size_t npairs = (n_occ+1)*n_occ/2;
//     std::vector<size_t> occ_i2(npairs);
//     idx2_list pairs(n_occ, n_occ, npairs,
//         array_view<size_t>(&occ_i2[0], occ_i2.size()));
//     for(size_t i = 0, ij = 0; i < n_occ; i++) {
//     for(size_t j = 0; j <= i; j++, ij++)
//         pairs.set(ij, idx2(i, j));
//     }
    
//     {
        
//         double excit=0.0;

//         arma::vec sigma_JG (n_vir, fill::zeros);
//         arma::vec sigma_I (n_vir, fill::zeros);
//         arma::mat E_vv (n_vir, n_vir, fill::zeros);
//         arma::mat Yia (n_aux, n_vir*n_occ, fill::zeros);
//         arma::mat Y_bar (n_aux, n_vir, fill::zeros);
        
//         /// step 3: form iQ, iQ_bar, F_ia, F_ab, F_ij
//         arma::vec iQ (n_aux, fill::zeros);
//         // iQ += BQov_a * t1;
//         iQ += BQov_a * vectorise(t1);

//         // Fov_hat
//         arma::Mat<double> F1 = 2.0 * iQ.st() * BQov_a;
//         arma::Mat<double> F11(F1.memptr(), n_vir, n_occ, false, true);
//         arma::Mat<double> Fov_hat1 = F11.st();
//         arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
//         arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);
//         arma::Mat<double> Fov_hat2 = BQoo.st() * BQvo;
//         arma::Mat<double> Fov_hat = Fov_hat1 - Fov_hat2;

//         // Fvv_hat
//         // arma::Mat<double> F3 = 2.0 * iQ.st() * BQpv_a;
//         // arma::Mat<double> F33(F3.memptr(), n_vir, n_vir, false, true);
//         // arma::Mat<double> Fvv_hat1 = F33.st();
//         arma::Mat<double> F3 = 2.0 * iQ.st() * BQvp_a;
//         arma::Mat<double> Fvv_hat1(F3.memptr(), n_vir, n_vir, false, true);
//         arma::Mat<double> BQpo(BQpo_a.memptr(), n_aux*n_occ, n_vir, false, true);
//         arma::Mat<double> Fvv_hat2 = BQpo.st() * BQvo;
//         arma::Mat<double> Fvv_hat = f_vv + Fvv_hat1 - Fvv_hat2;


//         /// step 4:
//         #pragma omp parallel
//         {

//             arma::mat Yia_local (n_aux, n_vir*n_occ, fill::zeros);
//             #pragma omp for
//             for(size_t ij = 0; ij < npairs; ij++) {
//                 idx2 i2 = pairs[ij];
//                 size_t i = i2.i, j = i2.j;
                
//                 // for t2: 
//                 arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
//                 arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

//                 // integrals
//                 arma::Mat<double> W0 = Bhp_i.st() * Bhp_j; // t2:   aibj
                
//                 double delta_ij = e_orb(i) + e_orb(j);
//                 double t2ab = 0.0;
//                 double t2ba = 0.0;
//                 double r2ab = 0.0;
//                 double r2ba = 0.0;
                
//                 if(i == j) {
//                     const double *w0 = W0.memptr();

//                     for(size_t b = 0; b < n_vir; b++) {
                        
//                         const double *w0b = w0 + b * n_vir;

//                         double dijb = delta_ij - e_orb[n_occ+b];

//                         for(size_t a = 0; a < n_vir; a++) {
//                             t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);

//                             for(size_t Q = 0; Q < n_aux; Q++) {
//                                 // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                             }
//                         }
//                     }

//                 } else {
//                     const double *w0 = W0.memptr();

//                     for(size_t b = 0; b < n_vir; b++) {
                        
//                         const double *w0b = w0 + b * n_vir;

//                         double dijb = delta_ij - e_orb[n_occ+b];

//                         for(size_t a = 0; a < n_vir; a++) {
//                             t2ab = w0b[a] / (dijb - e_orb[n_occ+a]);
//                             t2ba = w0[a*n_vir+b] / (dijb - e_orb[n_occ+a]);

//                             for(size_t Q = 0; Q < n_aux; Q++) {
//                                 //Yia[(a*n_occ*n_aux+i*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 //Yia[(b*n_occ*n_aux+j*n_aux+Q)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
//                                 // Yia[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
//                                 //                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 // Yia[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
//                                 //                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
//                                 Yia_local[(a*n_occ*n_aux+i*n_aux+Q)] += c_os * (t2ab) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)]
//                                                                 + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                                 Yia_local[(b*n_occ*n_aux+j*n_aux+Q)] += c_os * (t2ab) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)]
//                                                                 + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];
//                             }
//                         }
//                     }
//                 }
//             }
//             #pragma omp critical (Yia)
//             {
//                 Yia += Yia_local;
//             }
//         }


//         arma::Mat<double> YQia(Yia.memptr(), n_aux*n_occ, n_vir, false, true);
//         E_vv = Fvv_hat - YQia.st() * BQvo; // E_ab
        
//         arma::Mat<double> sigma_0 = E_vv*r1;

//         arma::Mat<double> BQvp(BQvp_a.memptr(), n_aux*n_vir, n_vir, false, true);
//         arma::Mat<double> BQa = BQvp * r1; // (n_aux*n_vir, n_vir)*n_vir
//         BQa.reshape(n_aux,n_vir);


//         #pragma omp parallel
//         {
//             arma::vec sigma_I_local (n_vir, fill::zeros);
//             arma::mat Y_bar_local (n_aux, n_vir, fill::zeros);
//             #pragma omp for
//             for(size_t a = 0; a < n_vir; a++) {
//                 for(size_t b = 0; b < n_vir; b++) {
//                     for(size_t j = 0; j < n_occ; j++) {
                        
//                         //denominator
//                         double delta_ijab = e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
//                         double r_ijab = 0.0;
//                         double r_ijba = 0.0;

                        
//                         for(size_t Q = 0; Q < n_aux; Q++) {

//                             r_ijab += BQa[(a*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+b*n_aux+Q)];
//                             r_ijba += BQa[(b*n_aux+Q)]*BQhp_a[(j*n_vir*n_aux+a*n_aux+Q)];

//                         }
                        
//                         r_ijab = r_ijab / (delta_ijab + exci);
//                         r_ijba = r_ijba / (delta_ijab + exci);
                        
                        
//                         for(size_t P = 0; P < n_aux; P++) {
//                             // Y_bar[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
//                             // Y_bar_local[(a*n_aux+P)] += (2.0 * r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
//                             Y_bar_local[(a*n_aux+P)] += c_os * r_ijab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
//                                                         + c_ss * (r_ijab - r_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
//                         }
                        
//                         // sigma_I
//                         // sigma_I(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
//                         // sigma_I_local(a) += ((2.0 * r_ijab - r_ijba) * Fov_hat(j,b));
//                         sigma_I_local(a) += c_os * r_ijab * Fov_hat(j,b)
//                                             + c_ss * (r_ijab - r_ijba) * Fov_hat(j,b);
                        
//                     }
//                 }
//             }
//             #pragma omp critical (YI)
//             {
//                 Y_bar += Y_bar_local;
//                 sigma_I += sigma_I_local;
//             }
//         }



//         /// step 5:
//         arma::Mat<double> gamma_G = Y_bar * CvirtA.st(); // (n_aux, n_vir)*(orb,vir).t = (n_aux,n_orb)

//         // V_PQ^(-1/2)
//         arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
//         arma::Mat<double> gamma_P (n_aux, n_orb, fill::zeros);
//         gamma_P = PQinvhalf * gamma_G; // (n_aux,n_aux)*(n_aux,n_orb)


//         // (P|ab)
//         arma::Mat<double> Unit(n_orb, n_orb, fill::eye);
//         std::vector<size_t> vblst(1);
//         idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
//         blst.populate();
//         op_coulomb op;
//         {
//             motran_2e3c_incore_result_container<double> buf(av_buff_ao);
//             scr_null<bat_2e3c_shellpair_cgto<double>> scr;
//             motran_2e3c<double, double> mot(op, m_b3, scr, m_dev);
//             mot.set_trn(Unit);
//             mot.run(m_dev, blst, buf);
//         }
//         arma::Mat<double> V_Pab(arrays<double>::ptr(av_buff_ao), Unit.n_cols * Unit.n_cols, n_aux, false, true);
//         mem.load_state(chkpt);

//         arma::vec JG (n_orb, fill::zeros);
//         #pragma omp parallel
//         {
//             arma::vec JG_local (n_orb, fill::zeros);
//             #pragma omp for
//             for(size_t P = 0; P < n_aux; P++) {
//                 for(size_t beta = 0; beta < n_orb; beta++) {
//                     for(size_t alpha = 0; alpha < n_orb; alpha++) {
                        
//                         JG_local(alpha) += gamma_P[(beta*n_aux+P)] * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
//                                     // (n_aux,n_orb)*(orb*orb,aux) = orb
                        
//                     }
//                 }
//             }
//             #pragma omp critical (JG)
//             {
//                 JG += JG_local;
//             }
//         }

// /*
//         // GPP: this is the digestor that replaces the formation of JG
//         arma::mat JG (n_orb, 1, fill::zeros);
//         {

//             //  Step 1: Read libqints-type basis set from files and form shellpair basis.
//             libqints::basis_1e2c_shellpair_cgto<double> bsp;
//             size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
//             size_t nsp = bsp.get_nsp();    //  # of munu shell pairs

//             //  Step 2: Construct the 2e3c shellpair basis and corresponding full basis range
//             libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
//             libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
//             libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);

//             //  Step 3: prepare required input settings
//             libqints::dev_omp dev;                  //  libqints-type device information.
//             size_t mem_total = 32 * 1024UL * 1024;  //  given total memory (Bytes) available
//             dev.init(1024);
//             dev.nthreads = 1;
//             dev.memory = mem_total / dev.nthreads;  //  memory in dev is memory per thread
//             libqints::deriv_code dc;
//             dc.set(0);                //  Set integral derivative level
//             libqints::op_coulomb op;  //  Use Coulomb operator as an example, you may use range-separated or other operator
//             libqints::qints_job qjob(op, m_b3, dc, dev);  //  Construct the libqints job
//             qjob.begin(fbr);                                //  Start the libqints job for full basis range

//             //  Compute (\mu\nu|Q) * L(\nu|Q) = V(\mu)
//             // size_t ni = n_occ;
//             size_t ni = 1;
//             arma::mat L(n_aux, n_orb * ni, arma::fill::randn);

//             //  Step 4: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
//             scr_2e3c scr(m_b3);

//             //  Step 5:
//             //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
//             size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
//             size_t nmunu = 0, nP = 0;

//             #pragma omp for 
//             for(size_t i = 0; i < frbra.distance(); i++) 
//                 nmunu += frbra[i].get_num_comp();

//             #pragma omp for 
//             for(size_t i = 0; i < frket.distance(); i++) 
//                 nP += frket[i].get_num_comp();
//             {
//                 nbsp_per_subrange = nmunu;
//                 naux_per_subrange = nP;
//             }

//             size_t min_nbsp_per_subrange = 0;
//             #pragma omp for 
//             for (size_t isp = 0; isp < nsp; isp++) {
//                 size_t nbsp_isp = bsp[isp].get_num_comp();  //  # of munu basis function pairs of this shell pair
//                 min_nbsp_per_subrange = std::max(nbsp_isp, min_nbsp_per_subrange);
//             }
//             nbsp_per_subrange = min_nbsp_per_subrange;  //  Use minimum subrange for simplicity

//             //  Step 6: Set up 2e3c integral digestor, which is used for digesting evaluated integrals
//             // arma::vec Fvec(nbsp);
//             JG.zeros();  //  Result will be accumulated in the output arrays, so we need to zero out them
//             dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);

//             //  Step 7: Loop over basis subranges and run libqints job
//             libqints::batching_info<2> binfo;
//             libqints::batching_cgto_size(nbsp_per_subrange).apply(frbra, binfo);
//             libqints::batching_cgto_size(naux_per_subrange).apply(frket, binfo);
//             for (libqints::batiter_colmaj<2> biter(binfo); !biter.end(); biter.next()) {
//                 //  Current basis subrange
//                 libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> r_bat(
//                     fbr, binfo.get_batch_window(biter.get_batch_number()));

//                 if (libqints::qints(qjob, r_bat, scr, dig, dev) != 0) {
//                     std::cout << " Failed to compute or digest 2e3c integrals" << std::endl;
//                     qjob.end();  //  End the libqints job before return
//                     throw std::runtime_error("motran_2e3c: qints failure");
//                 }
//             }

//             libaview::array_view<double> av_result(JG.memptr(), JG.n_elem);

//         }
// */

//         /// step 6:
//         sigma_JG += Lam_pA.st() * JG; //(orb,virt).t * orb = virt
        
//         //transformed vector
//         // arma::vec sigma (n_vir, fill::zeros);
//         #pragma omp parallel
//         {
//             double excit_local=0.0;
//             #pragma omp for
//             for(size_t a = 0; a < n_vir; a++) {

//                 sigma(a) = sigma_0(a) + sigma_I(a) + sigma_JG(a);
//                 excit_local += (sigma(a)*r1(a)) / pow(norm(r1,"fro"),2);

//             }
//             #pragma omp critical
//             { 
//                 excit += excit_local; 
//             }
//         }
        
        
//         // update of the trial vector
//         // residual.zeros();
//         arma::vec residual (n_vir, fill::zeros);
//         arma::vec update (n_vir, fill::zeros);
//         #pragma omp parallel
//         {
//             #pragma omp for
//             for(size_t a = 0; a < n_vir; a++) {
                
//                 double delta_a = -e_orb[n_occ+a];
//                 residual(a) = (sigma(a) - (excit*r1(a))) / norm(r1,"fro");
//                 update(a) = residual(a) / delta_a;
//                 r1(a) = (r1(a) + update(a)) / norm(r1,"fro");
                
//             }
//         }

//         exci = excit;
//     }
// }

template class ri_eomea_r<double, double>;
template class ri_eomea_r<complex<double>, double>;
template class ri_eomea_r<complex<double>, complex<double>>;

}
