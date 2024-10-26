#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <chrono> //timings
#include "ricc2.h"
#include <armadillo>
#include <libposthf/motran/motran_2e3c.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/arrays/memory_pool.h>
#include <libqints/batch/bat_2e3c_shellpair_cgto.h>
#include <libqints/digestors/dig_passthru_2e3c_shellpair_cgto.h> // for (\mu\nu|Q)
#include <libqints/algorithms/gto/gto_pack.h>
#include <libgmbpt/util/dig_2e3c.h>
#include <libgmbpt/util/scr_2e3c.h>

#include<complex>
using namespace std;
using namespace std::chrono;

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

/// GPP: RI-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: this is the optimized code
template<>
void ricc2<double,double>::restricted_energy (
    double &Eos, double &Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQvo_a, Mat<double> &BQov_a,
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQoh_a, Mat<double> &V_Pab,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &t1, Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss) {

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
    array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }

    // intermediates
    arma::Mat<double> omega_I (n_vir, n_occ, fill::zeros);
    arma::Mat<double> omega_JG (n_vir, n_occ, fill::zeros);
    arma::Mat<double> omega_H (n_vir, n_occ, fill::zeros);
    arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
    arma::Mat<double> Y (n_aux, n_vir*n_occ, fill::zeros); //novx
    double eos = 0.0, ess = 0.0;

    {

    	/// step 3: form i^Q and F_ia
        arma::vec iQ (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);

        arma::Mat<double> F11 = 2.0 * iQ.st() * BQov_a; //ov
        arma::Mat<double> F111(F11.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> F1 = F111.st();

        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);

        arma::Mat<double> F2 = BQoo.st() * BQvo;
        arma::Mat<double> F_hat = F1 - F2; //ov

        #pragma omp parallel
        {
	        double Eost = 0.0, Esst = 0.0; // local variable
            arma::Mat<double> omega_I_local (n_vir, n_occ, fill::zeros);
            arma::Mat<double> Y_local (n_aux, n_vir*n_occ, fill::zeros);
            #pragma omp for schedule(dynamic)
            for(size_t ij = 0; ij < npairs; ij++) {
	            idx2 i2 = pairs[ij];
	            size_t i = i2.i, j = i2.j;

	            // energy: vovo
	            arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vir), n_aux, n_vir, false, true);
	            arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vir), n_aux, n_vir, false, true);

	            // t2: VOVO
	            arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
	            arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

	            // integrals
	            arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
	            arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

	            double delta_ij = e_orb(i)+e_orb(j);
	            double t2ab = 0.0, t2ba = 0.0;

	            // Main loop
	            if(i == j) {
	                const double *w0 = W0.memptr();
	                const double *w1 = W1.memptr();

	                for(size_t b = 0; b < n_vir; b++) {

	                    const double *w0b = w0 + b * n_vir;
	                    const double *w1b = w1 + b * n_vir;
	                    double dijb = delta_ij - e_orb[n_occ+b];

	                    for(size_t a = 0; a < n_vir; a++) {
	                        t2ab = w1b[a] / (dijb - e_orb[n_occ+a]);
	                        // t2ba = w1[a*n_vir+b] / (dijb - e_orb[n_occ+a]);
	                        Eost += w0b[a] * (t1(a,i)*t1(b,j) + c_os*t2ab);
	                        Esst += (w0b[a] - w0[a*n_vir+b]) * (t1(a,i)*t1(b,j) + c_ss*t2ab);

				            for(size_t P = 0; P < n_aux; P++) {
	                            Y_local[(a*n_occ*n_aux+i*n_aux+P)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
	                        }

	                        omega_I_local(a,i) += c_os * t2ab * F_hat(j,b); // working

	                    }
	                }
	            }
                else {
	                const double *w0 = W0.memptr();
	                const double *w1 = W1.memptr();

	                for(size_t b = 0; b < n_vir; b++) {

	                    const double *w0b = w0 + b * n_vir;
	                    const double *w1b = w1 + b * n_vir;
	                    double dijb = delta_ij - e_orb[n_occ+b];

	                    for(size_t a = 0; a < n_vir; a++) {
	                        t2ab = w1b[a] / (dijb - e_orb[n_occ+a]);
	                        t2ba = w1[a*n_vir+b] / (dijb - e_orb[n_occ+a]);
	                        Eost += 2.0 * w0b[a] * (t1(a,i)*t1(b,j) + c_os*t2ab);
	                        Esst += (w0b[a] - w0[a*n_vir+b]) * (t1(a,i)*t1(b,j) + c_ss*t2ab);

	                        for(size_t P = 0; P < n_aux; P++) {
	                            // Y_local[(a*n_occ*n_aux+i*n_aux+P)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
	                            // Y_local[(b*n_occ*n_aux+j*n_aux+P)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+P)];
	                            Y_local[(a*n_occ*n_aux+i*n_aux+P)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                                    + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
	                            Y_local[(b*n_occ*n_aux+j*n_aux+P)] += c_os * t2ab * BQov_a[(i*n_vir*n_aux+a*n_aux+P)]
                                                                    + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+P)];
	                        }

	                        // omega_I_local(a,i) += (2.0*t2ab-t2ba) * F_hat(j,b);
	                        // omega_I_local(b,j) += (2.0*t2ab-t2ba) * F_hat(i,a);
	                        omega_I_local(a,i) += c_os * t2ab * F_hat(j,b)
                                                + c_ss * (t2ab-t2ba) * F_hat(j,b); 
	                        omega_I_local(b,j) += c_os * t2ab * F_hat(i,a)
                                                + c_ss * (t2ab-t2ba) * F_hat(i,a);

	                    }
	                }
	            }
	        }
  	        // eliminate the race condition by declaring a critical section
            #pragma omp critical (Y)
            {
	            eos += Eost;
	            ess += Esst;
                Y += Y_local;
	            omega_I += omega_I_local;
            }
        } // end parallel (1)

        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia(Y.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G11 = YQia * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G1 = gamma_G11.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_G2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_G22 = BQoh * (CvirtA * t1).st(); //(n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_G2 = gamma_G22.submat( 0, 0, n_aux-1, n_orb-1 );

        for(size_t i = 1; i < n_occ; i++) {
            gamma_G1.insert_cols(i*n_orb, gamma_G11.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G2.insert_cols(i*n_orb, gamma_G22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J0 = 2.0 * iQ * vectorise(Lam_hA).st(); //(n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_J(gamma_J0.memptr(), n_aux, n_orb*n_occ, false, true);

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G1 - gamma_G2 + gamma_J; //(n_aux*n_occ,n_orb)

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P = PQinvhalf * gamma_Q; //(n_aux*n_occ,n_orb)

        
        #pragma omp parallel
        {
            arma::Mat<double> JG_local (n_orb, n_occ, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occ; i++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)] //(occ,aux*orb) orb
                                                    * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)]; //(aux*orb,orb) occ

                        }
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
        } // end parallel(2)
        
        /// step 6:
        omega_JG += Lam_pA.st() * JG;

        #pragma omp parallel
        {
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occ; k++) {
                            omega_H(a,i) += Y[(a*n_occ*n_aux+k*n_aux+P)]
                                                    * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                        }
                    }

                    double delta_ia = e_orb(i) - e_orb[n_occ+a];
                    t1(a,i) = (omega_JG(a,i) - omega_H(a,i) + omega_I(a,i)) / delta_ia;

                }
            }
        } // end parallel(3)

    }

    Eos = eos;
    Ess = 2.0 * ess;

}

template<>
void ricc2<double,double>::restricted_energy_digestor (
    double &Eos, double &Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQvo_a, Mat<double> &BQov_a,
    Mat<double> &BQhp_a, Mat<double> &BQoo_a, 
    Mat<double> &BQoh_a,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &t1, Col<double> &e_orb,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss) {

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
    array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }

    // intermediates
    arma::Mat<double> omega_I (n_vir, n_occ, fill::zeros);
    arma::Mat<double> omega_JG (n_vir, n_occ, fill::zeros);
    arma::Mat<double> omega_H (n_vir, n_occ, fill::zeros);
    arma::Mat<double> JG (n_orb, n_occ, fill::zeros);
    arma::Mat<double> Y (n_aux, n_vir*n_occ, fill::zeros);
    double eos = 0.0, ess = 0.0;

    {

    	/// step 3: form i^Q and F_ia
        arma::vec iQ (n_aux, fill::zeros);
        iQ += BQov_a * vectorise(t1);

        arma::Mat<double> F11 = 2.0 * iQ.st() * BQov_a;
        arma::Mat<double> F111(F11.memptr(), n_vir, n_occ, false, true);
        arma::Mat<double> F1 = F111.st();

        arma::Mat<double> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);

        arma::Mat<double> F2 = BQoo.st() * BQvo;
        arma::Mat<double> F_hat = F1 - F2;

        #pragma omp parallel
        {
	        double Eost = 0.0, Esst = 0.0; // local variable
            arma::Mat<double> omega_I_local (n_vir, n_occ, fill::zeros);
            arma::Mat<double> Y_local (n_aux, n_vir*n_occ, fill::zeros);
            #pragma omp for schedule(dynamic)
            for(size_t ij = 0; ij < npairs; ij++) {
	            idx2 i2 = pairs[ij];
	            size_t i = i2.i, j = i2.j;

	            // energy: vovo
	            arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vir), n_aux, n_vir, false, true);
	            arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vir), n_aux, n_vir, false, true);

	            // t2: VOVO
	            arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
	            arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

	            // integrals
	            arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
	            arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

	            double delta_ij = e_orb(i)+e_orb(j);
	            double t2ab = 0.0, t2ba = 0.0;

	            // Main loop
	            if(i == j) {
	                const double *w0 = W0.memptr();
	                const double *w1 = W1.memptr();

	                for(size_t b = 0; b < n_vir; b++) {

	                    const double *w0b = w0 + b * n_vir;
	                    const double *w1b = w1 + b * n_vir;
	                    double dijb = delta_ij - e_orb[n_occ+b];

	                    for(size_t a = 0; a < n_vir; a++) {
	                        t2ab = w1b[a] / (dijb - e_orb[n_occ+a]);
	                        // t2ba = w1[a*n_vir+b] / (dijb - e_orb[n_occ+a]);
	                        Eost += w0b[a] * (t1(a,i)*t1(b,j) + c_os*t2ab);
	                        Esst += (w0b[a] - w0[a*n_vir+b]) * (t1(a,i)*t1(b,j) + c_ss*t2ab);

				            for(size_t P = 0; P < n_aux; P++) {
	                            Y_local[(a*n_occ*n_aux+i*n_aux+P)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
	                        }

	                        omega_I_local(a,i) += c_os * t2ab * F_hat(j,b); // working

	                    }
	                }
	            }
                else {
	                const double *w0 = W0.memptr();
	                const double *w1 = W1.memptr();

	                for(size_t b = 0; b < n_vir; b++) {

	                    const double *w0b = w0 + b * n_vir;
	                    const double *w1b = w1 + b * n_vir;
	                    double dijb = delta_ij - e_orb[n_occ+b];

	                    for(size_t a = 0; a < n_vir; a++) {
	                        t2ab = w1b[a] / (dijb - e_orb[n_occ+a]);
	                        t2ba = w1[a*n_vir+b] / (dijb - e_orb[n_occ+a]);
	                        Eost += 2.0 * w0b[a] * (t1(a,i)*t1(b,j) + c_os*t2ab);
	                        Esst += (w0b[a] - w0[a*n_vir+b]) * (t1(a,i)*t1(b,j) + c_ss*t2ab);

	                        for(size_t P = 0; P < n_aux; P++) {
	                            // Y_local[(a*n_occ*n_aux+i*n_aux+P)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
	                            // Y_local[(b*n_occ*n_aux+j*n_aux+P)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+P)];
	                            Y_local[(a*n_occ*n_aux+i*n_aux+P)] += c_os * t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)]
                                                                    + c_ss * (t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
	                            Y_local[(b*n_occ*n_aux+j*n_aux+P)] += c_os * t2ab * BQov_a[(i*n_vir*n_aux+a*n_aux+P)]
                                                                    + c_ss * (t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+P)];
	                        }

	                        // omega_I_local(a,i) += (2.0*t2ab-t2ba) * F_hat(j,b);
	                        // omega_I_local(b,j) += (2.0*t2ab-t2ba) * F_hat(i,a);
	                        omega_I_local(a,i) += c_os * t2ab * F_hat(j,b)
                                                + c_ss * (t2ab-t2ba) * F_hat(j,b); 
	                        omega_I_local(b,j) += c_os * t2ab * F_hat(i,a)
                                                + c_ss * (t2ab-t2ba) * F_hat(i,a);

	                    }
	                }
	            }
	        }
  	        // eliminate the race condition by declaring a critical section
            #pragma omp critical (Y)
            {
	            eos += Eost;
	            ess += Esst;
                Y += Y_local;
	            omega_I += omega_I_local;
            }
        } // end parallel (1)

        /// step 5:
        // omega_G1: first term of Γ(P,iβ)
        arma::Mat<double> YQia(Y.memptr(), n_aux*n_occ, n_vir, false, true);
        arma::Mat<double> gamma_G11 = YQia * CvirtA.st(); // (n_aux*n_occ,n_orb)
        arma::Mat<double> gamma_G1 = gamma_G11.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_G2: third term of Γ(P,iβ)
        arma::Mat<double> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        arma::Mat<double> gamma_G22 = BQoh * (CvirtA * t1).st(); // (n_aux*n_occ, n_orb)
        arma::Mat<double> gamma_G2 = gamma_G22.submat( 0, 0, n_aux-1, n_orb-1 );

        for(size_t i = 1; i < n_occ; i++) {
            gamma_G1.insert_cols(i*n_orb, gamma_G11.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G2.insert_cols(i*n_orb, gamma_G22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J: second term of Γ(P,iβ)
        arma::Mat<double> gamma_J0 = 2.0 * iQ * vectorise(Lam_hA).st();
        arma::Mat<double> gamma_J(gamma_J0.memptr(), n_aux, n_orb*n_occ, false, true);

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        arma::Mat<double> gamma_Q = gamma_G1 - gamma_G2 + gamma_J;

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P = PQinvhalf * gamma_Q;

        // GPP: this is the digestor that replaces the formation of JG
        {

            //  Step 1: Read libqints-type basis set from files and form shellpair basis.
            // libqints::basis_1e2c_shellpair_cgto<double> bsp;
            // libqints::basis_1e1c_cgto<double> b1x;  //  1e1c auxiliary basis
            const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
            const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
            size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
            size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
            size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
            
            //  Step 2: Construct the 2e3c shellpair basis and corresponding full basis range
            libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
            libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);
            
            //  Step 3: prepare required input settings
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
            
            //  Step 4: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
            scr_2e3c scr(m_b3);
            
            //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
            dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
            if (dev.memory * dev.nthreads > mem_total) {
                std::cout << " Given memory is not enough for computing integrals." << std::endl;
                qjob.end();  //  End the libqints job before return
                return;
            }
            
            size_t ni = n_occ;
            arma::mat L(n_aux, n_orb * ni, arma::fill::randn);
            size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity
            
            //  Step 5:
            //  Memory available for thread-local result arrays:
            size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
            //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
            size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
            // size_t nmunu = 0, nP = 0;
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
            JG.zeros();  
            dig_2e3c<double> dig(m_b3, ni, gamma_P, JG);
            
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
        }

        /// step 6:
        omega_JG += Lam_pA.st() * JG;

        #pragma omp parallel
        {
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occ; k++) {
                            omega_H(a,i) += Y[(a*n_occ*n_aux+k*n_aux+P)]
                                                    * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                        }
                    }

                    double delta_ia = e_orb(i) - e_orb[n_occ+a];
                    t1(a,i) = (omega_JG(a,i) - omega_H(a,i) + omega_I(a,i)) / delta_ia;

                }
            }
        } // end parallel(3)

    }

    Eos = eos;
    Ess = 2.0 * ess;

}


template<typename TC, typename TI>
void ricc2<TC, TI>::restricted_energy(
    complex<double>& Eos, complex<double>& Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<complex<double>> &BQvo_a,
    Mat<complex<double>> &BQov_a,
    Mat<complex<double>> &BQhp_a,
    Mat<complex<double>> &BQoo_a,
    Mat<complex<double>> &BQoh_a,
    Mat<complex<double>> &V_Pab,
    Mat<complex<double>> &Lam_hA,
    Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &CoccA,
    Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &t1,
    Col<complex<double>> &e_orb,
    array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
    double c_os, double c_ss) {


    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }

    // intermediates
    arma::Mat<complex<double>> omega_I (n_vir, n_occ, fill::zeros);
    arma::Mat<complex<double>> gamma_P (n_aux, n_orb*n_occ, fill::zeros);
    arma::Mat<complex<double>> omega_JG (n_vir, n_occ, fill::zeros);
    arma::Mat<complex<double>> gamma_Q (n_aux, n_orb*n_occ, fill::zeros);
    arma::Mat<complex<double>> omega_H (n_vir, n_occ, fill::zeros);
    arma::Mat<complex<double>> Y (n_aux, n_vir*n_occ, fill::zeros);
    arma::Mat<complex<double>> JG (n_orb, n_occ, fill::zeros);
    complex<double> eos = 0.0, ess = 0.0;

    {

    	/// step 3: form i^Q and F_ia

	    arma::cx_vec iQ (n_aux, fill::zeros);
        iQ += BQov_a * arma::vectorise(t1);

        Mat<complex<double>> F11 = 2.0 * iQ.st() * BQov_a;
        Mat<complex<double>> F111(F11.memptr(), n_vir, n_occ, false, true);
        Mat<complex<double>> F1 = F111.st();

        Mat<complex<double>> BQvo(BQvo_a.memptr(), n_aux*n_occ, n_vir, false, true);
        Mat<complex<double>> BQoo(BQoo_a.memptr(), n_aux*n_occ, n_occ, false, true);

        Mat<complex<double>> F2 = BQoo.st() * BQvo;

        Mat<complex<double>> F4 (n_occ, n_vir, fill::zeros);
        Mat<complex<double>> F_hat = F1 - F2;

	    /// step 4:

        #pragma omp parallel
        {
	        complex<double> Eost = 0.0, Esst = 0.0; // local variable
            Mat<complex<double>> omega_I_local (n_vir, n_occ, fill::zeros);
            Mat<complex<double>> Y_local (n_aux, n_vir*n_occ, fill::zeros);
            #pragma omp for schedule(dynamic)
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                Mat<complex<double>> Bov_i(BQov_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                Mat<complex<double>> Bov_j(BQov_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // t2: VOVO
                Mat<complex<double>> Bhp_i(BQhp_a.colptr(i*n_vir), n_aux, n_vir, false, true);
                Mat<complex<double>> Bhp_j(BQhp_a.colptr(j*n_vir), n_aux, n_vir, false, true);

                // integrals
                Mat<complex<double>> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                Mat<complex<double>> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                complex<double> delta_ij = e_orb(i)+e_orb(j);
                complex<double> t2ab(0.,0.), t2ba(0.,0.);


                if(i == j) {
                    const complex<double> *w0 = W0.memptr();
                    const complex<double> *w1 = W1.memptr();

                    for(size_t b = 0; b < n_vir; b++) {

                        const complex<double> *w0b = w0 + b * n_vir;
                        const complex<double> *w1b = w1 + b * n_vir;
                        complex<double> dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            complex<double> denom_t2ab = dijb - e_orb[n_occ+a];
                            t2ab = (conj(denom_t2ab) * w1b[a]) / (conj(denom_t2ab) * denom_t2ab);
                            Eost += w0b[a] * (t1(a,i)*t1(b,j) + t2ab);
                            Esst += (w0b[a] - w0[a*n_vir+b]) * (t1(a,i)*t1(b,j) + t2ab);

                            for(size_t P = 0; P < n_aux; P++) {
                                Y_local[(a*n_occ*n_aux+i*n_aux+P)] += t2ab * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                            }

                            omega_I_local(a,i) += t2ab * F_hat(j,b);

                        }
                    }
                }
                else {
                    const complex<double> *w0 = W0.memptr();
                    const complex<double> *w1 = W1.memptr();

                    for(size_t b = 0; b < n_vir; b++) {

                        const complex<double> *w0b = w0 + b * n_vir;
                        const complex<double> *w1b = w1 + b * n_vir;
                        complex<double> dijb = delta_ij - e_orb[n_occ+b];

                        for(size_t a = 0; a < n_vir; a++) {
                            complex<double> denom = dijb - e_orb[n_occ+a];
                            t2ab = (conj(denom) * w1b[a]) / (conj(denom) * denom);
                            t2ba = (conj(denom) * w1[a*n_vir+b]) / (conj(denom) * denom);
                            Eost += 2.0 * w0b[a] * (t1(a,i)*t1(b,j) + t2ab);
                            Esst += (w0b[a] - w0[a*n_vir+b]) * (t1(a,i)*t1(b,j) + t2ab);

                            for(size_t P = 0; P < n_aux; P++) {
                                Y_local[(a*n_occ*n_aux+i*n_aux+P)] += (2.0*t2ab-t2ba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                                Y_local[(b*n_occ*n_aux+j*n_aux+P)] += (2.0*t2ab-t2ba) * BQov_a[(i*n_vir*n_aux+a*n_aux+P)];
                            }

                            omega_I_local(a,i) += (2.0*t2ab-t2ba) * F_hat(j,b);
                            omega_I_local(b,j) += (2.0*t2ab-t2ba) * F_hat(i,a);

                        }
                    }
                }
            }
  	        // eliminate the race condition by declaring a critical section
            #pragma omp critical (Y)
            {
	            eos += Eost;
	            ess += Esst;
                Y += Y_local;
	            omega_I += omega_I_local;
            }
        } // end parallel (1)

        /// step 5:

	    // omega_G1: first term of Γ(P,iβ)
        Mat<complex<double>> YQia(Y.memptr(), n_aux*n_occ, n_vir, false, true);
        Mat<complex<double>> gamma_G11 = YQia * CvirtA.st(); // (n_aux*n_occ,n_orb)
        Mat<complex<double>> gamma_G1 = gamma_G11.submat( 0, 0, n_aux-1, n_orb-1 );

        // omega_G2: third term of Γ(P,iβ)
        Mat<complex<double>> BQoh(BQoh_a.memptr(), n_aux*n_occ, n_occ, false, true);
        Mat<complex<double>> gamma_G22 = BQoh * (CvirtA * t1).st(); // (n_aux*n_occ, n_orb)
        Mat<complex<double>> gamma_G2 = gamma_G22.submat( 0, 0, n_aux-1, n_orb-1 );
        for(size_t i = 1; i < n_occ; i++) {
            gamma_G1.insert_cols(i*n_orb, gamma_G11.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            gamma_G2.insert_cols(i*n_orb, gamma_G22.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
        }

        // omega_J: second term of Γ(P,iβ)
        Mat<complex<double>> gamma_J0 = 2.0 * iQ * vectorise(Lam_hA).st();
        Mat<complex<double>> gamma_J(gamma_J0.memptr(), n_aux*n_occ, n_orb, false, true);

        // combine omega_G and omega_J: full terms of Γ(P,iβ)
        Mat<complex<double>> gamma_Q = gamma_G1 - gamma_G2 + gamma_J;

	    // V_PQ^(-1/2)
        Mat<TI> PQinvhalf(arrays<TI>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);
        arma::Mat<double> gamma_P_re = (real(PQinvhalf) * real(gamma_Q)) - (imag(PQinvhalf) * imag(gamma_Q));
        arma::Mat<double> gamma_P_im = (real(PQinvhalf) * imag(gamma_Q)) + (imag(PQinvhalf) * real(gamma_Q));

        gamma_P.set_real(gamma_P_re);
        gamma_P.set_imag(gamma_P_im);


        #pragma omp parallel
	    {
            Mat<complex<double>> JG_local (n_orb, n_occ, fill::zeros);
            #pragma omp for
            for(size_t P = 0; P < n_aux; P++) {
                for(size_t i = 0; i < n_occ; i++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            JG_local(alpha,i) += gamma_P[(i*n_orb*n_aux+beta*n_aux+P)] //(occ,aux*orb) orb
                                                        * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)]; //(aux*orb,orb) occ

                        }
                    }
                }
            }
            #pragma omp critical (JG)
            {
                JG += JG_local;
            }
    	} // end parallel(3)


        /// step 6:

        // omega_JG
        omega_JG += Lam_pA.st() * JG;

        #pragma omp parallel
	    {
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occ; k++) {
                            omega_H(a,i) += Y[(a*n_occ*n_aux+k*n_aux+P)]
                                                * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                        }
                    }

                    complex<double> delta_ia = e_orb(i) - e_orb[n_occ+a];

                    t1(a,i) = (conj(delta_ia) * (omega_JG(a,i) - omega_H(a,i) + omega_I(a,i))) /
                                            (conj(delta_ia) * delta_ia);

                }
            }
        } // end parallel(4)
    }


    Eos = eos;
    Ess = 2.0 * ess;
    
}

template<typename TC, typename TI>
void ricc2<TC, TI>::restricted_energy_digestor(
    complex<double>& Eos, complex<double>& Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<complex<double>> &BQvo_a,
    Mat<complex<double>> &BQov_a,
    Mat<complex<double>> &BQhp_a,
    Mat<complex<double>> &BQoo_a,
    Mat<complex<double>> &BQoh_a,
    Mat<complex<double>> &Lam_hA,
    Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &CoccA,
    Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &t1,
    Col<complex<double>> &e_orb,
    array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
    double c_os, double c_ss) {

    // GPP: activate this with the digestor
    throw std::runtime_error("Digestor option for this algorithm is not yet implemented.");

}


template<>
void ricc2<double, double>::unrestricted_energy(
    double &Eos, double &Essa, double &Essb,
    const size_t n_occa, const size_t n_vira,
    const size_t n_occb, const size_t n_virb,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQvo_a, Mat<double> &BQov_a, 
    Mat<double> &BQhp_a, Mat<double> &BQoo_a, 
    Mat<double> &BQoh_a, Mat<double> &BQvo_b, 
    Mat<double> &BQov_b, Mat<double> &BQhp_b, 
    Mat<double> &BQoo_b, Mat<double> &BQoh_b, 
    Mat<double> &V_Pab,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &t1a, Mat<double> &t1b, 
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss) {

    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::mat F_hat_a (n_occa, n_vira, fill::zeros);
    arma::mat omega_H_a (n_vira, n_occa, fill::zeros);
    arma::mat omega_I_a (n_vira, n_occa, fill::zeros);
    arma::mat Y_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat gamma_Q_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::mat gamma_P_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat omega_JG_a (n_vira, n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::mat F_hat_b (n_occb, n_virb, fill::zeros);
    arma::mat omega_H_b (n_virb, n_occb, fill::zeros);
    arma::mat omega_I_b (n_virb, n_occb, fill::zeros);
    arma::mat Y_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat gamma_Q_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::mat gamma_P_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat omega_JG_b (n_virb, n_occb, fill::zeros);

    {

        /// step 3:  form i^Q and F_ia

        // form i^Q
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);


        // Form F_ia
        // (AA|AA), (BB|AA)
        arma::Mat<double> F11a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F111a(F11a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> F1a = F111a.st();

        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);

        arma::Mat<double> F2a = BQooA.st() * BQvoA;

        arma::Mat<double> F4a (n_occa, n_vira, fill::zeros);
        F_hat_a = F1a - F2a;


        // (BB|BB), (AA|BB)
        arma::Mat<double> F11b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F111b(F11b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> F1b = F111b.st();

        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);

        arma::Mat<double> F2b = BQooB.st() * BQvoB;

        arma::Mat<double> F4b (n_occb, n_virb, fill::zeros);
        F_hat_b = F1b - F2b;


        ///step 4:

        auto start_step4 = high_resolution_clock::now();
        
        // (AA|BB)
        double Eosa = 0.0;
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
            
            double Eost = 0.0;
            arma::Mat<double> omega_I_a_local (n_vira, n_occa, fill::zeros);
            arma::Mat<double> Y_a_local (n_aux, n_vira*n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bov_j(BQov_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w1b[a] / (dijb - eA[n_occa+a]);

                        // energy calculation
                        Eost += w0b[a] * (t1a(a,i)*t1b(b,j) + t2ab);
                        

                        // Omega_H:
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_a[(a*n_occa*n_aux+i*n_aux+P)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_a_local[(a*n_occa*n_aux+i*n_aux+P)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }
                        
                        // Omega_I
                        // omega_I_a(a,i) += t2ab * F_hat_b(j,b);
                        omega_I_a_local(a,i) += t2ab * F_hat_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Eosa += Eost;
                Y_a += Y_a_local;
                omega_I_a += omega_I_a_local;
            }
        } // end parallel (1)

        Eos = Eosa;

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

            arma::Mat<double> omega_I_b_local (n_virb, n_occb, fill::zeros);
            arma::Mat<double> Y_b_local (n_aux, n_virb*n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w1b[a] / (dijb - eB[n_occb+a]);
                        
                        
                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_b[(a*n_occb*n_aux+i*n_aux+P)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_b_local[(a*n_occb*n_aux+i*n_aux+P)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            // omega_I_b(a,i) += t2ba * F_hat_a(j,b);
                            omega_I_b_local(a,i) += t2ba * F_hat_a(j,b);

                    }
                }



            }
            #pragma omp critical (Y_b)
            {
                Y_b += Y_b_local;
                omega_I_b += omega_I_b_local;
            }
        } // end parallel (2)


        //(AA|AA)
        Essa = 0.0;
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


            double Esst=0.0;
            arma::Mat<double> omega_I_a_local (n_vira, n_occa, fill::zeros);
            arma::Mat<double> Y_a_local (n_aux, n_vira*n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // cout << "i: " << i << " j: " << j << endl;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // t2: VOVO
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);


                // Bov_i.print("Bov_i");
                // Bov_j.print("Bov_j");

                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                double delta_ij = eA(i) + eA(j);

                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();

                    for(size_t b = 0; b < n_vira; b++) {

                        const double *w0b = w0 + b * n_vira;
                        const double *w1b = w1 + b * n_vira;
                        double dijb = delta_ij - eA[n_occa+b];

                        for(size_t a = 0; a < n_vira; a++) {
                            double t2aa = w1b[a] / (dijb - eA[n_occa+a]);
                            double t2aa_2 = w1[a*n_vira+b] / (dijb - eA[n_occa+a]);
                            Esst += (w0b[a] - w0[a*n_vira+b]) * (t1a(a,i)*t1a(b,j) + t2aa);

                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_a[(a*n_occa*n_aux+i*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                // Y_a[(b*n_occa*n_aux+j*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+P)];
                                Y_a_local[(a*n_occa*n_aux+i*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_a_local[(b*n_occa*n_aux+j*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+P)];
                            }

                            // omega_I_a(a,i) += (t2aa - t2aa_2) * F_hat_a(j,b);
                            // omega_I_a(b,j) += (t2aa - t2aa_2) * F_hat_a(i,a);
                            omega_I_a_local(a,i) += (t2aa - t2aa_2) * F_hat_a(j,b);
                            omega_I_a_local(b,j) += (t2aa - t2aa_2) * F_hat_a(i,a);

                        }
                    }
            }
            #pragma omp critical (omega_I_a)
            {
                Essa += Esst;
                Y_a += Y_a_local;
                omega_I_a += omega_I_a_local;
            }
        }

        //(BB|BB)
        Essb = 0.0;
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

            double Esst=0.0;
            arma::Mat<double> omega_I_b_local (n_virb, n_occb, fill::zeros);
            arma::Mat<double> Y_b_local (n_aux, n_virb*n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bov_j(BQov_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // t2: VOVO
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                double delta_ij = eB(i)+eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();

                for(size_t b = 0; b < n_virb; b++) {

                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w1b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w1[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        Esst += (w0b[a] - w0[a*n_virb+b]) * (t1b(a,i)*t1b(b,j) + t2bb);

                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_b[(a*n_occb*n_aux+i*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            // Y_b[(b*n_occb*n_aux+j*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+P)];
                            Y_b_local[(a*n_occb*n_aux+i*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_b_local[(b*n_occb*n_aux+j*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+P)];
                        }

                        // omega_I_b(a,i) += (t2bb - t2bb_2) * F_hat_b(j,b);
                        // omega_I_b(b,j) += (t2bb - t2bb_2) * F_hat_b(i,a);
                        omega_I_b_local(a,i) += (t2bb - t2bb_2) * F_hat_b(j,b);
                        omega_I_b_local(b,j) += (t2bb - t2bb_2) * F_hat_b(i,a);

                    }
                }
            }
            #pragma omp critical (omega_I_b_local)
            {
                Essb+=Esst;
                Y_b += Y_b_local;
                omega_I_b += omega_I_b_local;
            }
        }


        /// step 5:

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (AA|AA), (AA|BB)
        #pragma omp parallel
        {

            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia(Y_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<double> gamma_G11a = YQia * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_G1a = gamma_G11a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_G1a.insert_cols(i*n_orb, gamma_G11a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_G2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_G22a = BQohA * (CvirtA * t1a).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_G2a = gamma_G22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_G2a.insert_cols(i*n_orb, gamma_G22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J0a = (iQ_a * vectorise(Lam_hA).st()) + (iQ_b * vectorise(Lam_hA).st());
            // arma::Mat<double> gamma_Ja(gamma_J0a.memptr(), n_aux*n_occa, n_orb, false, true);
            arma::Mat<double> gamma_Ja(gamma_J0a.memptr(), n_aux, n_orb*n_occa, false, true);

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_G1a - gamma_G2a + gamma_Ja;

            arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;

            arma::Mat<double> JG_a_local (n_orb, n_occa, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            // JG_a(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                            //                     * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
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

        } // end (AA|AA), (AA|BB)


        // (BB|BB), (BB|AA)
        #pragma omp parallel
        {

            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQib(Y_b.memptr(), n_aux*n_occb, n_virb, false, true);
            arma::Mat<double> gamma_G11b = YQib * CvirtB.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_G1b = gamma_G11b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_G1b.insert_cols(i*n_orb, gamma_G11b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_G2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_G22b = BQohB * (CvirtB * t1b).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_G2b = gamma_G22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_G2b.insert_cols(i*n_orb, gamma_G22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J0b = (iQ_b * vectorise(Lam_hB).st()) + (iQ_a * vectorise(Lam_hB).st());
            // arma::Mat<double> gamma_Jb(gamma_J0b.memptr(), n_aux*n_occb, n_orb, false, true);
            arma::Mat<double> gamma_Jb(gamma_J0b.memptr(), n_aux, n_orb*n_occb, false, true);

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_G1b - gamma_G2b + gamma_Jb;

            arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
            gamma_Pb = PQinvhalf * gamma_Qb;

            arma::mat JG_b_local (n_orb, n_occb, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            // JG_b(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                            //                     * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
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
        } // end (BB|BB), (BB|AA)


        /// step 6:

        // omega_JG
        omega_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            omega_H_a(a,i) += Y_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                    * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }

                    double delta_A = eA(i) - eA[n_occa+a];

                    t1a(a,i) = (omega_JG_a(a,i) - omega_H_a(a,i) + omega_I_a(a,i)) / delta_A;

                }
            }
        } // end (AA|AA)


        // omega_JG
        omega_JG_b += Lam_pB.st() * JG_b;

        // (BB|BB)
        #pragma omp parallel
        {

            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            omega_H_b(a,i) += Y_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                    * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }

                    double delta_B = eB(i) - eB[n_occb+a];

                    t1b(a,i) = (omega_JG_b(a,i) - omega_H_b(a,i) + omega_I_b(a,i)) / delta_B;

                }
            }
        } // end (BB|BB)


    }

}


template<>
void ricc2<double, double>::unrestricted_energy_digestor(
    double &Eos, double &Essa, double &Essb,
    const size_t n_occa, const size_t n_vira,
    const size_t n_occb, const size_t n_virb,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQvo_a, Mat<double> &BQov_a,
    Mat<double> &BQhp_a, Mat<double> &BQoo_a, 
    Mat<double> &BQoh_a, Mat<double> &BQvo_b, 
    Mat<double> &BQov_b, Mat<double> &BQhp_b, 
    Mat<double> &BQoo_b, Mat<double> &BQoh_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, 
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &CoccA, Mat<double> &CvirtA, 
    Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &t1a, Mat<double> &t1b, 
    Col<double> &eA, Col<double> &eB,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
    double c_os, double c_ss) {

    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::mat F_hat_a (n_occa, n_vira, fill::zeros);
    arma::mat omega_H_a (n_vira, n_occa, fill::zeros);
    arma::mat omega_I_a (n_vira, n_occa, fill::zeros);
    arma::mat Y_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat gamma_Q_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::mat gamma_P_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat omega_JG_a (n_vira, n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::mat F_hat_b (n_occb, n_virb, fill::zeros);
    arma::mat omega_H_b (n_virb, n_occb, fill::zeros);
    arma::mat omega_I_b (n_virb, n_occb, fill::zeros);
    arma::mat Y_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat gamma_Q_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::mat gamma_P_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat omega_JG_b (n_virb, n_occb, fill::zeros);

    {

        /// step 3:  form i^Q and F_ia

        // form i^Q
        // (AA|AA)
        iQ_a += BQov_a * vectorise(t1a);

        // (BB|BB)
        iQ_b += BQov_b * vectorise(t1b);


        // Form F_ia
        // (AA|AA), (BB|AA)
        arma::Mat<double> F11a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<double> F111a(F11a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<double> F1a = F111a.st();

        arma::Mat<double> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<double> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);

        arma::Mat<double> F2a = BQooA.st() * BQvoA;

        arma::Mat<double> F4a (n_occa, n_vira, fill::zeros);
        F_hat_a = F1a - F2a;


        // (BB|BB), (AA|BB)
        arma::Mat<double> F11b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<double> F111b(F11b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<double> F1b = F111b.st();

        arma::Mat<double> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<double> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);

        arma::Mat<double> F2b = BQooB.st() * BQvoB;

        arma::Mat<double> F4b (n_occb, n_virb, fill::zeros);
        F_hat_b = F1b - F2b;


        ///step 4:
        
        auto start_step4 = high_resolution_clock::now();

        // (AA|BB)
        double Eosa = 0.0;
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
            
            double Eost = 0.0;
            arma::Mat<double> omega_I_a_local (n_vira, n_occa, fill::zeros);
            arma::Mat<double> Y_a_local (n_aux, n_vira*n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bov_j(BQov_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // for t2
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO
                
                double delta_ij = eA(i) + eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();

                for(size_t b = 0; b < n_virb; b++) {
                    
                    const double *w0b = w0 + b * n_vira;
                    const double *w1b = w1 + b * n_vira;

                    double dijb = delta_ij - eB[n_occb+b];
                    
                    for(size_t a = 0; a < n_vira; a++) {
                        
                        double t2ab = w1b[a] / (dijb - eA[n_occa+a]);

                        // energy calculation
                        Eost += w0b[a] * (t1a(a,i)*t1b(b,j) + t2ab);
                        

                        // Omega_H:
                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_a[(a*n_occa*n_aux+i*n_aux+P)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_a_local[(a*n_occa*n_aux+i*n_aux+P)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                        }
                        
                        // Omega_I
                        // omega_I_a(a,i) += t2ab * F_hat_b(j,b);
                        omega_I_a_local(a,i) += t2ab * F_hat_b(j,b);

                    }
                }
            }
            #pragma omp critical (Y_a)
            {
                Eosa += Eost;
                Y_a += Y_a_local;
                omega_I_a += omega_I_a_local;
            }
        } // end parallel (1)

        Eos = Eosa;

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

            arma::Mat<double> omega_I_b_local (n_virb, n_occb, fill::zeros);
            arma::Mat<double> Y_b_local (n_aux, n_virb*n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // for t2
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);
                
                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO
                
                double delta_ij = eB(i) + eA(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();

                for(size_t b = 0; b < n_vira; b++) {
                    
                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;

                    double dijb = delta_ij - eA[n_occa+b];
                    
                    for(size_t a = 0; a < n_virb; a++) {
                        
                        double t2ba = w1b[a] / (dijb - eB[n_occb+a]);
                        
                        
                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_b[(a*n_occb*n_aux+i*n_aux+P)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_b_local[(a*n_occb*n_aux+i*n_aux+P)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            // omega_I_b(a,i) += t2ba * F_hat_a(j,b);
                            omega_I_b_local(a,i) += t2ba * F_hat_a(j,b);

                    }
                }



            }
            #pragma omp critical (Y_b)
            {
                Y_b += Y_b_local;
                omega_I_b += omega_I_b_local;
            }
        } // end parallel (2)


        //(AA|AA)
        Essa = 0.0;
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


            double Esst=0.0;
            arma::Mat<double> omega_I_a_local (n_vira, n_occa, fill::zeros);
            arma::Mat<double> Y_a_local (n_aux, n_vira*n_occa, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // cout << "i: " << i << " j: " << j << endl;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // t2: VOVO
                arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);


                // Bov_i.print("Bov_i");
                // Bov_j.print("Bov_j");

                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                double delta_ij = eA(i) + eA(j);

                    const double *w0 = W0.memptr();
                    const double *w1 = W1.memptr();

                    for(size_t b = 0; b < n_vira; b++) {

                        const double *w0b = w0 + b * n_vira;
                        const double *w1b = w1 + b * n_vira;
                        double dijb = delta_ij - eA[n_occa+b];

                        for(size_t a = 0; a < n_vira; a++) {
                            double t2aa = w1b[a] / (dijb - eA[n_occa+a]);
                            double t2aa_2 = w1[a*n_vira+b] / (dijb - eA[n_occa+a]);
                            Esst += (w0b[a] - w0[a*n_vira+b]) * (t1a(a,i)*t1a(b,j) + t2aa);

                            for(size_t P = 0; P < n_aux; P++) {
                                // Y_a[(a*n_occa*n_aux+i*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                // Y_a[(b*n_occa*n_aux+j*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+P)];
                                Y_a_local[(a*n_occa*n_aux+i*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_a_local[(b*n_occa*n_aux+j*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+P)];
                            }

                            // omega_I_a(a,i) += (t2aa - t2aa_2) * F_hat_a(j,b);
                            // omega_I_a(b,j) += (t2aa - t2aa_2) * F_hat_a(i,a);
                            omega_I_a_local(a,i) += (t2aa - t2aa_2) * F_hat_a(j,b);
                            omega_I_a_local(b,j) += (t2aa - t2aa_2) * F_hat_a(i,a);

                        }
                    }
            }
            #pragma omp critical (omega_I_a)
            {
                Essa += Esst;
                Y_a += Y_a_local;
                omega_I_a += omega_I_a_local;
            }
        }

        //(BB|BB)
        Essb = 0.0;
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

            double Esst=0.0;
            arma::Mat<double> omega_I_b_local (n_virb, n_occb, fill::zeros);
            arma::Mat<double> Y_b_local (n_aux, n_virb*n_occb, fill::zeros);
            #pragma omp for
            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<double> Bov_i(BQov_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bov_j(BQov_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // t2: VOVO
                arma::Mat<double> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<double> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                double delta_ij = eB(i)+eB(j);

                const double *w0 = W0.memptr();
                const double *w1 = W1.memptr();

                for(size_t b = 0; b < n_virb; b++) {

                    const double *w0b = w0 + b * n_virb;
                    const double *w1b = w1 + b * n_virb;
                    double dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        double t2bb = w1b[a] / (dijb - eB[n_occb+a]);
                        double t2bb_2 = w1[a*n_virb+b] / (dijb - eB[n_occb+a]);
                        Esst += (w0b[a] - w0[a*n_virb+b]) * (t1b(a,i)*t1b(b,j) + t2bb);

                        for(size_t P = 0; P < n_aux; P++) {
                            // Y_b[(a*n_occb*n_aux+i*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            // Y_b[(b*n_occb*n_aux+j*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+P)];
                            Y_b_local[(a*n_occb*n_aux+i*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_b_local[(b*n_occb*n_aux+j*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+P)];
                        }

                        // omega_I_b(a,i) += (t2bb - t2bb_2) * F_hat_b(j,b);
                        // omega_I_b(b,j) += (t2bb - t2bb_2) * F_hat_b(i,a);
                        omega_I_b_local(a,i) += (t2bb - t2bb_2) * F_hat_b(j,b);
                        omega_I_b_local(b,j) += (t2bb - t2bb_2) * F_hat_b(i,a);

                    }
                }
            }
            #pragma omp critical (omega_I_b_local)
            {
                Essb+=Esst;
                Y_b += Y_b_local;
                omega_I_b += omega_I_b_local;
            }
        }

        /// step 5:

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (AA|AA), (AA|BB)
        // #pragma omp parallel
        {

            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQia(Y_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<double> gamma_G11a = YQia * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_G1a = gamma_G11a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_G1a.insert_cols(i*n_orb, gamma_G11a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_G2: third term of Γ(P,iβ)
            arma::Mat<double> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<double> gamma_G22a = BQohA * (CvirtA * t1a).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_G2a = gamma_G22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_G2a.insert_cols(i*n_orb, gamma_G22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J0a = (iQ_a * vectorise(Lam_hA).st()) + (iQ_b * vectorise(Lam_hA).st());
            // arma::Mat<double> gamma_Ja(gamma_J0a.memptr(), n_aux*n_occa, n_orb, false, true);
            arma::Mat<double> gamma_Ja(gamma_J0a.memptr(), n_aux, n_orb*n_occa, false, true);

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qa = gamma_G1a - gamma_G2a + gamma_Ja;
            arma::Mat<double> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;

            // GPP: this is the digestor that replaces the formation of JG
            {

                //  Step 1: Read libqints-type basis set from files and form shellpair basis.
                // libqints::basis_1e2c_shellpair_cgto<double> bsp;
                // libqints::basis_1e1c_cgto<double> b1x;  //  1e1c auxiliary basis
                const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
                const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
                size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
                size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
                size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
                
                //  Step 2: Construct the 2e3c shellpair basis and corresponding full basis range
                libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
                libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
                libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);
                
                //  Step 3: prepare required input settings
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
                
                //  Step 4: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
                scr_2e3c scr(m_b3);
                
                //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
                dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
                if (dev.memory * dev.nthreads > mem_total) {
                    std::cout << " Given memory is not enough for computing integrals." << std::endl;
                    qjob.end();  //  End the libqints job before return
                    return;
                }
                
                size_t ni = n_occa;
                arma::mat L(n_aux, n_orb * ni, arma::fill::randn);
                size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity
                
                //  Step 5:
                //  Memory available for thread-local result arrays:
                size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
                //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
                size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
                // size_t nmunu = 0, nP = 0;
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
                JG_a.zeros();  
                dig_2e3c<double> dig(m_b3, ni, gamma_Pa, JG_a);
                
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
            }

        } // end (AA|AA), (AA|BB)
        

        // (BB|BB), (BB|AA)
        // #pragma omp parallel
        {

            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<double> YQib(Y_b.memptr(), n_aux*n_occb, n_virb, false, true);
            arma::Mat<double> gamma_G11b = YQib * CvirtB.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<double> gamma_G1b = gamma_G11b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_G1b.insert_cols(i*n_orb, gamma_G11b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_G2: third term of Γ(P,iβ)
            arma::Mat<double> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<double> gamma_G22b = BQohB * (CvirtB * t1b).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<double> gamma_G2b = gamma_G22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_G2b.insert_cols(i*n_orb, gamma_G22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J: second term of Γ(P,iβ)
            arma::Mat<double> gamma_J0b = (iQ_b * vectorise(Lam_hB).st()) + (iQ_a * vectorise(Lam_hB).st());
            // arma::Mat<double> gamma_Jb(gamma_J0b.memptr(), n_aux*n_occb, n_orb, false, true);
            arma::Mat<double> gamma_Jb(gamma_J0b.memptr(), n_aux, n_orb*n_occb, false, true);

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<double> gamma_Qb = gamma_G1b - gamma_G2b + gamma_Jb;
            arma::Mat<double> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
            gamma_Pb = PQinvhalf * gamma_Qb;

            // GPP: this is the digestor that replaces the formation of JG
            {

                //  Step 1: Read libqints-type basis set from files and form shellpair basis.
                // libqints::basis_1e2c_shellpair_cgto<double> bsp;
                // libqints::basis_1e1c_cgto<double> b1x;  //  1e1c auxiliary basis
                const libqints::basis_1e2c_shellpair_cgto<double> &bsp = m_b3.get_bra();
                const libqints::basis_1e1c_cgto<double> &b1x = m_b3.get_ket();
                size_t nbsp = bsp.get_nbsp();  //  # of munu basis function pairs
                size_t nsp = bsp.get_nsp();    //  # of munu shell pairs
                size_t ns_q = b1x.get_ns();    //  # of auxiliary basis shells
                
                //  Step 2: Construct the 2e3c shellpair basis and corresponding full basis range
                libqints::range<libqints::basis_2e3c_shellpair_cgto<double>> fbr(m_b3);
                libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 1> frbra(fbr);
                libqints::range1<libqints::basis_2e3c_shellpair_cgto<double>, 2> frket(fbr);
                
                //  Step 3: prepare required input settings
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
                
                //  Step 4: set up 2e3c integral screener, which is used for removing bra-ket pairs which are ignorable.
                scr_2e3c scr(m_b3);
                
                //  Step 4: Estimate memory requirement of libqints integral kernels per thread in Bytes
                dev.memory = libqints::qints_memreq(qjob, fbr, scr, dev);
                if (dev.memory * dev.nthreads > mem_total) {
                    std::cout << " Given memory is not enough for computing integrals." << std::endl;
                    qjob.end();  //  End the libqints job before return
                    return;
                }
                
                size_t ni = n_occb;
                arma::mat L(n_aux, n_orb * ni, arma::fill::randn);
                size_t mem_PWTFLV = 0;  //  memory for keeping these objects I just set to zero for simplicity
                
                //  Step 5:
                //  Memory available for thread-local result arrays:
                size_t mem_avail = mem_total - dev.memory * dev.nthreads - mem_PWTFLV;
                //  We need to make smaller basis ranges along either munu shellpair basis or auxiliary basis, or both.
                size_t nbsp_per_subrange = 0, naux_per_subrange = 0;
                // size_t nmunu = 0, nP = 0;
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
                JG_b.zeros();  
                dig_2e3c<double> dig(m_b3, ni, gamma_Pb, JG_b);
                
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
            }


        } // end (BB|BB), (BB|AA)

        /// step 6:
        // omega_JG
        omega_JG_a += Lam_pA.st() * JG_a;

        // (AA|AA)
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            omega_H_a(a,i) += Y_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                    * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }

                    double delta_A = eA(i) - eA[n_occa+a];

                    t1a(a,i) = (omega_JG_a(a,i) - omega_H_a(a,i) + omega_I_a(a,i)) / delta_A;

                }
            }
        } // end (AA|AA)
        // cout << "step 6 (AA|AA) done." << endl;


        // omega_JG
        omega_JG_b += Lam_pB.st() * JG_b;

        // (BB|BB)
        #pragma omp parallel
        {

            #pragma omp for
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            omega_H_b(a,i) += Y_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                    * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }

                    double delta_B = eB(i) - eB[n_occb+a];

                    t1b(a,i) = (omega_JG_b(a,i) - omega_H_b(a,i) + omega_I_b(a,i)) / delta_B;

                }
            }
        } // end (BB|BB)
        // cout << "step 6 (BB|BB) done." << endl;


    }

}

template<typename TC, typename TI>
void ricc2<TC, TI>::unrestricted_energy(
    complex<double> &Eos, complex<double> &Essa, complex<double> &Essb,
    const size_t n_occa, const size_t n_vira,
    const size_t n_occb, const size_t n_virb,
    const size_t n_aux, const size_t n_orb,
    Mat<complex<double>> &BQvo_a, Mat<complex<double>> &BQov_a,
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoo_a,
    Mat<complex<double>> &BQoh_a, Mat<complex<double>> &BQvo_b,
    Mat<complex<double>> &BQov_b, Mat<complex<double>> &BQhp_b,
    Mat<complex<double>> &BQoo_b, Mat<complex<double>> &BQoh_b,
    Mat<complex<double>> &V_Pab,
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hB, Mat<complex<double>> &Lam_pB,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &CoccB, Mat<complex<double>> &CvirtB,
    Mat<complex<double>> &t1a, Mat<complex<double>> &t1b,
    Col<complex<double>> &eA, Col<complex<double>> &eB,
    array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
    double c_os, double c_ss) {

    
    // intermediates
    arma::cx_vec iQ_a (n_aux, fill::zeros);
    arma::cx_mat F_hat_a (n_occa, n_vira, fill::zeros);
    arma::cx_mat omega_H_a (n_vira, n_occa, fill::zeros);
    arma::cx_mat omega_I_a (n_vira, n_occa, fill::zeros);
    arma::cx_mat Y_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::cx_mat gamma_Q_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::cx_mat gamma_P_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::cx_mat JG_a (n_orb, n_occa, fill::zeros);
    arma::cx_mat omega_JG_a (n_vira, n_occa, fill::zeros);

    arma::cx_vec iQ_b (n_aux, fill::zeros);
    arma::cx_mat F_hat_b (n_occb, n_virb, fill::zeros);
    arma::cx_mat omega_H_b (n_virb, n_occb, fill::zeros);
    arma::cx_mat omega_I_b (n_virb, n_occb, fill::zeros);
    arma::cx_mat Y_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::cx_mat gamma_Q_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::cx_mat gamma_P_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::cx_mat JG_b (n_orb, n_occb, fill::zeros);
    arma::cx_mat omega_JG_b (n_virb, n_occb, fill::zeros);

    {
        /// step 3:  form i^Q and F_ia

        // form i^Q
        // (AA|AA)
        iQ_a += BQov_a * t1a;

        // (BB|BB)
        iQ_b += BQov_b * t1b;


        // Form F_ia
        // (AA|AA), (BB|AA)
        arma::Mat<complex<double>> F11a = (iQ_a.st() * BQov_a) + (iQ_b.st() * BQov_a);
        arma::Mat<complex<double>> F111a(F11a.memptr(), n_vira, n_occa, false, true);
        arma::Mat<complex<double>> F1a = F111a.st();

        arma::Mat<complex<double>> BQvoA(BQvo_a.memptr(), n_aux*n_occa, n_vira, false, true);
        arma::Mat<complex<double>> BQooA(BQoo_a.memptr(), n_aux*n_occa, n_occa, false, true);

        arma::Mat<complex<double>> F2a = BQooA.st() * BQvoA;

        arma::Mat<complex<double>> F4a (n_occa, n_vira, fill::zeros);
        F_hat_a = F1a - F2a;


        // (BB|BB), (AA|BB)
        arma::Mat<complex<double>> F11b = (iQ_b.st() * BQov_b) + (iQ_a.st() * BQov_b);
        arma::Mat<complex<double>> F111b(F11b.memptr(), n_virb, n_occb, false, true);
        arma::Mat<complex<double>> F1b = F111b.st();

        arma::Mat<complex<double>> BQvoB(BQvo_b.memptr(), n_aux*n_occb, n_virb, false, true);
        arma::Mat<complex<double>> BQooB(BQoo_b.memptr(), n_aux*n_occb, n_occb, false, true);

        arma::Mat<complex<double>> F2b = BQooB.st() * BQvoB;

        arma::Mat<complex<double>> F4b (n_occb, n_virb, fill::zeros);
        F_hat_b = F1b - F2b;


        ///step 4:

        // (AA|BB)
        complex<double> Eosa(0.,0.);
        {
            complex<double> Eost(0.,0.);
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {

                            //denominator
                            complex<double> delta_AB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];

                            complex<double> num_os(0.,0.);
                            complex<double> t2ab(0.,0.);

                            for(size_t Q = 0; Q < n_aux; Q++) {

                                num_os += BQov_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                                // t2ab += BQph_a[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)];
                                // GPP: check if this is correct, using BQhp_a instead of BQph_a
                                t2ab += BQhp_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQhp_b[(j*n_virb*n_aux+b*n_aux+Q)];

                            }

                            t2ab = (conj(delta_AB) * t2ab) / (conj(delta_AB) * delta_AB);

			                // energy calculation
                            Eost += num_os * (t1a(a,i)*t1b(b,j) + t2ab);


                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                Y_a[(a*n_occa*n_aux+i*n_aux+P)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            omega_I_a(a,i) += t2ab * F_hat_b(j,b);

                        }
                    }
                }
            }

            Eosa += Eost;
        }



        // (BB|AA)
        complex<double> Eosb(0.,0.);
        {
            complex<double> Eost(0.,0.);
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {

                            //denominator
                            complex<double> delta_BA = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];

                            complex<double> num_os(0.,0.);
                            complex<double> t2ba(0.,0.);

                            for(size_t Q = 0; Q < n_aux; Q++) {

                                num_os += BQov_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                                // t2ba += BQph_b[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)];
                                t2ba += BQhp_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQhp_a[(j*n_vira*n_aux+b*n_aux+Q)];

                            }

                            t2ba = (conj(delta_BA) * t2ba) / (conj(delta_BA) * delta_BA);

                            // energy calculation
                            Eost += num_os * (t1b(a,i)*t1a(b,j) + t2ba);


                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                Y_b[(a*n_occb*n_aux+i*n_aux+P)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            omega_I_b(a,i) += t2ba * F_hat_a(j,b);

                        }
                    }
                }
            }

            Eosb += Eost;

        }

        Eos = 0.5*Eosa + 0.5*Eosb;

        //(AA|AA)
        Essa = (0.0,0.0);
        {

            complex<double> Esst(0.,0.);

            size_t npairs = (n_occa+1)*n_occa/2;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occa, n_occa, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occa; i++) {
            for(size_t j = 0; j <= i; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<complex<double>> Bov_i(BQov_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<complex<double>> Bov_j(BQov_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // t2: VOVO
                arma::Mat<complex<double>> Bhp_i(BQhp_a.colptr(i*n_vira), n_aux, n_vira, false, true);
                arma::Mat<complex<double>> Bhp_j(BQhp_a.colptr(j*n_vira), n_aux, n_vira, false, true);

                // integrals
                arma::Mat<complex<double>> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<complex<double>> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                complex<double> delta_ij = eA(i) + eA(j);

                    const complex<double> *w0 = W0.memptr();
                    const complex<double> *w1 = W1.memptr();

                    for(size_t b = 0; b < n_vira; b++) {

                        const complex<double> *w0b = w0 + b * n_vira;
                        const complex<double> *w1b = w1 + b * n_vira;
                        complex<double> dijb = delta_ij - eA[n_occa+b];

                        for(size_t a = 0; a < n_vira; a++) {
                            complex<double> denom_a = dijb - eA[n_occa+a];
                            complex<double> t2aa = (conj(denom_a) * w1b[a]) / (conj(denom_a) * denom_a);
                            complex<double> t2aa_2 = (conj(denom_a) * w1[a*n_vira+b]) / (conj(denom_a) * denom_a);
                            Esst += (w0b[a] - w0[a*n_vira+b]) * (t1a(a,i)*t1a(b,j) + t2aa);

                            for(size_t P = 0; P < n_aux; P++) {
                                Y_a[(a*n_occa*n_aux+i*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                                Y_a[(b*n_occa*n_aux+j*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(i*n_vira*n_aux+a*n_aux+P)];
                            }

                            omega_I_a(a,i) += (t2aa - t2aa_2) * F_hat_a(j,b);
                            omega_I_a(b,j) += (t2aa - t2aa_2) * F_hat_a(i,a);

                        }
                    }
            }
            Essa += Esst;
        }


        //(BB|BB)
        Essb = (0.0,0.0);
        {

            complex<double> Esst(0.,0.);

            size_t npairs = (n_occb+1)*n_occb/2;
            std::vector<size_t> occ_i2(npairs);
            idx2_list pairs(n_occb, n_occb, npairs,
                array_view<size_t>(&occ_i2[0], occ_i2.size()));
            for(size_t i = 0, ij = 0; i < n_occb; i++) {
            for(size_t j = 0; j <= i; j++, ij++)
                pairs.set(ij, idx2(i, j));
            }

            for(size_t ij = 0; ij < npairs; ij++) {
                idx2 i2 = pairs[ij];
                size_t i = i2.i, j = i2.j;

                // energy: vovo
                arma::Mat<complex<double>> Bov_i(BQov_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<complex<double>> Bov_j(BQov_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // t2: VOVO
                arma::Mat<complex<double>> Bhp_i(BQhp_b.colptr(i*n_virb), n_aux, n_virb, false, true);
                arma::Mat<complex<double>> Bhp_j(BQhp_b.colptr(j*n_virb), n_aux, n_virb, false, true);

                // integrals
                arma::Mat<complex<double>> W0 = Bov_i.st() * Bov_j; // energy:  vovo
                arma::Mat<complex<double>> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

                complex<double> delta_ij = eB(i)+eB(j);

                const complex<double> *w0 = W0.memptr();
                const complex<double> *w1 = W1.memptr();

                for(size_t b = 0; b < n_virb; b++) {

                    const complex<double> *w0b = w0 + b * n_virb;
                    const complex<double> *w1b = w1 + b * n_virb;
                    complex<double> dijb = delta_ij - eB[n_occb+b];

                    for(size_t a = 0; a < n_virb; a++) {
                        complex<double> denom_b = dijb - eB[n_occb+a];
                        complex<double> t2bb = (conj(denom_b) * w1b[a]) / (conj(denom_b) * denom_b);
                        complex<double> t2bb_2 = (conj(denom_b) * w1[a*n_virb+b]) / (conj(denom_b) * denom_b);
                        Esst += (w0b[a] - w0[a*n_virb+b]) * (t1b(a,i)*t1b(b,j) + t2bb);

                        for(size_t P = 0; P < n_aux; P++) {
                            Y_b[(a*n_occb*n_aux+i*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            Y_b[(b*n_occb*n_aux+j*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(i*n_virb*n_aux+a*n_aux+P)];
                        }

                        omega_I_b(a,i) += (t2bb - t2bb_2) * F_hat_b(j,b);
                        omega_I_b(b,j) += (t2bb - t2bb_2) * F_hat_b(i,a);

                    }
                }
            }
            Essb+=Esst;
        }

        /// step 5:

        // V_PQ^(-1/2)
        arma::Mat<TI> PQinvhalf(arrays<TI>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

	    // (AA|AA), (AA|BB)
        {

            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<complex<double>> YQia(Y_a.memptr(), n_aux*n_occa, n_vira, false, true);
            arma::Mat<complex<double>> gamma_G11a = YQia * CvirtA.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<complex<double>> gamma_G1a = gamma_G11a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_G1a.insert_cols(i*n_orb, gamma_G11a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_G2: third term of Γ(P,iβ)
            arma::Mat<complex<double>> BQohA(BQoh_a.memptr(), n_aux*n_occa, n_occa, false, true);
            arma::Mat<complex<double>> gamma_G22a = BQohA * (CvirtA * t1a).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<complex<double>> gamma_G2a = gamma_G22a.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occa; i++) {
                gamma_G2a.insert_cols(i*n_orb, gamma_G22a.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J: second term of Γ(P,iβ)
            arma::Mat<complex<double>> gamma_J0a = (iQ_a * vectorise(Lam_hA).st()) + (iQ_b * vectorise(Lam_hA).st());
            arma::Mat<complex<double>> gamma_Ja(gamma_J0a.memptr(), n_aux*n_occa, n_orb, false, true);

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<complex<double>> gamma_Qa = gamma_G1a - gamma_G2a + gamma_Ja;

            arma::Mat<complex<double>> gamma_Pa (n_aux, n_orb*n_occa, fill::zeros);
            gamma_Pa = PQinvhalf * gamma_Qa;

            for(size_t i = 0; i < n_occa; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            JG_a(alpha,i) += gamma_Pa[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];

                        }
                    }
                }
            }
        } // end (AA|AA), (AA|BB)

        // (BB|BB), (BB|AA)
        {

            // omega_G1: first term of Γ(P,iβ)
            arma::Mat<complex<double>> YQib(Y_b.memptr(), n_aux*n_occb, n_virb, false, true);
            arma::Mat<complex<double>> gamma_G11b = YQib * CvirtB.st(); // (n_aux*n_occ,n_orb)
            arma::Mat<complex<double>> gamma_G1b = gamma_G11b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_G1b.insert_cols(i*n_orb, gamma_G11b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_G2: third term of Γ(P,iβ)
            arma::Mat<complex<double>> BQohB(BQoh_b.memptr(), n_aux*n_occb, n_occb, false, true);
            arma::Mat<complex<double>> gamma_G22b = BQohB * (CvirtB * t1b).st(); // (n_aux*n_occ, n_orb)
            arma::Mat<complex<double>> gamma_G2b = gamma_G22b.submat( 0, 0, n_aux-1, n_orb-1 );
            for(size_t i = 1; i < n_occb; i++) {
                gamma_G2b.insert_cols(i*n_orb, gamma_G22b.submat( i*n_aux, 0, (i+1)*n_aux-1, n_orb-1 ));
            }

            // omega_J: second term of Γ(P,iβ)
            arma::Mat<complex<double>> gamma_J0b = (iQ_b * vectorise(Lam_hB).st()) + (iQ_a * vectorise(Lam_hB).st());
            arma::Mat<complex<double>> gamma_Jb(gamma_J0b.memptr(), n_aux*n_occb, n_orb, false, true);

            // combine omega_G and omega_J: full terms of Γ(P,iβ)
            arma::Mat<complex<double>> gamma_Qb = gamma_G1b - gamma_G2b + gamma_Jb;

            arma::Mat<complex<double>> gamma_Pb (n_aux, n_orb*n_occb, fill::zeros);
            gamma_Pb = PQinvhalf * gamma_Qb;


            for(size_t i = 0; i < n_occb; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {

                            JG_b(alpha,i) += gamma_Pb[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];

                        }
                    }
                }
            }
        } // end (BB|BB), (BB|AA)


        /// step 6:

        // (AA|AA)
        {

            // omega_JG
            omega_JG_a += Lam_pA.st() * JG_a;

            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            omega_H_a(a,i) += Y_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                    * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }

                    complex<double> delta_A = eA(i) - eA[n_occa+a];

                    t1a(a,i) = (conj(delta_A) * (omega_JG_a(a,i) - omega_H_a(a,i) + omega_I_a(a,i)))
                               / (conj(delta_A) * delta_A);

                }
            }
        } // end (AA|AA)


        // (BB|BB)
        {
            // omega_JG
            omega_JG_b += Lam_pB.st() * JG_b;

            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            omega_H_b(a,i) += Y_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                    * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }

                    complex<double> delta_B = eB(i) - eB[n_occb+a];

                    t1b(a,i) = (conj(delta_B) * (omega_JG_b(a,i) - omega_H_b(a,i) + omega_I_b(a,i)))
                               / (conj(delta_B) * delta_B);

                }
            }
        } // end (BB|BB)
    }
    
}


template<typename TC, typename TI>
void ricc2<TC, TI>::unrestricted_energy_digestor(
    complex<double> &Eos, complex<double> &Essa, complex<double> &Essb,
    const size_t n_occa, const size_t n_vira,
    const size_t n_occb, const size_t n_virb,
    const size_t n_aux, const size_t n_orb,
    Mat<complex<double>> &BQvo_a, Mat<complex<double>> &BQov_a,
    Mat<complex<double>> &BQhp_a, Mat<complex<double>> &BQoo_a,
    Mat<complex<double>> &BQoh_a, Mat<complex<double>> &BQvo_b,
    Mat<complex<double>> &BQov_b, Mat<complex<double>> &BQhp_b,
    Mat<complex<double>> &BQoo_b, Mat<complex<double>> &BQoh_b,
    Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
    Mat<complex<double>> &Lam_hB, Mat<complex<double>> &Lam_pB,
    Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
    Mat<complex<double>> &CoccB, Mat<complex<double>> &CvirtB,
    Mat<complex<double>> &t1a, Mat<complex<double>> &t1b,
    Col<complex<double>> &eA, Col<complex<double>> &eB,
    array_view<TI> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
    double c_os, double c_ss) {


    // GPP: activate this with the digestor
    throw std::runtime_error("Digestor option for this algorithm is not yet implemented. Please increase mem_total.");

}

#if 0
/// GPP: RI-MP2 calculation only
template<>
void ricc2<double>::restricted_energy(
    double &Eos, double &Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQov_a, Mat<double> &BQvo_a, Mat<double> &BQph_a,
    Mat<double> &BQhp_a, Mat<double> &BQoo_a, Mat<double> &BQvv_a,
    Mat<double> &BQhh_a, Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &H1_a, Mat<double> &H2_a,
    Mat<double> &t1, Col<double> &e_orb) {

    size_t npairs = (n_occ+1)*n_occ/2;
    std::vector<size_t> occ_i2(npairs);
    idx2_list pairs(n_occ, n_occ, npairs,
        array_view<size_t>(&occ_i2[0], occ_i2.size()));
    for(size_t i = 0, ij = 0; i < n_occ; i++) {
    for(size_t j = 0; j <= i; j++, ij++)
        pairs.set(ij, idx2(i, j));
    }


    double eos = 0.0, ess = 0.0;

    {
        double Eost=0.0;
        double Esst=0.0;

        size_t naux_ov = BQov_a.n_rows;

        // pointers
        int vvv = n_vir*n_vir*n_vir;
        int ovv = n_occ*n_vir*n_vir;
        int oov = n_occ*n_occ*n_vir;
        int ooo = n_occ*n_occ*n_occ;
        int vv = n_vir*n_vir;
        int ov = n_occ*n_vir;
        int oo = n_occ*n_occ;
        int v = n_vir;
        int o = n_occ;

        // for i and j
        for(size_t ij = 0; ij < npairs; ij++) {
            idx2 i2 = pairs[ij];
            size_t i = i2.i, j = i2.j;

            // energy: vovo
            arma::Mat<double> Bov_i(BQov_a.colptr(i*n_vir), naux_ov, n_vir, false, true);
            arma::Mat<double> Bov_j(BQov_a.colptr(j*n_vir), naux_ov, n_vir, false, true);

            // t2: VOVO
            arma::Mat<double> Bhp_i(BQhp_a.colptr(i*n_vir), naux_ov, n_vir, false, true);
            arma::Mat<double> Bhp_j(BQhp_a.colptr(j*n_vir), naux_ov, n_vir, false, true);

            // integrals
            arma::Mat<double> W0 = Bov_i.st() * Bov_j; // energy:  vovo
            arma::Mat<double> W1 = Bhp_i.st() * Bhp_j; // t2:   VOVO

            const double *w0 = W0.memptr();
            const double *w1 = W1.memptr();

            // Main loop
            if(i == j) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t a = 0; a < n_vir; a++) {

                        double d_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];

                        double num_os = w0[(b*v+a)];
                        double num_ss = w0[(b*v+a)] - w0[(a*v+b)];
                        double t_ijab = w1[(b*v+a)] / d_ijab;

                        Eost += num_os * t_ijab;
                        Esst += num_ss * t_ijab;
                    }
                }
            } else {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t a = 0; a < n_vir; a++) {

                        double d_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];

                        double num_os = w0[(b*v+a)];
                        double num_ss = w0[(b*v+a)] - w0[(a*v+b)];
                        double t_ijab = w1[(b*v+a)] / d_ijab;

                        Eost += 2.0 * num_os * t_ijab;
                        Esst += num_ss * t_ijab;
                    }
                }
            }
        }


        eos += Eost;
        ess += Esst;

    }


    Eos = eos;
    Ess = 2.0 * ess;
}



/// GPP: RI-CC2 calculation but NOT Haettig's algorithm
template<>
void ricc2<double>::restricted_energy(
    double &Eos, double &Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQov_a, Mat<double> &BQph_a, Mat<double> &BQoh_a,
    Mat<double> &BQvo_a, Mat<double> &BQpv_a,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &H1_a, Mat<double> &H2_a,
    Mat<double> &t1, Col<double> &e_orb) {

    double eos = 0.0, ess = 0.0;

    {
        double Eost=0.0;
        double Esst=0.0;

        // intermediates
        arma::mat G (n_vir, n_occ, fill::zeros);
        arma::mat H (n_vir, n_occ, fill::zeros);
        arma::mat I (n_vir, n_occ, fill::zeros);
        arma::mat F1 (n_occ, n_vir, fill::zeros);
        arma::mat F2 (n_vir, n_occ, fill::zeros);

        arma::Mat<double> W0 = BQvo_a.st() * BQvo_a; // Ene:  (vo|vo)
        arma::Mat<double> W1 = BQph_a.st() * BQph_a; // t2:   (VO|VO)
        arma::Mat<double> W2 = BQpv_a.st() * BQov_a; // G:    (ov|Vv) - (jb|ca)
        arma::Mat<double> W3 = BQoh_a.st() * BQov_a; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)
        arma::Mat<double> W4 = BQoh_a.st() * BQph_a; // J:    (VO|oO) - (ai|kk)

        const double *w0 = W0.memptr();
        const double *w1 = W1.memptr();
        const double *w2 = W2.memptr();
        const double *w3 = W3.memptr();
        const double *w4 = W4.memptr();

        // Form F_hat
        // F1(i,a) - (ia|kk) ovoo for omega_I
        // F2(a,i) - (ai|kk) vooo for omega_J
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t k = 0; k < n_occ; k++) {

                    F1(i,a) += 2.0 * w3[(i*n_occ*n_occ*n_vir+a*n_occ*n_occ+k*n_occ+k)]
                                        - w3[(k*n_occ*n_occ*n_vir+a*n_occ*n_occ+i*n_occ+k)];
                    F2(a,i) += 2.0 * w4[(a*n_occ*n_occ*n_occ+i*n_occ*n_occ+k*n_occ+k)]
                                        - w4[(a*n_occ*n_occ*n_occ+k*n_occ*n_occ+k*n_occ+i)];

                }
                F1(i,a) += H1_a(i,a);
                F2(a,i) += H2_a(a,i);
            }
        }

        // main loop
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {

                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];

                        // t2 amplitude - vovo in moints (ai|bj)
                        double num_os = w0[(a*n_occ*n_occ*n_vir+i*n_occ*n_vir+b*n_occ+j)];
                        double num_ss = w0[(a*n_occ*n_occ*n_vir+i*n_occ*n_vir+b*n_occ+j)]
                                            - w0[(b*n_occ*n_occ*n_vir+i*n_occ*n_vir+a*n_occ+j)];
                        double t_ijab = w1[(a*n_occ*n_occ*n_vir+i*n_occ*n_vir+b*n_occ+j)] / delta_ijab;
                        double t_ijba = w1[(b*n_occ*n_occ*n_vir+i*n_occ*n_vir+a*n_occ+j)] / delta_ijab;


                        // Omega_G - (jb|ca) ovVv permute 2,4
                        for(size_t c = 0; c < n_vir; c++) {
                            G(c,i) += (2.0 * t_ijab - t_ijba) * w2[(j*n_vir*n_vir*n_vir+b*n_vir*n_vir+c*n_vir+a)];
                        }

                        // Omega_H - (jb|ik) ovoO permute 1,3
                        for(size_t k = 0; k < n_occ; k++) {
                            H(a,k) -= (2.0 * t_ijab - t_ijba) * w3[(j*n_occ*n_occ*n_vir+b*n_occ*n_occ+i*n_occ+k)];
                        }

                        // Omega_I
                        I(a,i) += (2.0 * t_ijab - t_ijba) * F1(j,b);

                        // energy calculation
                        Eost += num_os * (t1(a,i)*t1(b,j) + t_ijab);
                        Esst += num_ss * ((t1(a,i)*t1(b,j))-(t1(a,j)*t1(b,i)) + (t_ijab - t_ijba));

                    }
                }

                // Omega_J
                F2(a,i) -= ((e_orb[n_occ+a] - e_orb(i)) * t1(a,i));


            }
        }

        // Form new t1 amplitudes
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {

                double delta_ia = e_orb(i) - e_orb[n_occ+a];

                t1(a,i) = (G(a,i) + H(a,i) + I(a,i) + F2(a,i)) / delta_ia;

            }
        }

        eos += Eost;
        ess += Esst;

    }


    Eos = eos;
    Ess = 0.5 * ess;
}


template<>
void ricc2<double>::unrestricted_energy(
    double &Eos, double &Essa, double &Essb,
    const size_t n_occa, const size_t n_vira,
    const size_t n_occb, const size_t n_virb,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQov_a, Mat<double> &BQph_a,
    Mat<double> &BQoh_a, Mat<double> &BQvo_a,
    Mat<double> &BQpv_a, Mat<double> &BQov_b,
    Mat<double> &BQph_b, Mat<double> &BQoh_b,
    Mat<double> &BQvo_b, Mat<double> &BQpv_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &H1_a, Mat<double> &H2_a,
    Mat<double> &H1_b, Mat<double> &H2_b,
    Mat<double> &t1a, Mat<double> &t1b,
    Col<double> &eA, Col<double> &eB) {

    {

        // intermediates
        arma::mat Ga (n_vira, n_occa, fill::zeros);
        arma::mat Ha (n_vira, n_occa, fill::zeros);
        arma::mat Ia (n_vira, n_occa, fill::zeros);
        arma::mat F1a (n_occa, n_vira, fill::zeros);
        arma::mat F2a (n_vira, n_occa, fill::zeros);

        arma::mat Gb (n_virb, n_occb, fill::zeros);
        arma::mat Hb (n_virb, n_occb, fill::zeros);
        arma::mat Ib (n_virb, n_occb, fill::zeros);
        arma::mat F1b (n_occb, n_virb, fill::zeros);
        arma::mat F2b (n_virb, n_occb, fill::zeros);


        // Form F_hat
        // F1(i,a) - (ia|kk) ovoo for omega_I
        // F2(a,i) - (ai|kk) vooo for omega_J

        // (AA|AA), (AA|BB)
        {

            arma::Mat<double> W1 = BQoh_a.st() * BQov_a; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)
            arma::Mat<double> W2 = BQoh_a.st() * BQph_a; // J:    (VO|oO) - (ai|kk)
            arma::Mat<double> W3 = BQoh_a.st() * BQov_b; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)
            arma::Mat<double> W4 = BQoh_a.st() * BQph_b; // J:    (VO|oO) - (ai|kk)

            const double *w1 = W1.memptr();
            const double *w2 = W2.memptr();
            const double *w3 = W3.memptr();
            const double *w4 = W4.memptr();

            // (AA|AA)
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    for(size_t k = 0; k < n_occa; k++) {

                        F1a(i,a) += w1[(i*n_occa*n_occa*n_vira+a*n_occa*n_occa+k*n_occa+k)]
                                            - w1[(k*n_occa*n_occa*n_vira+a*n_occa*n_occa+i*n_occa+k)];
                        F2a(a,i) += w2[(a*n_occa*n_occa*n_occa+i*n_occa*n_occa+k*n_occa+k)]
                                            - w2[(a*n_occa*n_occa*n_occa+k*n_occa*n_occa+k*n_occa+i)];

                    }
                }
            }

            // (AA|BB)
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {
                    for(size_t k = 0; k < n_occb; k++) {

                        F1a(i,a) += w3[(i*n_occb*n_occb*n_vira+a*n_occb*n_occb+k*n_occb+k)];
                        F2a(a,i) += w4[(a*n_occb*n_occb*n_occa+i*n_occb*n_occb+k*n_occb+k)];

                    }

                    F1a(i,a) += H1_a(i,a);
                    F2a(a,i) += H2_a(a,i);

                }
            }

        }

        // (BB|BB), (BB|AA)
        {

            arma::Mat<double> W1 = BQoh_b.st() * BQov_b; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)
            arma::Mat<double> W2 = BQoh_b.st() * BQph_b; // J:    (VO|oO) - (ai|kk)
            arma::Mat<double> W3 = BQoh_b.st() * BQov_a; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)
            arma::Mat<double> W4 = BQoh_b.st() * BQph_a; // J:    (VO|oO) - (ai|kk)

            const double *w1 = W1.memptr();
            const double *w2 = W2.memptr();
            const double *w3 = W3.memptr();
            const double *w4 = W4.memptr();

            // (BB|BB)
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    for(size_t k = 0; k < n_occb; k++) {

                        F1b(i,a) += w1[(i*n_occb*n_occb*n_virb+a*n_occb*n_occb+k*n_occb+k)]
                                            - w1[(k*n_occb*n_occb*n_virb+a*n_occb*n_occb+i*n_occb+k)];
                        F2b(a,i) += w2[(a*n_occb*n_occb*n_occb+i*n_occb*n_occb+k*n_occb+k)]
                                            - w2[(a*n_occb*n_occb*n_occb+k*n_occb*n_occb+k*n_occb+i)];

                    }
                }
            }
            // (BB|AA)
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {
                    for(size_t k = 0; k < n_occa; k++) {

                        F1b(i,a) += w3[(i*n_occa*n_occa*n_virb+a*n_occa*n_occa+k*n_occa+k)];
                        F2b(a,i) += w4[(a*n_occa*n_occa*n_occb+i*n_occa*n_occa+k*n_occa+k)];

                    }

                    F1b(i,a) += H1_b(i,a);
                    F2b(a,i) += H2_b(a,i);
                }
            }

        }


        // Form the intermediates G, H, I for (AA|BB), (BB|AA), (AA|AA), (BB|BB)

        // (AA|BB)
        double eosa = 0.0;
        {
            double Eost=0.0;
            double delta_AB=0.0;
            double num_os=0.0;
            double t2ab=0.0;

            arma::Mat<double> W0 = BQvo_a.st() * BQvo_b; // Ene:  (vo|vo)
            arma::Mat<double> W1 = BQph_a.st() * BQph_b; // t2:   (VO|VO)
            arma::Mat<double> W2 = BQpv_b.st() * BQov_a; // G:    (ov|Vv) - (jb|ca)
            arma::Mat<double> W3 = BQoh_b.st() * BQov_a; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)

            const double *w0 = W0.memptr();
            const double *w1 = W1.memptr();
            const double *w2 = W2.memptr();
            const double *w3 = W3.memptr();

            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {

                            //denominator
                            delta_AB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];

                            // t2 amplitude (ai|bj) AABB
                            num_os = w0[(a*n_occa*n_occb*n_virb+i*n_occb*n_virb+b*n_occb+j)];
                            t2ab = w1[(a*n_occa*n_occb*n_virb+i*n_occb*n_virb+b*n_occb+j)] / delta_AB;

                            // Omega_G - (jb|ca) ovVv permute 2,4
                            for(size_t c = 0; c < n_vira; c++) {
                                Ga(c,i) += t2ab * w2[(j*n_virb*n_vira*n_vira+b*n_vira*n_vira+c*n_vira+a)];
                            }

                            // Omega_H - (jb|ik) ovoO permute 1,3
                            for(size_t k = 0; k < n_occa; k++) {
                                Ha(a,k) -= t2ab * w3[(j*n_occa*n_occa*n_virb+b*n_occa*n_occa+i*n_occa+k)];
                            }

                            // Omega_I
                            Ia(a,i) += t2ab * F1b(j,b);

                            // energy calculation
                            Eost += num_os * (t1a(a,i)*t1b(b,j) + t2ab);

                        }
                    }
                }
            }

            eosa += Eost;

        }


        // (BB|AA)
        {
            double delta_BA=0.0;
            double num_os=0.0;
            double t2ba=0.0;

            arma::Mat<double> W1 = BQph_a.st() * BQph_b; // t2:   (VO|VO)
            arma::Mat<double> W2 = BQpv_a.st() * BQov_b; // G:    (ov|Vv) - (jb|ca)
            arma::Mat<double> W3 = BQoh_a.st() * BQov_b; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)

            const double *w1 = W1.memptr();
            const double *w2 = W2.memptr();
            const double *w3 = W3.memptr();

            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {

                            //denominator
                            delta_BA = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];

                            // t2 amplitude BBAA
                            t2ba = w1[(a*n_occb*n_occa*n_vira+i*n_occa*n_vira+b*n_occa+j)] / delta_BA;
                            //t2ba = w1[(b*n_occb*n_virb*n_occa+j*n_occb*n_virb+a*n_occb+i)] / delta_BA;

                            // Omega_G - (jb|ca) ovVv permute 2,4
                            for(size_t c = 0; c < n_virb; c++) {
                                Gb(c,i) + t2ba * w2[(j*n_vira*n_virb*n_virb+b*n_virb*n_virb+c*n_virb+a)];
                            }

                            // Omega_H - (jb|ik) ovoO permute 1,3
                            for(size_t k = 0; k < n_occb; k++) {
                                Hb(a,k) -= t2ba * w3[(j*n_occb*n_occb*n_vira+b*n_occb*n_occb+i*n_occb+k)];
                            }

                            // Omega_I
                            Ib(a,i) += t2ba * F1a(j,b);

                        }
                    }
                }
            }
        }

        Eos = eosa;


        // (AA|AA)
        Essa = 0.0;
        {
            double Esst=0.0;
            double delta_AA=0.0;
            double delta_A=0.0;
            double delta_AB=0.0;
            double num_ss=0.0;
            double t2aa=0.0, t2aa_2=0.0;

            arma::Mat<double> W0 = BQvo_a.st() * BQvo_a; // Ene:  (vo|vo)
            arma::Mat<double> W1 = BQph_a.st() * BQph_a; // t2:   (VO|VO)
            arma::Mat<double> W2 = BQpv_a.st() * BQov_a; // G:    (ov|Vv) - (jb|ca)
            arma::Mat<double> W3 = BQoh_a.st() * BQov_a; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)

            const double *w0 = W0.memptr();
            const double *w1 = W1.memptr();
            const double *w2 = W2.memptr();
            const double *w3 = W3.memptr();


            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {

                            //denominator
                            delta_AA = eA(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];

                            // t2 amplitude (not stored)
                            num_ss = w0[(a*n_occa*n_occa*n_vira+i*n_occa*n_vira+b*n_occa+j)]
                                        - w0[(b*n_occa*n_occa*n_vira+i*n_occa*n_vira+a*n_occa+j)];
                            t2aa = w1[(a*n_occa*n_occa*n_vira+i*n_occa*n_vira+b*n_occa+j)] / delta_AA;
                            t2aa_2 = w1[(b*n_occa*n_occa*n_vira+i*n_occa*n_vira+a*n_occa+j)] / delta_AA;

                            // Omega_G - (jb|ca) ovVv permute 2,4
                            for(size_t c = 0; c < n_vira; c++) {
                                Ga(c,i) += (t2aa - t2aa_2) * w2[(j*n_vira*n_vira*n_vira+b*n_vira*n_vira+c*n_vira+a)];
                            }

                            // Omega_H - (jb|ik) ovoO permute 1,3
                            for(size_t k = 0; k < n_occa; k++) {
                                Ha(a,k) -= (t2aa - t2aa_2) * w3[(j*n_occa*n_occa*n_vira+b*n_occa*n_occa+i*n_occa+k)];
                            }

                            // Omega_I
                            Ia(a,i) += (t2aa - t2aa_2) * F1a(j,b);

                            // energy calculation
                            Esst += num_ss * ((t1a(a,i)*t1a(b,j))-(t1a(a,j)*t1a(b,i)) + (t2aa - t2aa_2));

                        }
                    }

                    // Omega_J
                    F2a(a,i) -= ((eA[n_occa+a] - eA(i)) * t1a(a,i));

                }
            }

            // Form new t1 amplitudes
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {

                    delta_A = eA(i) - eA[n_occa+a];
                    t1a(a,i) = (Ga(a,i) + Ha(a,i) + Ia(a,i) + F2a(a,i)) / delta_A;

                }
            }

            Essa+=Esst;

        }
        Essa *= 0.25;


        // (BB|BB)
        Essb = 0.0;
        {
            double Esst=0.0;
            double delta_BB=0.0;
            double delta_B=0.0;
            double delta_BA=0.0;
            double num_ss = 0.0;
            double t2bb=0.0, t2bb_2=0.0;

            arma::Mat<double> W0 = BQvo_b.st() * BQvo_b; // Ene:  (vo|vo)
            arma::Mat<double> W1 = BQph_b.st() * BQph_b; // t2:   (VO|VO)
            arma::Mat<double> W2 = BQpv_b.st() * BQov_b; // G:    (ov|Vv) - (jb|ca)
            arma::Mat<double> W3 = BQoh_b.st() * BQov_b; // H/I:  (ov|oO) - (jb|ik)/(ia|kk)

            const double *w0 = W0.memptr();
            const double *w1 = W1.memptr();
            const double *w2 = W2.memptr();
            const double *w3 = W3.memptr();


            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {

                            //denominator
                            delta_BB = eB(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];

                            // t2 amplitude (not stored)
                            num_ss = w0[(a*n_occb*n_occb*n_virb+i*n_occb*n_virb+b*n_occb+j)]
                                                - w0[(b*n_occb*n_occb*n_virb+i*n_occb*n_virb+a*n_occb+j)];
                            t2bb = w1[(a*n_occb*n_occb*n_virb+i*n_occb*n_virb+b*n_occb+j)] / delta_BB;
                            t2bb_2 = w1[(b*n_occb*n_occb*n_virb+i*n_occb*n_virb+a*n_occb+j)] / delta_BB;

                            // Omega_G - (jb|ca) ovVv permute 2,4
                            for(size_t c = 0; c < n_virb; c++) {
                                Gb(c,i) += (t2bb - t2bb_2) * w2[(j*n_virb*n_virb*n_virb+b*n_virb*n_virb+c*n_virb+a)];
                            }

                            // Omega_H - (jb|ik) ovoO permute 1,3
                            for(size_t k = 0; k < n_occb; k++) {
                                Hb(a,k) -= (t2bb - t2bb_2) * w3[(j*n_occb*n_occb*n_virb+b*n_occb*n_occb+i*n_occb+k)];
                            }

                            // Omega_I
                            Ib(a,i) += (t2bb - t2bb_2) * F1b(j,b);

                            // energy calculation
                            Esst += num_ss * ((t1b(a,i)*t1b(b,j))-(t1b(a,j)*t1b(b,i)) + (t2bb - t2bb_2));

                        }
                    }

                    // Omega_J
                    F2b(a,i) -= ((eB[n_occb+a] - eB(i)) * t1b(a,i));

                }
            }

            // Form new t1 amplitudes
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {

                    delta_B = eB(i) - eB[n_occb+a];
                    t1b(a,i) = (Gb(a,i) + Hb(a,i) + Ib(a,i) + F2b(a,i)) / delta_B;

                }
            }

            Essb+=Esst;

        }
        Essb *= 0.25;

    }

}


/// GPP: RI-CC2 calculation Haettig's algorithm
/// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
/// GPP: THIS IS NOT THE OPTIMIZED CODE
template<>
void ricc2<double>::restricted_energy(
    double &Eos, double &Ess,
    const size_t n_occ, const size_t n_vir,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQvo_a, Mat<double> &BQov_a,
    Mat<double> &BQhp_a, Mat<double> &BQoo_a,
    Mat<double> &BQoh_a,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA,
    Mat<double> &CoccA, Mat<double> &CvirtA,
    Mat<double> &t1, Col<double> &e_orb,
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

    double Eost=0.0, Esst=0.0;

    cout << "n_occ: " << n_occ << endl;
    cout << "n_vir: " << n_vir << endl;
    cout << "n_aux: " << n_aux << endl;
    cout << "n_orb: " << n_orb << endl;

    {
        //QTimer tim;
        //tim.On();

        /// step 3: form i^Q and F_ia
        arma::vec iQ (n_aux, fill::zeros);
        for(size_t k = 0; k < n_occ; k++) {
            for(size_t c = 0; c < n_vir; c++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                    iQ(Q) += BQov_a[(k*n_vir*n_aux+c*n_aux+Q)] * t1(c,k);
                }
            }
        }

        arma::Mat<double> F_hat (n_occ, n_vir, fill::zeros);
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t Q = 0; Q < n_aux; Q++) {

                    F_hat(i,a) += 2.0 * iQ(Q) * BQov_a[(i*n_vir*n_aux+a*n_aux+Q)];

                }

                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            F_hat(i,a) -= BQov_a[(j*n_vir*n_aux+a*n_aux+Q)]
                                            * BQov_a[(i*n_vir*n_aux+b*n_aux+Q)]*t1(b,j);

                        }
                    }
                }
            }
        }

        //tim.Off();
        //tim.Print("Step 3 Time:");


        //tim.On();

        /// step 4:
        arma::Mat<double> omega_I (n_vir, n_occ, fill::zeros);
        arma::Mat<double> Y (n_aux, n_vir*n_occ, fill::zeros);
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {

                        //denominator
                        double delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];

                        double num_os = 0.0;
                        double num_ss = 0.0;
                        double t_ijab = 0.0;
                        double t_ijba = 0.0;

                        for(size_t Q = 0; Q < n_aux; Q++) {

                            num_os += BQvo_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQvo_a[(b*n_occ*n_aux+j*n_aux+Q)];
                            num_ss += BQvo_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQvo_a[(b*n_occ*n_aux+j*n_aux+Q)]
                                        - BQvo_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQvo_a[(a*n_occ*n_aux+j*n_aux+Q)];

                            t_ijab += BQph_a[(a*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occ*n_aux+j*n_aux+Q)];
                            t_ijba += BQph_a[(b*n_occ*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occ*n_aux+j*n_aux+Q)];

                        }

                        t_ijab = t_ijab / delta_ijab;
                        t_ijba = t_ijba / delta_ijab;


                        // energy calculation
                        Eost += num_os * (t1(a,i)*t1(b,j) + t_ijab);
                        Esst += num_ss * ((t1(a,i)*t1(b,j))-(t1(a,j)*t1(b,i)) + (t_ijab - t_ijba));

                        // Omega_H:
                        for(size_t P = 0; P < n_aux; P++) {
                            Y[(a*n_occ*n_aux+i*n_aux+P)] += (2.0 * t_ijab - t_ijba) * BQov_a[(j*n_vir*n_aux+b*n_aux+P)];
                        }

                        // Omega_I
                        omega_I(a,i) += (2.0 * t_ijab - t_ijba) * F_hat(j,b);

                    }
                }
            }
        }

        //tim.Off();
        //tim.Print("Step 4 Time:");


        //tim.On();

        /// step 5:
        arma::mat gamma_Q (n_aux, n_orb*n_occ, fill::zeros);
        for(size_t beta = 0; beta < n_orb; beta++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t Q = 0; Q < n_aux; Q++) {
                    for(size_t a = 0; a < n_vir; a++) {

                        // omega_G1
                        gamma_Q[(i*n_orb*n_aux+beta*n_aux+Q)] += Y[(a*n_occ*n_aux+i*n_aux+Q)] * CvirtA(beta,a);

                        for(size_t k = 0; k < n_occ; k++) {

                            // omega_G2
                            gamma_Q[(i*n_orb*n_aux+beta*n_aux+Q)] -= CvirtA(beta,a) * t1(a,k)
                                                                        * BQoh_a[(k*n_occ*n_aux+i*n_aux+Q)];

                        }

                    }

                    // omega_J
                    gamma_Q[(i*n_orb*n_aux+beta*n_aux+Q)] += 2.0 * Lam_hA(beta,i) * iQ(Q);

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
        int count = 0;
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

        //tim.Off();
        //tim.Print("Step 5 Time:");



        //tim.On();

        /// step 6:
        arma::Mat<double> omega_H (n_vir, n_occ, fill::zeros);
        arma::Mat<double> omega_JG (n_vir, n_occ, fill::zeros);

        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {

                // omega_H
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        omega_H(a,i) += Y[(a*n_occ*n_aux+k*n_aux+P)]
                                            * BQoh_a[(k*n_occ*n_aux+i*n_aux+P)];
                    }
                }

                // omega_JG
                for(size_t z = 0; z < n_orb; z++) {
                    omega_JG(a,i) += Lam_pA(z,a) * JG(z,i);
                }

                double delta_ia = e_orb(i) - e_orb[n_occ+a];

                t1(a,i) = (omega_JG(a,i) - omega_H(a,i) + omega_I(a,i)) / delta_ia;

            }
        }


        //tim.Off();
        //tim.Print("Step 6 Time:");


    }

    Eos = Eost;
    Ess = 0.5 * Esst;

}


template<>
void ricc2<double>::unrestricted_energy(
    double &Eos, double &Essa, double &Essb,
    const size_t n_occa, const size_t n_vira,
    const size_t n_occb, const size_t n_virb,
    const size_t n_aux, const size_t n_orb,
    Mat<double> &BQov_a, Mat<double> &BQph_a, Mat<double> &BQoh_a,
    Mat<double> &BQov_b, Mat<double> &BQph_b, Mat<double> &BQoh_b,
    Mat<double> &Lam_hA, Mat<double> &Lam_pA, Mat<double> &Lam_hB, Mat<double> &Lam_pB,
    Mat<double> &CoccA, Mat<double> &CvirtA, Mat<double> &CoccB, Mat<double> &CvirtB,
    Mat<double> &t1a, Mat<double> &t1b, Col<double> &eA, Col<double> &eB,
    array_view<double> av_buff_ao,
    array_view<double> av_pqinvhalf,
    const libqints::dev_omp &m_dev,
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3) {

    memory_pool<double> mem(av_buff_ao);
    typename memory_pool<double>::checkpoint chkpt = mem.save_state();

    // intermediates
    arma::vec iQ_a (n_aux, fill::zeros);
    arma::mat F_hat_a (n_occa, n_vira, fill::zeros);
    arma::mat omega_H_a (n_vira, n_occa, fill::zeros);
    arma::mat omega_I_a (n_vira, n_occa, fill::zeros);
    arma::mat Y_a (n_aux, n_vira*n_occa, fill::zeros);
    arma::mat gamma_Q_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::mat gamma_P_a (n_aux, n_orb*n_occa, fill::zeros);
    arma::mat JG_a (n_orb, n_occa, fill::zeros);
    arma::mat omega_JG_a (n_vira, n_occa, fill::zeros);

    arma::vec iQ_b (n_aux, fill::zeros);
    arma::mat F_hat_b (n_occb, n_virb, fill::zeros);
    arma::mat omega_H_b (n_virb, n_occb, fill::zeros);
    arma::mat omega_I_b (n_virb, n_occb, fill::zeros);
    arma::mat Y_b (n_aux, n_virb*n_occb, fill::zeros);
    arma::mat gamma_Q_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::mat gamma_P_b (n_aux, n_orb*n_occb, fill::zeros);
    arma::mat JG_b (n_orb, n_occb, fill::zeros);
    arma::mat omega_JG_b (n_virb, n_occb, fill::zeros);

    {

        /// step 3:  form i^Q and F_ia


        // form i^Q
        // (AA|AA)
        for(size_t k = 0; k < n_occa; k++) {
            for(size_t c = 0; c < n_vira; c++) {
                for(size_t Q = 0; Q < n_aux; Q++) {

                    iQ_a(Q) += BQov_a[(k*n_vira*n_aux+c*n_aux+Q)] * t1a(c,k);

                }
            }
        }
        // (BB|BB)
        for(size_t k = 0; k < n_occb; k++) {
            for(size_t c = 0; c < n_virb; c++) {
                for(size_t Q = 0; Q < n_aux; Q++) {

                    iQ_b(Q) += BQov_b[(k*n_virb*n_aux+c*n_aux+Q)] * t1b(c,k);

                }
            }
        }

        // Form F_ia
        // (AA|AA), (AA|BB)
        for(size_t a = 0; a < n_vira; a++) {
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t Q = 0; Q < n_aux; Q++) {

                    F_hat_a(i,a) += iQ_a(Q) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)]
                                        + iQ_b(Q) * BQov_a[(i*n_vira*n_aux+a*n_aux+Q)];

                }

                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            F_hat_a(i,a) -= BQov_a[(j*n_vira*n_aux+a*n_aux+Q)]
                                                * BQov_a[(i*n_vira*n_aux+b*n_aux+Q)]*t1a(b,j);

                        }
                    }
                }

            }
        }

        // (BB|BB), (BB|AA)
        for(size_t a = 0; a < n_virb; a++) {
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t Q = 0; Q < n_aux; Q++) {

                    F_hat_b(i,a) += iQ_b(Q) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)]
                                        + iQ_a(Q) * BQov_b[(i*n_virb*n_aux+a*n_aux+Q)];

                }

                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        for(size_t Q = 0; Q < n_aux; Q++) {

                            F_hat_b(i,a) -= BQov_b[(j*n_virb*n_aux+a*n_aux+Q)]
                                                * BQov_b[(i*n_virb*n_aux+b*n_aux+Q)]*t1b(b,j);

                        }
                    }
                }

            }
        }




        ///step 4:


        // (AA|BB)
        double Eosa = 0.0;
        {
            double Eost = 0.0;
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {

                            //denominator
                            double delta_AB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];

                            double num_os = 0.0;
                            double t2ab = 0.0;

                            for(size_t Q = 0; Q < n_aux; Q++) {

                                num_os += BQov_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQov_b[(j*n_virb*n_aux+b*n_aux+Q)];
                                t2ab += BQph_a[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)];

                            }

                            t2ab = t2ab / delta_AB;


                            // energy calculation
                            Eost += num_os * (t1a(a,i)*t1b(b,j) + t2ab);


                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                Y_a[(a*n_occa*n_aux+i*n_aux+P)] += t2ab * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            omega_I_a(a,i) += t2ab * F_hat_b(j,b);

                        }
                    }
                }
            }

            Eosa += Eost;

        }

        // (BB|AA)
        double Eosb = 0.0;
        {
            double Eost = 0.0;
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {

                            //denominator
                            double delta_BA = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];

                            double num_os = 0.0;
                            double t2ba = 0.0;

                            for(size_t Q = 0; Q < n_aux; Q++) {

                                num_os += BQov_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQov_a[(j*n_vira*n_aux+b*n_aux+Q)];
                                t2ba += BQph_b[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)];

                            }

                            t2ba = t2ba / delta_BA;


                            // energy calculation
                            Eost += num_os * (t1b(a,i)*t1a(b,j) + t2ba);


                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                Y_b[(a*n_occb*n_aux+i*n_aux+P)] += t2ba * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            omega_I_b(a,i) += t2ba * F_hat_a(j,b);

                        }
                    }
                }
            }

            Eosb += Eost;

        }

        Eos = 0.5*Eosa + 0.5*Eosb;


        //(AA|AA)
        Essa = 0.0;
        {
            double Esst=0.0;

            for(size_t a = 0; a < n_vira; a++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t b = 0; b < n_vira; b++) {
                        for(size_t j = 0; j < n_occa; j++) {

                            //denominator
                            double delta_AA = eA(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];

                            double num_ss = 0.0;
                            double t2aa = 0.0;
                            double t2aa_2 = 0.0;

                            for(size_t Q = 0; Q < n_aux; Q++) {

                                num_ss += BQov_a[(i*n_vira*n_aux+a*n_aux+Q)]*BQov_a[(j*n_vira*n_aux+b*n_aux+Q)]
                                            - BQov_a[(i*n_vira*n_aux+b*n_aux+Q)]*BQov_a[(j*n_vira*n_aux+a*n_aux+Q)];

                                t2aa += BQph_a[(a*n_occa*n_aux+i*n_aux+Q)]*BQph_a[(b*n_occa*n_aux+j*n_aux+Q)];
                                t2aa_2 += BQph_a[(b*n_occa*n_aux+i*n_aux+Q)]*BQph_a[(a*n_occa*n_aux+j*n_aux+Q)];

                            }

                            t2aa = t2aa / delta_AA;
                            t2aa_2 = t2aa_2 / delta_AA;


                            // energy calculation
                            Esst += num_ss * ((t1a(a,i)*t1a(b,j))-(t1a(a,j)*t1a(b,i)) + (t2aa - t2aa_2));


                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                Y_a[(a*n_occa*n_aux+i*n_aux+P)] += (t2aa - t2aa_2) * BQov_a[(j*n_vira*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            omega_I_a(a,i) += (t2aa - t2aa_2) * F_hat_a(j,b);

                        }
                    }
                }
            }

            Essa+=Esst;

        }
        Essa *= 0.25;


        //(BB|BB)
        Essb = 0.0;
        {
            double Esst=0.0;

            for(size_t a = 0; a < n_virb; a++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t b = 0; b < n_virb; b++) {
                        for(size_t j = 0; j < n_occb; j++) {

                            //denominator
                            double delta_BB = eB(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];

                            double num_ss = 0.0;
                            double t2bb = 0.0;
                            double t2bb_2 = 0.0;

                            for(size_t Q = 0; Q < n_aux; Q++) {

                                num_ss += BQov_b[(i*n_virb*n_aux+a*n_aux+Q)]*BQov_b[(j*n_virb*n_aux+b*n_aux+Q)]
                                            - BQov_b[(i*n_virb*n_aux+b*n_aux+Q)]*BQov_b[(j*n_virb*n_aux+a*n_aux+Q)];

                                t2bb += BQph_b[(a*n_occb*n_aux+i*n_aux+Q)]*BQph_b[(b*n_occb*n_aux+j*n_aux+Q)];
                                t2bb_2 += BQph_b[(b*n_occb*n_aux+i*n_aux+Q)]*BQph_b[(a*n_occb*n_aux+j*n_aux+Q)];

                            }

                            t2bb = t2bb / delta_BB;
                            t2bb_2 = t2bb_2 / delta_BB;


                            // energy calculation
                            Esst += num_ss * ((t1b(a,i)*t1b(b,j))-(t1b(a,j)*t1b(b,i)) + (t2bb - t2bb_2));


                            // Omega_H:
                            for(size_t P = 0; P < n_aux; P++) {
                                Y_b[(a*n_occb*n_aux+i*n_aux+P)] += (t2bb - t2bb_2) * BQov_b[(j*n_virb*n_aux+b*n_aux+P)];
                            }

                            // Omega_I
                            omega_I_b(a,i) += (t2bb - t2bb_2) * F_hat_b(j,b);

                        }
                    }
                }
            }

            Essb+=Esst;

        }
        Essb *= 0.25;



        /// step 5:

        // V_PQ^(-1/2)
        arma::mat PQinvhalf(arrays<double>::ptr(av_pqinvhalf), n_aux, n_aux, false, true);

        // (P|ab) - original
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

        // (AA|AA), (AA|BB)
        {
            for(size_t beta = 0; beta < n_orb; beta++) {
                for(size_t i = 0; i < n_occa; i++) {
                    for(size_t Q = 0; Q < n_aux; Q++) {
                        for(size_t a = 0; a < n_vira; a++) {

                            // omega_G
                            gamma_Q_a[(i*n_orb*n_aux+beta*n_aux+Q)] += Y_a[(a*n_occa*n_aux+i*n_aux+Q)] * CvirtA(beta,a);

                            for(size_t k = 0; k < n_occa; k++) {

                                // omega_J
                                gamma_Q_a[(i*n_orb*n_aux+beta*n_aux+Q)] -=
                                            CvirtA(beta,a)*t1a(a,k)*BQoh_a[(k*n_occa*n_aux+i*n_aux+Q)];

                            }

                        }

                        // omega_J
                        gamma_Q_a[(i*n_orb*n_aux+beta*n_aux+Q)] += Lam_hA(beta,i)*iQ_a(Q) + Lam_hA(beta,i)*iQ_b(Q);

                    }
                }
            }

            gamma_P_a = PQinvhalf * gamma_Q_a;

            for(size_t i = 0; i < n_occa; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {
                            JG_a(alpha,i) += gamma_P_a[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        }
                    }
                }
            }
        } // end (AA|AA), (AA|BB)

        // (BB|BB), (BB|AA)
        {
            for(size_t beta = 0; beta < n_orb; beta++) {
                for(size_t i = 0; i < n_occb; i++) {
                    for(size_t Q = 0; Q < n_aux; Q++) {
                        for(size_t a = 0; a < n_virb; a++) {

                            // omega_G
                            gamma_Q_b[(i*n_orb*n_aux+beta*n_aux+Q)] += Y_b[(a*n_occb*n_aux+i*n_aux+Q)] * CvirtB(beta,a);

                            for(size_t k = 0; k < n_occb; k++) {

                                // omega_J
                                gamma_Q_b[(i*n_orb*n_aux+beta*n_aux+Q)] -=
                                            CvirtB(beta,a)*t1b(a,k)*BQoh_b[(k*n_occb*n_aux+i*n_aux+Q)];

                            }

                        }

                        // omega_J
                        gamma_Q_b[(i*n_orb*n_aux+beta*n_aux+Q)] += Lam_hB(beta,i)*iQ_b(Q) + Lam_hB(beta,i)*iQ_a(Q);

                    }
                }
            }

            gamma_P_b = PQinvhalf * gamma_Q_b;

            for(size_t i = 0; i < n_occb; i++) {
                for(size_t P = 0; P < n_aux; P++) {
                    for(size_t beta = 0; beta < n_orb; beta++) {
                        for(size_t alpha = 0; alpha < n_orb; alpha++) {
                            JG_b(alpha,i) += gamma_P_b[(i*n_orb*n_aux+beta*n_aux+P)]
                                                * V_Pab[(P*n_orb*n_orb+alpha*n_orb+beta)];
                        }
                    }
                }
            }
        } // end (BB|BB), (BB|AA)


        /// step 6:

        // (AA|AA)
        {
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t a = 0; a < n_vira; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occa; k++) {
                            omega_H_a(a,i) += Y_a[(a*n_occa*n_aux+k*n_aux+P)]
                                                    * BQoh_a[(k*n_occa*n_aux+i*n_aux+P)];
                        }
                    }

                    // omega_JG
                    for(size_t z = 0; z < n_orb; z++) {
                        omega_JG_a(a,i) += Lam_pA(z,a) * JG_a(z,i);
                    }

                    double delta_A = eA(i) - eA[n_occa+a];

                    t1a(a,i) = (omega_JG_a(a,i) - omega_H_a(a,i) + omega_I_a(a,i)) / delta_A;

                }
            }
        } // end (AA|AA)


        // (BB|BB)
        {
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t a = 0; a < n_virb; a++) {

                    // omega_H
                    for(size_t P = 0; P < n_aux; P++) {
                        for(size_t k = 0; k < n_occb; k++) {
                            omega_H_b(a,i) += Y_b[(a*n_occb*n_aux+k*n_aux+P)]
                                                    * BQoh_b[(k*n_occb*n_aux+i*n_aux+P)];
                        }
                    }

                    // omega_JG
                    for(size_t z = 0; z < n_orb; z++) {
                        omega_JG_b(a,i) += Lam_pB(z,a) * JG_b(z,i);
                    }

                    double delta_B = eB(i) - eB[n_occb+a];

                    t1b(a,i) = (omega_JG_b(a,i) - omega_H_b(a,i) + omega_I_b(a,i)) / delta_B;

                }
            }
        } // end (BB|BB)

    }

}
#endif



template class ricc2<double,double>;
template class ricc2<complex<double>,double>;
template class ricc2<complex<double>,complex<double>>;

} // end libgmbpt
