#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <armadillo>
#include "cc2.h"

#include<complex>
using namespace std;

namespace libgmbpt{

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
void cc2<double>::restricted_energy(
    double &Eos, double &Ess,
    const size_t& n_occ, const size_t& n_vir,
    Mat<double> &V_vovo_u, Mat<double> &V_vovo,
    Mat<double> &V_ovvv, Mat<double> &V_ovoo,
    Mat<double> &V_vooo,
    Mat<double> &H1_a, Mat<double> &H2_a,
    Mat<double> &t1, Col<double> &e_orb,
    double c_os, double c_ss) {
    
    double eos = 0.0, ess = 0.0;
    
    {

        double Eost=0.0;
        double Esst=0.0;
        double delta_ijab=0.0, delta_ia=0.0;
        double num_os=0.0, num_ss=0.0;
        double t_ijab=0.0, t_ijba=0.0;
        
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
        
        // intermediates
        arma::mat G (n_vir, n_occ, fill::zeros);
        arma::mat H (n_vir, n_occ, fill::zeros);
        arma::mat I (n_vir, n_occ, fill::zeros);
        arma::mat F1 (n_occ, n_vir, fill::zeros);
        arma::mat F2 (n_vir, n_occ, fill::zeros);
        
        // integrals
        const double *w0 = V_vovo_u.memptr();
        const double *w1 = V_vovo.memptr();
        const double *w2 = V_ovvv.memptr();
        const double *w3 = V_ovoo.memptr();
        const double *w4 = V_vooo.memptr();
        
        // Form F_hat
        // F1(i,a) - (ia|kk) ovoo for omega_I
        // F2(a,i) - (ai|kk) vooo for omega_J
        #pragma omp parallel
        {
            arma::mat F1_local (n_occ, n_vir, fill::zeros);
            arma::mat F2_local (n_vir, n_occ, fill::zeros);
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t a = 0; a < n_vir; a++) {
                    for(size_t k = 0; k < n_occ; k++) {
                        // F1(i,a) += 2.0 * w3[(i*oov+a*oo+k*o+k)] - w3[(k*oov+a*oo+i*o+k)];
                        // F2(a,i) += 2.0 * w4[(a*ooo+i*oo+k*o+k)] - w4[(a*ooo+k*oo+k*o+i)];
                        F1_local(i,a) += 2.0 * w3[(i*oov+a*oo+k*o+k)] - w3[(k*oov+a*oo+i*o+k)];
                        F2_local(a,i) += 2.0 * w4[(a*ooo+i*oo+k*o+k)] - w4[(a*ooo+k*oo+k*o+i)];
                    }
                    // F1(i,a) += H1_a(i,a);
                    // F2(a,i) += H2_a(a,i);
                    F1_local(i,a) += H1_a(i,a);
                    F2_local(a,i) += H2_a(a,i);
                }
            }
            #pragma omp critical (F)
            {
                F1 += F1_local;
                F2 += F2_local;
            }
        }
        
        
        #pragma omp parallel
        {
	        double Eost_local = 0.0, Esst_local = 0.0; 
            arma::mat G_local (n_vir, n_occ, fill::zeros);
            arma::mat H_local (n_vir, n_occ, fill::zeros);
            arma::mat I_local (n_vir, n_occ, fill::zeros);
            arma::mat F2_local (n_vir, n_occ, fill::zeros);
            #pragma omp for
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t i = 0; i < n_occ; i++) {
                    for(size_t b = 0; b < n_vir; b++) {
                        for(size_t j = 0; j < n_occ; j++) {
                            
                            //denominator
                            delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];
                            
                            // t2 amplitude (not stored)
                            // vovo in moints (ai|bj)
                            num_os = w0[(a*oov+i*ov+b*o+j)];
                            num_ss = w0[(a*oov+i*ov+b*o+j)] - w0[(b*oov+i*ov+a*o+j)];
                            t_ijab = w1[(a*oov+i*ov+b*o+j)] / delta_ijab;
                            t_ijba = w1[(b*oov+i*ov+a*o+j)] / delta_ijab;
                        
                            // Omega_G - (jb|ca) ovvv permute 2,4
                            for(size_t c = 0; c < n_vir; c++) {
                                // G(c,i) += (2.0 * t_ijab - t_ijba) * w2[(j*vvv+b*vv+c*v+a)];
                                // G(c,i) += c_os * t_ijab * w2[(j*vvv+b*vv+c*v+a)]
                                //         + c_ss * (t_ijab - t_ijba) * w2[(j*vvv+b*vv+c*v+a)];
                                G_local(c,i) += c_os * t_ijab * w2[(j*vvv+b*vv+c*v+a)]
                                                + c_ss * (t_ijab - t_ijba) * w2[(j*vvv+b*vv+c*v+a)];
                            }
                            
                            // Omega_H - (jb|ik) ovoo permute 1,3
                            for(size_t k = 0; k < n_occ; k++) {
                                // H(a,k) -= (2.0 * t_ijab - t_ijba) * w3[(j*oov+b*oo+i*o+k)];
                                // H(a,k) -= c_os * t_ijab * w3[(j*oov+b*oo+i*o+k)]
                                //         + c_ss * (t_ijab - t_ijba) * w3[(j*oov+b*oo+i*o+k)];
                                H_local(a,k) -= c_os * t_ijab * w3[(j*oov+b*oo+i*o+k)]
                                                + c_ss * (t_ijab - t_ijba) * w3[(j*oov+b*oo+i*o+k)];
                            }

                            // Omega_I
                            // I(a,i) += (2.0 * t_ijab - t_ijba) * F1(j,b);
                            // I(a,i) += c_os * t_ijab * F1(j,b)
                            //         + c_ss * (t_ijab - t_ijba) * F1(j,b);
                            I_local(a,i) += c_os * t_ijab * F1(j,b)
                                            + c_ss * (t_ijab - t_ijba) * F1(j,b);
                        
                            
                            // energy calculation
                            // Eost += num_os * (t1(a,i)*t1(b,j) + c_os*t_ijab);
                            // Esst += num_ss * ((t1(a,i)*t1(b,j))-(t1(a,j)*t1(b,i)) + c_ss*(t_ijab - t_ijba));
                            Eost_local += num_os * (t1(a,i)*t1(b,j) + c_os*t_ijab);
                            Esst_local += num_ss * ((t1(a,i)*t1(b,j))-(t1(a,j)*t1(b,i)) + c_ss*(t_ijab - t_ijba));
        
                        }
                    }
                    // Omega_J
                    // F2(a,i) -= ((e_orb[n_occ+a] - e_orb(i)) * t1(a,i));
                    F2_local(a,i) -= ((e_orb[n_occ+a] - e_orb(i)) * t1(a,i));
                }
            }
            #pragma omp critical (GHI)
            {
                G += G_local;
                H += H_local;
                I += I_local;
                F2 += F2_local;
	            Eost += Eost_local;
	            Esst += Esst_local;
            }
        }
        
        // Form new t1 amplitudes
        #pragma omp parallel
        {
            #pragma omp for
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t a = 0; a < n_vir; a++) {
                    
                    delta_ia = e_orb(i) - e_orb[n_occ+a];
                    
                    t1(a,i) = (G(a,i) + H(a,i) + I(a,i) + F2(a,i)) / delta_ia;
                    
                }
            }
        }
        
        eos += Eost;
        ess += Esst;
        
    }
    
    Eos = eos;
    Ess = 0.5 * ess;    // Same spin needs to be scaled in the end
}

    
template<>
void cc2<complex<double> >::restricted_energy(
    complex<double>& Eos, complex<double>& Ess,
    const size_t& n_occ, const size_t& n_vir,
    Mat<complex<double> > &V_vovo_u,
    Mat<complex<double> > &V_vovo,
    Mat<complex<double> > &V_ovvv,
    Mat<complex<double> > &V_ovoo,
    Mat<complex<double> > &V_vooo,
    Mat<complex<double> > &H1_a,
    Mat<complex<double> > &H2_a,
    Mat<complex<double> > &t1,
    Col<complex<double>> &e_orb,
    double c_os, double c_ss){

    complex<double> eos(0.,0.), ess(0.,0.);

    {

        complex<double> Eost(0.,0.), Esst(0.,0.);
        complex<double> delta_ijab(0.,0.), delta_ia(0.,0.);
        complex<double> num_os(0.,0.), num_ss(0.,0.);
        complex<double> t_ijab(0.,0.), t_ijba(0.,0.);

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

        // intermediates
        arma::cx_mat G (n_vir, n_occ, fill::zeros);
        arma::cx_mat H (n_vir, n_occ, fill::zeros);
        arma::cx_mat I (n_vir, n_occ, fill::zeros);
        arma::cx_mat F1 (n_occ, n_vir, fill::zeros);
        arma::cx_mat F2 (n_vir, n_occ, fill::zeros);

        // integrals
        const complex<double> *w0 = V_vovo_u.memptr();
        const complex<double> *w1 = V_vovo.memptr();
        const complex<double> *w2 = V_ovvv.memptr();
        const complex<double> *w3 = V_ovoo.memptr();
        const complex<double> *w4 = V_vooo.memptr();

        // Form F_hat
        // F1(i,a) - (ia|kk) ovoo for omega_I
        // F2(a,i) - (ai|kk) vooo for omega_J
        for(size_t i = 0; i < n_occ; i++) {
            for(size_t a = 0; a < n_vir; a++) {
                for(size_t k = 0; k < n_occ; k++) {
                    F1(i,a) += 2.0 * w3[(i*oov+a*oo+k*o+k)] - w3[(k*oov+a*oo+i*o+k)];
                    F2(a,i) += 2.0 * w4[(a*ooo+i*oo+k*o+k)] - w4[(a*ooo+k*oo+k*o+i)];
                }
                F1(i,a) += H1_a(i,a);
                F2(a,i) += H2_a(a,i);
            }
        }
 
        for(size_t a = 0; a < n_vir; a++) {
            for(size_t i = 0; i < n_occ; i++) {
                for(size_t b = 0; b < n_vir; b++) {
                    for(size_t j = 0; j < n_occ; j++) {


                            //denominator
                            delta_ijab = e_orb(i) + e_orb(j) - e_orb[n_occ+a] - e_orb[n_occ+b];

                            // t2 amplitude (not stored)
                            // vovo in moints (ai|bj)
                            num_os = w0[(a*oov+i*ov+b*o+j)];
                            num_ss = w0[(a*oov+i*ov+b*o+j)] - w0[(b*oov+i*ov+a*o+j)];
                            t_ijab = (conj(delta_ijab) * w1[(a*oov+i*ov+b*o+j)]) /
                                     (conj(delta_ijab) * delta_ijab);
                            t_ijba = (conj(delta_ijab) * w1[(b*oov+i*ov+a*o+j)]) / 
                                     (conj(delta_ijab) * delta_ijab);

                            // Omega_G - (jb|ca) ovvv permute 2,4
                            for(size_t c = 0; c < n_vir; c++) {
                                G(c,i) += (2.0 * t_ijab - t_ijba) * w2[(j*vvv+b*vv+c*v+a)];
                            }

                            // Omega_H - (jb|ik) ovoo permute 1,3
                            for(size_t k = 0; k < n_occ; k++) {
                                H(a,k) -= (2.0 * t_ijab - t_ijba) * w3[(j*oov+b*oo+i*o+k)];
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

                delta_ia = e_orb(i) - e_orb[n_occ+a];
                t1(a,i) = (conj(delta_ia)*(G(a,i) + H(a,i) + I(a,i) + F2(a,i))) / 
                          (conj(delta_ia) * delta_ia);

            }
        }

        eos += Eost;
        ess += Esst;
    }

    Eos = eos;
    Ess = 0.5 * ess;    // Same spin needs to be scaled in the end

}
    
template<>
void cc2<double>::unrestricted_energy(
    double &Eos, double &Essa, double &Essb,
    const size_t& n_occa, const size_t& n_vira, 
    const size_t& n_occb, const size_t& n_virb,
    Mat<double> &V_vovo_u_a, Mat<double> &V_vovo_u_b, Mat<double> &V_vovo_u_ab,
    Mat<double> &V_vovo_a, Mat<double> &V_vovo_b, Mat<double> &V_vovo_ab,
    Mat<double> &V_ovvv_a, Mat<double> &V_ovvv_b, Mat<double> &V_ovvv_ab, Mat<double> &V_ovvv_ba,
    Mat<double> &V_ovoo_a, Mat<double> &V_ovoo_b, Mat<double> &V_ovoo_ab, Mat<double> &V_ovoo_ba,
    Mat<double> &V_vooo_a, Mat<double> &V_vooo_b, Mat<double> &V_vooo_ab, Mat<double> &V_vooo_ba,
    Mat<double> &H1_a, Mat<double> &H2_a, Mat<double> &H1_b, Mat<double> &H2_b,
    Mat<double> &t1a, Mat<double> &t1b,
    Col<double> &eA, Col<double> &eB,
    double c_os, double c_ss){
    
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
    
    // Form F1 and F2 first for (AA|AA), (AA|BB), (BB|BB), (BB|AA)
    
    // (AA|AA), (AA|BB)
    #pragma omp parallel
    {
        // w1
        int oov = n_occa*n_occa*n_vira;
        int oo = n_occa*n_occa;
        int o = n_occa;
        //w2
        int oov_bba = n_occb*n_occb*n_vira;
        int oo_bb = n_occb*n_occb;
        int o_b = n_occb;
        
        int ooo_bba = n_occb*n_occb*n_occa;
        int ooo = n_occa*n_occa*n_occa;
        
        const double *w1 = V_ovoo_a.memptr(); //F1 AA
        const double *w2 = V_ovoo_ab.memptr(); //F1 AB
        const double *w3 = V_vooo_a.memptr(); //F2 AA
        const double *w4 = V_vooo_ab.memptr(); //F2 AB
        
        // (ia|kk) - (AA|AA)
        arma::mat F1a_local (n_occa, n_vira, fill::zeros);
        arma::mat F2a_local (n_vira, n_occa, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t k = 0; k < n_occa; k++) {
                    // F1a(i,a) += w1[(i*oov+a*oo+k*o+k)] - w1[(k*oov+a*oo+i*o+k)];
                    // F2a(a,i) += w3[(a*ooo+i*oo+k*o+k)] - w3[(a*ooo+k*oo+k*o+i)];
                    F1a_local(i,a) += w1[(i*oov+a*oo+k*o+k)] - w1[(k*oov+a*oo+i*o+k)];
                    F2a_local(a,i) += w3[(a*ooo+i*oo+k*o+k)] - w3[(a*ooo+k*oo+k*o+i)];
                }
            }
        }
        
        // (ia|kk) - (AA|BB)
        #pragma omp for
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t k = 0; k < n_occb; k++) {
                    // F1a(i,a) += w2[(i*oov_bba+a*oo_bb+k*o_b+k)];
                    // F2a(a,i) += w4[(a*ooo_bba+i*oo_bb+k*o_b+k)];
                    F1a_local(i,a) += w2[(i*oov_bba+a*oo_bb+k*o_b+k)];
                    F2a_local(a,i) += w4[(a*ooo_bba+i*oo_bb+k*o_b+k)];
                }
                F1a_local(i,a) += H1_a(i,a);
                F2a_local(a,i) += H2_a(a,i);
            }
        }
        #pragma omp critical (Fa)
        {
            F1a += F1a_local;
            F2a += F2a_local;
        }
        
    }
    
    // (BB|BB), (BB|AA)
    #pragma omp parallel
    {
        // w1
        int oov = n_occb*n_occb*n_virb;
        int oo = n_occb*n_occb;
        int o = n_occb;
        // w2
        int oov_aab = n_occa*n_occa*n_virb;
        int oo_aa = n_occa*n_occa;
        int o_a = n_occa;
        
        int ooo_aab = n_occa*n_occa*n_occb;
        int ooo = n_occb*n_occb*n_occb;
        
        const double *w1 = V_ovoo_b.memptr(); //F1 BB
        const double *w2 = V_ovoo_ba.memptr(); //F1 BA
        const double *w3 = V_vooo_b.memptr(); //F2 BB
        const double *w4 = V_vooo_ba.memptr(); //F2 BA
        
        // (ia|kk) - (BB|BB)
        arma::mat F1b_local (n_occb, n_virb, fill::zeros);
        arma::mat F2b_local (n_virb, n_occb, fill::zeros);
        #pragma omp for
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t k = 0; k < n_occb; k++) {
                    // F1b(i,a) += w1[(i*oov+a*oo+k*o+k)] - w1[(k*oov+a*oo+i*o+k)];
                    // F2b(a,i) += w3[(a*ooo+i*oo+k*o+k)] - w3[(a*ooo+k*oo+k*o+i)];
                    F1b_local(i,a) += w1[(i*oov+a*oo+k*o+k)] - w1[(k*oov+a*oo+i*o+k)];
                    F2b_local(a,i) += w3[(a*ooo+i*oo+k*o+k)] - w3[(a*ooo+k*oo+k*o+i)];
                }
            }
        }
        
        // (ia|kk) - (BB|AA)
        #pragma omp for
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t k = 0; k < n_occa; k++) {
                    // F1b(i,a) += w2[(i*oov_aab+a*oo_aa+k*o_a+k)];
                    // F2b(a,i) += w4[(a*ooo_aab+i*oo_aa+k*o_a+k)];
                    F1b_local(i,a) += w2[(i*oov_aab+a*oo_aa+k*o_a+k)];
                    F2b_local(a,i) += w4[(a*ooo_aab+i*oo_aa+k*o_a+k)];
                }
                // F1b(i,a) += H1_b(i,a);
                // F2b(a,i) += H2_b(a,i);
                F1b_local(i,a) += H1_b(i,a);
                F2b_local(a,i) += H2_b(a,i);
            }
        }
        #pragma omp critical (Fb)
        {
            F1b += F1b_local;
            F2b += F2b_local;
        }
        
    }
    
    
    // Form the intermediates G, H, I for (AA|BB), (BB|AA), (AA|AA), (BB|BB)
    
    // Alpha/Beta (AA|BB)
    double eosa = 0.0;
    
    #pragma omp parallel
    {
        double Eost=0.0;
        double delta_AB=0.0;
        double num_os=0.0;
        double t2ab=0.0;
        
        // pointers
        // w0 and w1
        int o_b = n_occb;
        int ov_bb = n_occb*n_virb;
        int ovo_bba = n_occb*n_virb*n_occa;
        // w2
        int v_a = n_vira;
        int vv_aa = n_vira*n_vira;
        int vvv_aab = n_vira*n_vira*n_virb;
        // w3
        int o_a = n_occa;
        int oo_aa = n_occa*n_occa;
        int oov_aab = n_occa*n_occa*n_virb;
        
        // integrals
        const double *w0 = V_vovo_u_ab.memptr(); //num_os
        const double *w1 = V_vovo_ab.memptr(); //t2ab
        const double *w2 = V_ovvv_ba.memptr(); //Ga
        const double *w3 = V_ovoo_ba.memptr(); //Ha

        double Eost_local = 0.0;
        arma::mat Ga_local (n_vira, n_occa, fill::zeros);
        arma::mat Ha_local (n_vira, n_occa, fill::zeros);
        arma::mat Ia_local (n_vira, n_occa, fill::zeros);
        #pragma omp for
        for(size_t a = 0; a < n_vira; a++) {
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        
                        //denominator
                        delta_AB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];
                        
                        // t2 amplitude (ai|bj) AABB
                        num_os = w0[(a*ovo_bba+i*ov_bb+b*o_b+j)];
                        t2ab = w1[(a*ovo_bba+i*ov_bb+b*o_b+j)] / delta_AB;
                    
                        // (jb|ca) BBAA
                        for(size_t c = 0; c < n_vira; c++) {
                            // Ga(c,i) += c_os * t2ab * w2[(j*vvv_aab+b*vv_aa+c*v_a+a)];
                            Ga_local(c,i) += c_os * t2ab * w2[(j*vvv_aab+b*vv_aa+c*v_a+a)];
                        }
                        
                        // (jb|ik) BBAA
                        for(size_t k = 0; k < n_occa; k++) {
                            // Ha(a,k) -= c_os * t2ab * w3[(j*oov_aab+b*oo_aa+i*o_a+k)];
                            Ha_local(a,k) -= c_os * t2ab * w3[(j*oov_aab+b*oo_aa+i*o_a+k)];
                        }
                        
                        // Ia(a,i) += c_os * t2ab * F1b(j,b);
                        Ia_local(a,i) += c_os * t2ab * F1b(j,b);
                        
                        // energy calculation
                        // Eost += num_os * (t1a(a,i)*t1b(b,j) + c_os * t2ab);
                        Eost_local += num_os * (t1a(a,i)*t1b(b,j) + c_os * t2ab);
                        
                    }
                }
            }
        }
        #pragma omp critical (GHIab)
        {
            Ga += Ga_local;
            Ha += Ha_local;
            Ia += Ia_local;
            Eost += Eost_local;
        }

        eosa += Eost;
        
    }
    
    
    // Alpha/Beta (BB|AA)
    #pragma omp parallel
    {
        double delta_BA=0.0;
        double num_os=0.0;
        double t2ba=0.0;
        
        // pointers
        // w0 and w1
        int o_a = n_occa;
        int ov_aa = n_occa*n_vira;
        int ovo_aab = n_occa*n_vira*n_occb;
        
        // w0 and w1
        int ov_bb = n_occb*n_virb;
        int ovo_bba = n_occb*n_virb*n_occa;
        
        // w2
        int v_b = n_virb;
        int vv_bb = n_virb*n_virb;
        int vvv_bba = n_virb*n_virb*n_vira;
        
        // w3
        int o_b = n_occb;
        int oo_bb = n_occb*n_occb;
        int oov_bba = n_occb*n_occb*n_vira;
        
        // integrals
        const double *w1 = V_vovo_ab.memptr(); //t2ba
        const double *w2 = V_ovvv_ab.memptr(); //Gb
        const double *w3 = V_ovoo_ab.memptr(); //Hb
        
        arma::mat Gb_local (n_virb, n_occb, fill::zeros);
        arma::mat Hb_local (n_virb, n_occb, fill::zeros);
        arma::mat Ib_local (n_virb, n_occb, fill::zeros);
        #pragma omp for
        for(size_t a = 0; a < n_virb; a++) {
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        
                        //denominator
                        delta_BA = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];
                        
                        // t2 amplitude BBAA
                        t2ba = w1[(b*ovo_bba+j*ov_bb+a*o_b+i)] / delta_BA;

                        // jbca AABB
                        for(size_t c = 0; c < n_virb; c++) {
                            // Gb(c,i) += c_os * t2ba * w2[(j*vvv_bba+b*vv_bb+c*v_b+a)];
                            Gb_local(c,i) += c_os * t2ba * w2[(j*vvv_bba+b*vv_bb+c*v_b+a)];
                        }
                        
                        // jbik AABB
                        for(size_t k = 0; k < n_occb; k++) {
                            // Hb(a,k) -= c_os * t2ba * w3[(j*oov_bba+b*oo_bb+i*o_b+k)];
                            Hb_local(a,k) -= c_os * t2ba * w3[(j*oov_bba+b*oo_bb+i*o_b+k)];
                            
                        }
                        
                        // Ib(a,i) += c_os * t2ba * F1a(j,b);
                        Ib_local(a,i) += c_os * t2ba * F1a(j,b);
                        
                    }
                }
            }
        }
        #pragma omp critical (GHIba)
        {
            Gb += Gb_local;
            Hb += Hb_local;
            Ib += Ib_local;
        }
    }
    
    Eos = eosa;
    
    
    // (AA|AA)
    Essa = 0.0;
    
    #pragma omp parallel
    {
        double Esst=0.0;
        double delta_AA=0.0;
        double delta_AB=0.0;
        double num_ss=0.0;
        double t2aa=0.0, t2aa_2=0.0;
        double t2ab=0.0;
    
        // pointers
        // w0 and w1
        int o = n_occa;
        int ov = n_occa*n_vira;
        int oov = n_occa*n_occa*n_vira;
        // w2
        int v = n_vira;
        int vv = n_vira*n_vira;
        int vvv = n_vira*n_vira*n_vira;
        // w3
        int oo = n_occa*n_occa;
        
        // integrals
        const double *w0 = V_vovo_u_a.memptr(); //num_os
        const double *w1 = V_vovo_a.memptr(); //t2
        const double *w2 = V_ovvv_a.memptr(); //G
        const double *w3 = V_ovoo_a.memptr(); //H
        
        double Esst_local = 0.0; 
        arma::mat Ga_local (n_vira, n_occa, fill::zeros);
        arma::mat Ha_local (n_vira, n_occa, fill::zeros);
        arma::mat Ia_local (n_vira, n_occa, fill::zeros);
        arma::mat F2a_local (n_vira, n_occa, fill::zeros);
        #pragma omp for
        for(size_t a = 0; a < n_vira; a++) {
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                            
                        //denominator
                        delta_AA = eA(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];
                        
                        // t2 amplitude (not stored)
                        num_ss = w0[(a*oov+i*ov+b*o+j)] - w0[(b*oov+i*ov+a*o+j)];
                        t2aa = w1[(a*oov+i*ov+b*o+j)] / delta_AA;
                        t2aa_2 = w1[(b*oov+i*ov+a*o+j)] / delta_AA;
                        
                        // Omega_G - (jb|ca) ovvv permute 2,4
                        for(size_t c = 0; c < n_vira; c++) {
                            // Ga(c,i) += c_ss * (t2aa - t2aa_2) * w2[(j*vvv+b*vv+c*v+a)];
                            Ga_local(c,i) += c_ss * (t2aa - t2aa_2) * w2[(j*vvv+b*vv+c*v+a)];
                        }
                            
                        // Omega_H - (jb|ik) ovoo permute 1,3
                        for(size_t k = 0; k < n_occa; k++) {
                            // Ha(a,k) -= c_ss * (t2aa - t2aa_2) * w3[(j*oov+b*oo+i*o+k)];
                            Ha_local(a,k) -= c_ss * (t2aa - t2aa_2) * w3[(j*oov+b*oo+i*o+k)];
                        }

                        // Omega_I
                        // Ia(a,i) += c_ss * (t2aa - t2aa_2) * F1a(j,b);
                        Ia_local(a,i) += c_ss * (t2aa - t2aa_2) * F1a(j,b);
                        
                         
                        // energy calculation
                        // Esst += num_ss * ((t1a(a,i)*t1a(b,j))-(t1a(a,j)*t1a(b,i)) + c_ss * (t2aa - t2aa_2));
                        Esst_local += num_ss * ((t1a(a,i)*t1a(b,j))-(t1a(a,j)*t1a(b,i)) + c_ss * (t2aa - t2aa_2));
                        
                    }
                }
                
                // Omega_J
                // F2a(a,i) -= ((eA[n_occa+a] - eA(i)) * t1a(a,i));
                F2a_local(a,i) -= ((eA[n_occa+a] - eA(i)) * t1a(a,i));

            }
        }
        #pragma omp critical (GHIa)
        {
            Ga += Ga_local;
            Ha += Ha_local;
            Ia += Ia_local;
            F2a += F2a_local;
            Esst += Esst_local;
        }

        Essa+=Esst;

    }
    Essa *= 0.25;
    
    
    
    // (BB|BB)
    Essb = 0.0;
    
    #pragma omp parallel
    {
        double Esst=0.0;
        double delta_BB=0.0;
        double delta_BA=0.0;
        double num_ss = 0.0;
        double t2bb=0.0, t2bb_2=0.0;
        double t2ba=0.0;
    
        // pointers
        // w0 and w1
        int o = n_occb;
        int ov = n_occb*n_virb;
        int oov = n_occb*n_occb*n_virb;
        // w2
        int v = n_virb;
        int vv = n_virb*n_virb;
        int vvv = n_virb*n_virb*n_virb;
        // w3
        int oo = n_occb*n_occb;
        
        // integrals
        const double *w0 = V_vovo_u_b.memptr(); //num_os
        const double *w1 = V_vovo_b.memptr(); //t2
        const double *w2 = V_ovvv_b.memptr(); //G
        const double *w3 = V_ovoo_b.memptr(); //H
        
        double Esst_local = 0.0; 
        arma::mat Gb_local (n_virb, n_occb, fill::zeros);
        arma::mat Hb_local (n_virb, n_occb, fill::zeros);
        arma::mat Ib_local (n_virb, n_occb, fill::zeros);
        arma::mat F2b_local (n_virb, n_occb, fill::zeros);
        #pragma omp for
        for(size_t a = 0; a < n_virb; a++) {
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                            
                        //denominator
                        delta_BB = eB(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];
                        
                        // t2 amplitude (not stored)
                        num_ss = w0[(a*oov+i*ov+b*o+j)] - w0[(b*oov+i*ov+a*o+j)];
                        t2bb = w1[(a*oov+i*ov+b*o+j)] / delta_BB;
                        t2bb_2 = w1[(b*oov+i*ov+a*o+j)] / delta_BB;
                        
                        // Omega_G - (jb|ca) ovvv permute 2,4
                        for(size_t c = 0; c < n_virb; c++) {
                            // Gb(c,i) += c_ss * (t2bb - t2bb_2) * w2[(j*vvv+b*vv+c*v+a)];
                            Gb_local(c,i) += c_ss * (t2bb - t2bb_2) * w2[(j*vvv+b*vv+c*v+a)];
                        }
                            
                        // Omega_H - (jb|ik) ovoo permute 1,3
                        for(size_t k = 0; k < n_occb; k++) {
                            // Hb(a,k) -= c_ss * (t2bb - t2bb_2) * w3[(j*oov+b*oo+i*o+k)];
                            Hb_local(a,k) -= c_ss * (t2bb - t2bb_2) * w3[(j*oov+b*oo+i*o+k)];
                        }

                        // Omega_I
                        // Ib(a,i) += c_ss * (t2bb - t2bb_2) * F1b(j,b);
                        Ib_local(a,i) += c_ss * (t2bb - t2bb_2) * F1b(j,b);
                        
                        
                        // energy calculation
                        // Esst += num_ss * ((t1b(a,i)*t1b(b,j))-(t1b(a,j)*t1b(b,i)) + c_ss * (t2bb - t2bb_2));
                        Esst_local += num_ss * ((t1b(a,i)*t1b(b,j))-(t1b(a,j)*t1b(b,i)) + c_ss * (t2bb - t2bb_2));
                        
                    }
                }
                
                // Omega_J
                // F2b(a,i) -= ((eB[n_occb+a] - eB(i)) * t1b(a,i));
                F2b_local(a,i) -= ((eB[n_occb+a] - eB(i)) * t1b(a,i));
                
            }
        }
        #pragma omp critical (GHIb)
        {
            Gb += Gb_local;
            Hb += Hb_local;
            Ib += Ib_local;
            F2b += F2b_local;
            Esst += Esst_local;
        }
        
        Essb+=Esst;

    }
    Essb *= 0.25;



    // Form new t1a amplitudes
    #pragma omp parallel
    {
        double delta_A=0.0;
        #pragma omp for
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t a = 0; a < n_vira; a++) {
                
                delta_A = eA(i) - eA[n_occa+a];
                t1a(a,i) = (Ga(a,i) + Ha(a,i) + Ia(a,i) + F2a(a,i)) / delta_A;
                
            }
        }

    }

    // Form new t1b amplitudes
    #pragma omp parallel
    {
        double delta_B=0.0;
        #pragma omp for
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t a = 0; a < n_virb; a++) {
                
                delta_B = eB(i) - eB[n_occb+a];
                t1b(a,i) = (Gb(a,i) + Hb(a,i) + Ib(a,i) + F2b(a,i)) / delta_B;
            }
        }
    }

}
    
template<>
void cc2<complex<double>>::unrestricted_energy(
    complex<double> &Eos, complex<double> &Essa, complex<double> &Essb,
    const size_t& n_occa, const size_t& n_vira,
    const size_t& n_occb, const size_t& n_virb,
    Mat<complex<double> > &V_vovo_u_a,
    Mat<complex<double> > &V_vovo_u_b,
    Mat<complex<double> > &V_vovo_u_ab,
    Mat<complex<double> > &V_vovo_a,
    Mat<complex<double> > &V_vovo_b,
    Mat<complex<double> > &V_vovo_ab,
    Mat<complex<double> > &V_ovvv_a,
    Mat<complex<double> > &V_ovvv_b,
    Mat<complex<double> > &V_ovvv_ab,
    Mat<complex<double> > &V_ovvv_ba,
    Mat<complex<double> > &V_ovoo_a,
    Mat<complex<double> > &V_ovoo_b,
    Mat<complex<double> > &V_ovoo_ab,
    Mat<complex<double> > &V_ovoo_ba,
    Mat<complex<double> > &V_vooo_a,
    Mat<complex<double> > &V_vooo_b,
    Mat<complex<double> > &V_vooo_ab,
    Mat<complex<double> > &V_vooo_ba,
    Mat<complex<double> > &H1_a,
    Mat<complex<double> > &H2_a,
    Mat<complex<double> > &H1_b,
    Mat<complex<double> > &H2_b,
    Mat<complex<double> > &t1a,
    Mat<complex<double> > &t1b,
    Col<complex<double>> &eA, 
    Col<complex<double>> &eB,
    double c_os, double c_ss)
    
    {

    // intermediates
    arma::cx_mat Ga (n_vira, n_occa, fill::zeros);
    arma::cx_mat Ha (n_vira, n_occa, fill::zeros);
    arma::cx_mat Ia (n_vira, n_occa, fill::zeros);
    arma::cx_mat F1a (n_occa, n_vira, fill::zeros);
    arma::cx_mat F2a (n_vira, n_occa, fill::zeros);
          
    arma::cx_mat Gb (n_virb, n_occb, fill::zeros);
    arma::cx_mat Hb (n_virb, n_occb, fill::zeros);
    arma::cx_mat Ib (n_virb, n_occb, fill::zeros);
    arma::cx_mat F1b (n_occb, n_virb, fill::zeros);
    arma::cx_mat F2b (n_virb, n_occb, fill::zeros);
    
    // Form F1 and F2 first for (AA|AA), (AA|BB), (BB|BB), (BB|AA)
    
    // (AA|AA), (AA|BB)
    {
        // w1
        int oov = n_occa*n_occa*n_vira;
        int oo = n_occa*n_occa;
        int o = n_occa;
        //w2
        int oov_bba = n_occb*n_occb*n_vira;
        int oo_bb = n_occb*n_occb;
        int o_b = n_occb;
        
        int ooo_bba = n_occb*n_occb*n_occa;
        int ooo = n_occa*n_occa*n_occa;
        
        const complex<double> *w1 = V_ovoo_a.memptr(); //F1 AA
        const complex<double> *w2 = V_ovoo_ab.memptr(); //F1 AB
        const complex<double> *w3 = V_vooo_a.memptr(); //F2 AA
        const complex<double> *w4 = V_vooo_ab.memptr(); //F2 AB
        
        // (ia|kk) - (AA|AA)
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t k = 0; k < n_occa; k++) {
                    F1a(i,a) += w1[(i*oov+a*oo+k*o+k)] - w1[(k*oov+a*oo+i*o+k)];
                    F2a(a,i) += w3[(a*ooo+i*oo+k*o+k)] - w3[(a*ooo+k*oo+k*o+i)];
                }
            }
        }
        
        // (ia|kk) - (AA|BB)
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t a = 0; a < n_vira; a++) {
                for(size_t k = 0; k < n_occb; k++) {
                    F1a(i,a) += w2[(i*oov_bba+a*oo_bb+k*o_b+k)];
                    F2a(a,i) += w4[(a*ooo_bba+i*oo_bb+k*o_b+k)];
                }
                F1a(i,a) += H1_a(i,a);
                F2a(a,i) += H2_a(a,i);
            }
        }
        
    }
    
    // (BB|BB), (BB|AA)
    {
        // w1
        int oov = n_occb*n_occb*n_virb;
        int oo = n_occb*n_occb;
        int o = n_occb;
        // w2
        int oov_aab = n_occa*n_occa*n_virb;
        int oo_aa = n_occa*n_occa;
        int o_a = n_occa;
        
        int ooo_aab = n_occa*n_occa*n_occb;
        int ooo = n_occb*n_occb*n_occb;
        
        const complex<double> *w1 = V_ovoo_b.memptr(); //F1 BB
        const complex<double> *w2 = V_ovoo_ba.memptr(); //F1 BA
        const complex<double> *w3 = V_vooo_b.memptr(); //F2 BB
        const complex<double> *w4 = V_vooo_ba.memptr(); //F2 BA
        
        // (ia|kk) - (BB|BB)
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t k = 0; k < n_occb; k++) {
                    F1b(i,a) += w1[(i*oov+a*oo+k*o+k)] - w1[(k*oov+a*oo+i*o+k)];
                    F2b(a,i) += w3[(a*ooo+i*oo+k*o+k)] - w3[(a*ooo+k*oo+k*o+i)];
                }
            }
        }
        
        // (ia|kk) - (BB|AA)
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t a = 0; a < n_virb; a++) {
                for(size_t k = 0; k < n_occa; k++) {
                    F1b(i,a) += w2[(i*oov_aab+a*oo_aa+k*o_a+k)];
                    F2b(a,i) += w4[(a*ooo_aab+i*oo_aa+k*o_a+k)];
                }
                F1b(i,a) += H1_b(i,a);
                F2b(a,i) += H2_b(a,i);
            }
        }
        
    }
    
    
    // Form the intermediates G, H, I for (AA|BB), (BB|AA), (AA|AA), (BB|BB)
    
    // Alpha/Beta (AA|BB)
    complex<double> eosa(0.,0.);
    
    {
        complex<double> Eost(0.,0.);
        complex<double> delta_AB(0.,0.);
        complex<double> num_os(0.,0.);
        complex<double> t2ab(0.,0.);
        
        // pointers
        // w0 and w1
        int o_b = n_occb;
        int ov_bb = n_occb*n_virb;
        int ovo_bba = n_occb*n_virb*n_occa;
        // w2
        int v_a = n_vira;
        int vv_aa = n_vira*n_vira;
        int vvv_aab = n_vira*n_vira*n_virb;
        // w3
        int o_a = n_occa;
        int oo_aa = n_occa*n_occa;
        int oov_aab = n_occa*n_occa*n_virb;
        
        // integrals
        const complex<double> *w0 = V_vovo_u_ab.memptr(); //num_os
        const complex<double> *w1 = V_vovo_ab.memptr(); //t2ab
        const complex<double> *w2 = V_ovvv_ba.memptr(); //Ga
        const complex<double> *w3 = V_ovoo_ba.memptr(); //Ha
        
        for(size_t a = 0; a < n_vira; a++) {
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                        
                        //denominator
                        delta_AB = eA(i) + eB(j) - eA[n_occa+a] - eB[n_occb+b];
                        
                        // t2 amplitude (ai|bj) AABB
                        num_os = w0[(a*ovo_bba+i*ov_bb+b*o_b+j)];
                        t2ab = (conj(delta_AB) * w1[(a*ovo_bba+i*ov_bb+b*o_b+j)]) / 
                               (conj(delta_AB) * delta_AB);
                    
                        // (jb|ca) BBAA
                        for(size_t c = 0; c < n_vira; c++) {
                            Ga(c,i) += t2ab * w2[(j*vvv_aab+b*vv_aa+c*v_a+a)];
                        }
                        
                        // (jb|ik) BBAA
                        for(size_t k = 0; k < n_occa; k++) {
                            Ha(a,k) -= t2ab * w3[(j*oov_aab+b*oo_aa+i*o_a+k)];
                        }
                        
                        Ia(a,i) += t2ab * F1b(j,b);
                        
                        // energy calculation
                        Eost += num_os * (t1a(a,i)*t1b(b,j) + t2ab);
                        
                    }
                }
            }
        }

        eosa += Eost;
        
    }
    
    
    // Alpha/Beta (BB|AA)
    {
        complex<double> delta_BA(0.,0.);
        complex<double> num_os(0.,0.);
        complex<double> t2ba(0.,0.);
        
        // pointers
        // w0 and w1
        int o_a = n_occa;
        int ov_aa = n_occa*n_vira;
        int ovo_aab = n_occa*n_vira*n_occb;
        
        // w0 and w1
        int ov_bb = n_occb*n_virb;
        int ovo_bba = n_occb*n_virb*n_occa;
        
        // w2
        int v_b = n_virb;
        int vv_bb = n_virb*n_virb;
        int vvv_bba = n_virb*n_virb*n_vira;
        
        // w3
        int o_b = n_occb;
        int oo_bb = n_occb*n_occb;
        int oov_bba = n_occb*n_occb*n_vira;
        
        // integrals
        const complex<double> *w1 = V_vovo_ab.memptr(); //t2ba
        const complex<double> *w2 = V_ovvv_ab.memptr(); //Gb
        const complex<double> *w3 = V_ovoo_ab.memptr(); //Hb
        
        for(size_t a = 0; a < n_virb; a++) {
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                        
                        //denominator
                        delta_BA = eB(i) + eA(j) - eB[n_occb+a] - eA[n_occa+b];
                        
                        // t2 amplitude BBAA
                        t2ba = (conj(delta_BA) * w1[(b*ovo_bba+j*ov_bb+a*o_b+i)]) / 
                               (conj(delta_BA) * delta_BA);

                        // jbca AABB
                        for(size_t c = 0; c < n_virb; c++) {
                            Gb(c,i) += t2ba * w2[(j*vvv_bba+b*vv_bb+c*v_b+a)];
                        }
                        
                        // jbik AABB
                        for(size_t k = 0; k < n_occb; k++) {
                            Hb(a,k) -= t2ba * w3[(j*oov_bba+b*oo_bb+i*o_b+k)];
                            
                        }
                        
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
        complex<double> Esst(0.,0.);
        complex<double> num_ss(0.,0.);
        complex<double> delta_AA(0.,0.), delta_A(0.,0.), delta_AB(0.,0.);
        complex<double> t2aa(0.,0.), t2aa_2(0.,0.), t2ab(0.,0.);
    
        // pointers
        // w0 and w1
        int o = n_occa;
        int ov = n_occa*n_vira;
        int oov = n_occa*n_occa*n_vira;
        // w2
        int v = n_vira;
        int vv = n_vira*n_vira;
        int vvv = n_vira*n_vira*n_vira;
        // w3
        int oo = n_occa*n_occa;
        
        // integrals
        const complex<double> *w0 = V_vovo_u_a.memptr(); //num_os
        const complex<double> *w1 = V_vovo_a.memptr(); //t2
        const complex<double> *w2 = V_ovvv_a.memptr(); //G
        const complex<double> *w3 = V_ovoo_a.memptr(); //H
        
        for(size_t a = 0; a < n_vira; a++) {
            for(size_t i = 0; i < n_occa; i++) {
                for(size_t b = 0; b < n_vira; b++) {
                    for(size_t j = 0; j < n_occa; j++) {
                            
                        //denominator
                        delta_AA = eA(i) + eA(j) - eA[n_occa+a] - eA[n_occa+b];
                        
                        // t2 amplitude (not stored)
                        num_ss = w0[(a*oov+i*ov+b*o+j)] - w0[(b*oov+i*ov+a*o+j)];
                        t2aa = (conj(delta_AA) * w1[(a*oov+i*ov+b*o+j)]) / 
                               (conj(delta_AA) * delta_AA);
                        t2aa_2 = (conj(delta_AA) * w1[(b*oov+i*ov+a*o+j)]) / 
                                 (conj(delta_AA) * delta_AA);
                        
                        // Omega_G - (jb|ca) ovvv permute 2,4
                        for(size_t c = 0; c < n_vira; c++) {
                            Ga(c,i) += (t2aa - t2aa_2) * w2[(j*vvv+b*vv+c*v+a)];
                        }
                            
                        // Omega_H - (jb|ik) ovoo permute 1,3
                        for(size_t k = 0; k < n_occa; k++) {
                            Ha(a,k) -= (t2aa - t2aa_2) * w3[(j*oov+b*oo+i*o+k)];
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
        
        // Form new t1a amplitudes
        for(size_t i = 0; i < n_occa; i++) {
            for(size_t a = 0; a < n_vira; a++) {
                
                delta_A = eA(i) - eA[n_occa+a];
                t1a(a,i) = (conj(delta_A) * (Ga(a,i) + Ha(a,i) + Ia(a,i) + F2a(a,i))) / 
                           (conj(delta_A) * delta_A);
                
            }
        }
        

        Essa+=Esst;

    }
    Essa *= 0.25;
    
    
    
    // (BB|BB)
    Essb = 0.0;
    
    {
        complex<double> Esst=0.0;
        complex<double> num_ss = 0.0;
        complex<double> delta_BB(0.,0.), delta_B(0.,0.), delta_BA(0.,0.);
        complex<double> t2bb(0.,0.), t2bb_2(0.,0.), t2ba(0.,0.);
    
        // pointers
        // w0 and w1
        int o = n_occb;
        int ov = n_occb*n_virb;
        int oov = n_occb*n_occb*n_virb;
        // w2
        int v = n_virb;
        int vv = n_virb*n_virb;
        int vvv = n_virb*n_virb*n_virb;
        // w3
        int oo = n_occb*n_occb;
        
        // integrals
        const complex<double> *w0 = V_vovo_u_b.memptr(); //num_os
        const complex<double> *w1 = V_vovo_b.memptr(); //t2
        const complex<double> *w2 = V_ovvv_b.memptr(); //G
        const complex<double> *w3 = V_ovoo_b.memptr(); //H
        
        for(size_t a = 0; a < n_virb; a++) {
            for(size_t i = 0; i < n_occb; i++) {
                for(size_t b = 0; b < n_virb; b++) {
                    for(size_t j = 0; j < n_occb; j++) {
                            
                        //denominator
                        delta_BB = eB(i) + eB(j) - eB[n_occb+a] - eB[n_occb+b];
                        
                        // t2 amplitude (not stored)
                        num_ss = w0[(a*oov+i*ov+b*o+j)] - w0[(b*oov+i*ov+a*o+j)];
                        t2bb = (conj(delta_BB) * w1[(a*oov+i*ov+b*o+j)]) / 
                               (conj(delta_BB) * delta_BB);
                        t2bb_2 = (conj(delta_BB) * w1[(b*oov+i*ov+a*o+j)]) / 
                                 (conj(delta_BB) * delta_BB);
                        
                        // Omega_G - (jb|ca) ovvv permute 2,4
                        for(size_t c = 0; c < n_virb; c++) {
                            Gb(c,i) += (t2bb - t2bb_2) * w2[(j*vvv+b*vv+c*v+a)];
                        }
                            
                        // Omega_H - (jb|ik) ovoo permute 1,3
                        for(size_t k = 0; k < n_occb; k++) {
                            Hb(a,k) -= (t2bb - t2bb_2) * w3[(j*oov+b*oo+i*o+k)];
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
        
        // Form new t1b amplitudes
        for(size_t i = 0; i < n_occb; i++) {
            for(size_t a = 0; a < n_virb; a++) {
                
                delta_B = eB(i) - eB[n_occb+a];
                t1b(a,i) = (conj(delta_B) * (Gb(a,i) + Hb(a,i) + Ib(a,i) + F2b(a,i))) / 
                           (conj(delta_B) * delta_B);
            }
        }
        
        Essb+=Esst;

    }
    Essb *= 0.25;
    
    
}


template class cc2<double>;
template class cc2<std::complex<double> >;

}
