#ifndef LIBGMBPT_RI_EOMIP_UNR_R_H
#define LIBGMBPT_RI_EOMIP_UNR_R_H

#include <memory>
#include <armadillo>
#include <string>
#include <libqints/basis/basis_2e2c_cgto.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/basis/bagen_1e2c_shellpair_cgto.h>
#include <libgmbpt/eom/regularization.h>
#include <complex>

using namespace arma;
using namespace libqints;
using namespace std;

namespace libgmbpt{

    template<typename TC, typename TI>
    class ri_eomip_unr_r{
    public:
        regularization_i<double>& m_reg;
        null_regularization<double> m_reg0;
    public:
        ri_eomip_unr_r(regularization_i<double>& reg): m_reg (reg) {};
        ri_eomip_unr_r(): m_reg (m_reg0) {};
        
        void ccs_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira,
            const size_t& n_occb, const size_t& n_virb,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
            Mat<TC> &BQoh_b, Mat<TC> &BQho_b, Mat<TC> &BQoo_b, 
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
            Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b,  
            Row<TC> &r1a, Row<TC> &r1b, 
            Row<TC> &res_a, Row<TC> &res_b, 
            Col<TC> &eA, Col<TC> &eB,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, rowvec &sigma_a, rowvec &sigma_b);

        void davidson_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira,
            const size_t& n_occb, const size_t& n_virb,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
            Mat<TC> &BQoh_b, Mat<TC> &BQho_b, Mat<TC> &BQoo_b, 
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
            Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b,  
            Row<TC> &r1a, Row<TC> &r1b, 
            Row<TC> &res_a, Row<TC> &res_b, 
            Col<TC> &eA, Col<TC> &eB,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, rowvec &sigma_a, rowvec &sigma_b);

        void diis_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira,
            const size_t& n_occb, const size_t& n_virb,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
            Mat<TC> &BQoh_b, Mat<TC> &BQho_b, Mat<TC> &BQoo_b, 
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
            Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b,  
            Row<TC> &r1a, Row<TC> &r1b, 
            Row<TC> &res_a, Row<TC> &res_b, 
            Col<TC> &eA, Col<TC> &eB,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, rowvec &sigma_a, rowvec &sigma_b);

        #if 0
        void restricted_energy(
            complex<double> &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<complex<double>> &BQov_a, Mat<complex<double>> &BQvo_a, Mat<complex<double>> &BQhp_a, 
	        Mat<complex<double>> &BQoh_a, Mat<complex<double>> &BQho_a, Mat<complex<double>> &BQoo_a, 
	        Mat<complex<double>> &BQob_a, Mat<complex<double>> &BQpv_a, Mat<complex<double>> &BQpo_a, 
	        Mat<complex<double>> &BQhb_a, Mat<complex<double>> &BQbp_a, Mat<complex<double>> &V_Pab,  
	        Mat<complex<double>> &Lam_hA, Mat<complex<double>> &Lam_pA,
            Mat<complex<double>> &Lam_hA_bar, Mat<complex<double>> &Lam_pA_bar,
            Mat<complex<double>> &CoccA, Mat<complex<double>> &CvirtA,
            Mat<complex<double>> &f_vv, Mat<complex<double>> &f_oo,
            Mat<complex<double>> &t1, Mat<complex<double>> &r1, Mat<complex<double>> &residual, 
	        Col<complex<double>> &e_orb, array_view<TI> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3);
        #endif
        
    };

};

#endif // LIBGMBPT_RI_EOMIP_UNR_R_H
