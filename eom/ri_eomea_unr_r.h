#ifndef LIBGMBPT_RI_EOMEA_UNR_R_H
#define LIBGMBPT_RI_EOMEA_UNR_R_H

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
    class ri_eomea_unr_r{
    public:
        regularization_i<double>& m_reg;
        null_regularization<double> m_reg0;
    public:
        ri_eomea_unr_r(regularization_i<double>& reg): m_reg (reg) {};
        ri_eomea_unr_r(): m_reg (m_reg0) {};
        
        void css_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira,
            const size_t& n_occb, const size_t& n_virb,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoo_a, Mat<TC> &BQpo_a, Mat<TC> &BQvp_a,
            Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQpo_b, Mat<TC> &BQvp_b,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
            Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, 
            Col<TC> &r1a, Col<TC> &r1b, 
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_buff_ao,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Col<TC> &sigma_a, Col<TC> &sigma_b);

        
        void davidson_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira,
            const size_t& n_occb, const size_t& n_virb,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoo_a, Mat<TC> &BQpo_a, Mat<TC> &BQvp_a,
            Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQpo_b, Mat<TC> &BQvp_b,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
            Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, 
            Col<TC> &r1a, Col<TC> &r1b, 
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_buff_ao,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Col<TC> &sigma_a, Col<TC> &sigma_b);

        
        void diis_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira,
            const size_t& n_occb, const size_t& n_virb,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoo_a, Mat<TC> &BQpo_a, Mat<TC> &BQvp_a,
            Mat<TC> &BQov_b, Mat<TC> &BQvo_b, Mat<TC> &BQhp_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQpo_b, Mat<TC> &BQvp_b,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a,
            Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, 
            Col<TC> &r1a, Col<TC> &r1b, 
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_buff_ao,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Col<TC> &sigma_a, Col<TC> &sigma_b);

        
    };

};

#endif // LIBGMBPT_RI_EOMEA_UNR_R_H
