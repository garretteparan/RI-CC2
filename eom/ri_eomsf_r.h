#ifndef LIBGMBPT_RI_EOMSF_R_H
#define LIBGMBPT_RI_EOMSF_R_H

#include <memory>
#include <armadillo>
#include <string>
#include <libqints/basis/basis_2e2c_cgto.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/basis/bagen_1e2c_shellpair_cgto.h>
//#include <libgmbpt/ricc2/pqpow_consumer.h>
#include <libgmbpt/eom/regularization.h>

using namespace arma;
using namespace libqints;

namespace libgmbpt{

    template<typename TC>
    class ri_eomsf_r{
    public:
        regularization_i<TC>& m_reg;
        null_regularization<TC> m_reg0;
    public:
        ri_eomsf_r(regularization_i<TC>& reg): m_reg (reg) {};
        ri_eomsf_r(): m_reg (m_reg0) {};

        void ccs_spinflip(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a,  
            Mat<TC> &BQoo_a, Mat<TC> &BQpv_a,
            Mat<TC> &BQpo_a, Mat<TC> &BQov_b, 
            Mat<TC> &BQvo_b, Mat<TC> &BQoh_b, 
            Mat<TC> &BQho_b, Mat<TC> &BQoo_b, 
            Mat<TC> &BQpv_b, Mat<TC> &BQpo_b,  
            Mat<TC> &BQob_ab, Mat<TC> &BQob_ba, 
            Mat<TC> &BQhp_a, Mat<TC> &BQhp_b, 
            Mat<TC> &BQhb_ba, Mat<TC> &BQhb_ab,
            Mat<TC> &BQbp_ba, Mat<TC> &BQbp_ab, 
            Mat<TC> &V_Pab,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, 
            Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1_Ai, Mat<TC> &r1_aI,  
            Mat<TC> &res_Ai, Mat<TC> &res_aI, Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_Ai, Mat<TC> &sigma_aI);
 
        void davidson_spinflip(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a,  
            Mat<TC> &BQoo_a, Mat<TC> &BQpv_a,
            Mat<TC> &BQpo_a, Mat<TC> &BQov_b, 
            Mat<TC> &BQvo_b, Mat<TC> &BQoh_b, 
            Mat<TC> &BQho_b, Mat<TC> &BQoo_b, 
            Mat<TC> &BQpv_b, Mat<TC> &BQpo_b,  
            Mat<TC> &BQob_ab, Mat<TC> &BQob_ba, 
            Mat<TC> &BQhp_a, Mat<TC> &BQhp_b, 
            Mat<TC> &BQhb_ba, Mat<TC> &BQhb_ab,
            Mat<TC> &BQbp_ba, Mat<TC> &BQbp_ab, 
            Mat<TC> &V_Pab,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, 
            Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1_Ai, Mat<TC> &r1_aI,  
            Mat<TC> &res_Ai, Mat<TC> &res_aI, Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_Ai, Mat<TC> &sigma_aI);
        
        void diis_spinflip(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a,  
            Mat<TC> &BQoo_a, Mat<TC> &BQpv_a,
            Mat<TC> &BQpo_a, Mat<TC> &BQov_b, 
            Mat<TC> &BQvo_b, Mat<TC> &BQoh_b, 
            Mat<TC> &BQho_b, Mat<TC> &BQoo_b, 
            Mat<TC> &BQpv_b, Mat<TC> &BQpo_b,  
            Mat<TC> &BQob_ab, Mat<TC> &BQob_ba, 
            Mat<TC> &BQhp_a, Mat<TC> &BQhp_b, 
            Mat<TC> &BQhb_ba, Mat<TC> &BQhb_ab,
            Mat<TC> &BQbp_ba, Mat<TC> &BQbp_ab, 
            Mat<TC> &V_Pab,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, 
            Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1_Ai, Mat<TC> &r1_aI,  
            Mat<TC> &res_Ai, Mat<TC> &res_aI, Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_Ai, Mat<TC> &sigma_aI);

        
    };

};

#endif // LIBGMBPT_RI_EOMSF_R_H
