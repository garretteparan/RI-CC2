#ifndef LIBGMBPT_RI_EOMEE_UNR_R_H
#define LIBGMBPT_RI_EOMEE_UNR_R_H

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
    class ri_eomee_unr_r{
    public:
        regularization_i<double>& m_reg;
        null_regularization<double> m_reg0;
    public:
        ri_eomee_unr_r(regularization_i<double>& reg): m_reg (reg) {};
        ri_eomee_unr_r(): m_reg (m_reg0) {};
        
	void ccs_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQov_b, Mat<TC> &BQvo_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoh_b, Mat<TC> &BQho_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQob_b, Mat<TC> &BQpo_b, 
            Mat<TC> &BQhb_b, Mat<TC> &BQbp_b, 
            Mat<TC> &BQpv_a, Mat<TC> &BQpv_b, Mat<TC> &V_Pab, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1a, Mat<TC> &r1b,  
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_a, Mat<TC> &sigma_b);

	void davidson_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQov_b, Mat<TC> &BQvo_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoh_b, Mat<TC> &BQho_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQob_b, Mat<TC> &BQpo_b, 
            Mat<TC> &BQhb_b, Mat<TC> &BQbp_b, 
            Mat<TC> &BQpv_a, Mat<TC> &BQpv_b, Mat<TC> &V_Pab, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1a, Mat<TC> &r1b,  
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_a, Mat<TC> &sigma_b);

	void diis_unrestricted_energy(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQov_b, Mat<TC> &BQvo_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoh_b, Mat<TC> &BQho_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQob_b, Mat<TC> &BQpo_b, 
            Mat<TC> &BQhb_b, Mat<TC> &BQbp_b, 
            Mat<TC> &BQpv_a, Mat<TC> &BQpv_b, Mat<TC> &V_Pab, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1a, Mat<TC> &r1b,  
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_a, Mat<TC> &sigma_b);


	void ccs_unrestricted_energy_digestor(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQov_b, Mat<TC> &BQvo_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoh_b, Mat<TC> &BQho_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQob_b, Mat<TC> &BQpo_b, 
            Mat<TC> &BQhb_b, Mat<TC> &BQbp_b, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1a, Mat<TC> &r1b,  
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_a, Mat<TC> &sigma_b);

	void davidson_unrestricted_energy_digestor(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQov_b, Mat<TC> &BQvo_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoh_b, Mat<TC> &BQho_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQob_b, Mat<TC> &BQpo_b, 
            Mat<TC> &BQhb_b, Mat<TC> &BQbp_b, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1a, Mat<TC> &r1b,  
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_a, Mat<TC> &sigma_b);

	void diis_unrestricted_energy_digestor(
            double &exci, const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQov_b, Mat<TC> &BQvo_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoh_b, Mat<TC> &BQho_b, 
            Mat<TC> &BQoo_b, Mat<TC> &BQob_b, Mat<TC> &BQpo_b, 
            Mat<TC> &BQhb_b, Mat<TC> &BQbp_b, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar, 
            Mat<TC> &Lam_hB_bar, Mat<TC> &Lam_pB_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &f_vv_a, Mat<TC> &f_oo_a, Mat<TC> &f_vv_b, Mat<TC> &f_oo_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Mat<TC> &r1a, Mat<TC> &r1b,  
            Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            Mat<TC> &sigma_a, Mat<TC> &sigma_b);
        
    };

};

#endif // LIBGMBPT_RI_EOMEE_UNR_R_H
