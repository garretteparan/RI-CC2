#ifndef LIBGMBPT_RI_EOMEE_TRIP_R_H
#define LIBGMBPT_RI_EOMEE_TRIP_R_H

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
    class ri_eomee_trip_r{
    public:
        regularization_i<double>& m_reg;
        null_regularization<double> m_reg0;
    public:
        ri_eomee_trip_r(regularization_i<double>& reg): m_reg (reg) {};
        ri_eomee_trip_r(): m_reg (m_reg0) {};
        

        void ccs_restricted_energy_triplets(
            double &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQpv_a, Mat<TC> &V_Pab,  
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &f_vv, Mat<TC> &f_oo,
            Mat<TC> &t1, Mat<TC> &r1, Col<TC> &e_orb,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Mat<TC> &sigma);

        void davidson_restricted_energy_triplets(
            double &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQpv_a, Mat<TC> &V_Pab,  
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &f_vv, Mat<TC> &f_oo,
            Mat<TC> &t1, Mat<TC> &r1, Col<TC> &e_orb,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Mat<TC> &sigma);

        void diis_restricted_energy_triplets(
            double &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, Mat<TC> &BQpv_a, Mat<TC> &V_Pab,  
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &f_vv, Mat<TC> &f_oo,
            Mat<TC> &t1, Mat<TC> &r1, Col<double> &e_orb,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Mat<TC> &sigma);

        void ccs_restricted_energy_triplets_digestor(
            double &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &f_vv, Mat<TC> &f_oo,
            Mat<TC> &t1, Mat<TC> &r1, Col<double> &e_orb,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Mat<TC> &sigma);

        void davidson_restricted_energy_triplets_digestor(
            double &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a,  
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &f_vv, Mat<TC> &f_oo,
            Mat<TC> &t1, Mat<TC> &r1, Col<double> &e_orb,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Mat<TC> &sigma);

        void diis_restricted_energy_triplets_digestor(
            double &exci, const size_t& n_occ, const size_t& n_vir,
            const size_t& n_aux, const size_t& n_orb,
            Mat<TC> &BQov_a, Mat<TC> &BQvo_a, Mat<TC> &BQhp_a, 
            Mat<TC> &BQoh_a, Mat<TC> &BQho_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQob_a, Mat<TC> &BQpo_a, Mat<TC> &BQhb_a, 
            Mat<TC> &BQbp_a, 
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &Lam_hA_bar, Mat<TC> &Lam_pA_bar,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &f_vv, Mat<TC> &f_oo,
            Mat<TC> &t1, Mat<TC> &r1, Col<double> &e_orb,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss, Mat<TC> &sigma);

    };

};

#endif // LIBGMBPT_RI_EOMEE_TRIP_R_H
