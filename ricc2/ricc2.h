#ifndef LIBGMBPT_RICC2_H
#define LIBGMBPT_RICC2_H

#include <memory>
#include <armadillo>
#include <string>
#include <libqints/basis/basis_2e2c_cgto.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/basis/bagen_1e2c_shellpair_cgto.h>
#include <libgmbpt/ricc2/regularization.h>
#include <cassert>
#include <libqints/qints.h>
#include <libqints/arrays/idx_list.h>
#include <libqints/arrays/multi_array.h>
#include <libqints/operators/op_coulomb.h>
#include <libqints/screeners/scr_null.h>
#include <libqints/digestors/dig_passthru_2e2c_cgto.h>
#include <libposthf/motran/motran_2e3c.h>
#include <complex>

using namespace std;
using namespace arma;
using namespace libqints;

namespace libgmbpt {

    template<typename TC, typename TI>
    class ricc2 {

    public:
        regularization_i<double>& m_reg;
        null_regularization<double> m_reg0;

    public:
        ricc2(regularization_i<double>& reg): m_reg (reg) {};
        ricc2(): m_reg (m_reg0) {};


        // Haettig algorithm
        void restricted_energy(
            double &Eos, double &Ess,
            const size_t n_occ, const size_t n_vir,
            const size_t n_aux, const size_t n_orb,
            Mat<TC> &BQvo_a, Mat<TC> &BQov_a,
            Mat<TC> &BQhp_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQoh_a, Mat<TC> &V_Pab,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &t1, Col<TC> &e_orb,
            array_view<TI> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
            double c_os, double c_ss);


        void restricted_energy_digestor(
            double &Eos, double &Ess,
            const size_t n_occ, const size_t n_vir,
            const size_t n_aux, const size_t n_orb,
            Mat<TC> &BQvo_a, Mat<TC> &BQov_a,
            Mat<TC> &BQhp_a, Mat<TC> &BQoo_a, 
            Mat<TC> &BQoh_a,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
            Mat<TC> &CoccA, Mat<TC> &CvirtA,
            Mat<TC> &t1, Col<TC> &e_orb,
            array_view<TI> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
            double c_os, double c_ss);

        // CU
        void restricted_energy(
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
            double c_os, double c_ss);

        void restricted_energy_digestor(
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
            double c_os, double c_ss);

        void unrestricted_energy(
            double &Eos, double &Essa, double &Essb,
            const size_t n_occa, const size_t n_vira,
            const size_t n_occb, const size_t n_virb,
            const size_t n_aux, const size_t n_orb,
            Mat<TC> &BQvo_a, Mat<TC> &BQov_a,
            Mat<TC> &BQhp_a, Mat<TC> &BQoo_a, Mat<TC> &BQoh_a,
            Mat<TC> &BQvo_b, Mat<TC> &BQov_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoo_b, Mat<TC> &BQoh_b, Mat<TC> &V_Pab,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &t1a, Mat<TC> &t1b, Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss);

        void unrestricted_energy_digestor(
            double &Eos, double &Essa, double &Essb,
            const size_t n_occa, const size_t n_vira,
            const size_t n_occb, const size_t n_virb,
            const size_t n_aux, const size_t n_orb,
            Mat<TC> &BQvo_a, Mat<TC> &BQov_a,
            Mat<TC> &BQhp_a, Mat<TC> &BQoo_a, Mat<TC> &BQoh_a,
            Mat<TC> &BQvo_b, Mat<TC> &BQov_b,
            Mat<TC> &BQhp_b, Mat<TC> &BQoo_b, Mat<TC> &BQoh_b,
            Mat<TC> &Lam_hA, Mat<TC> &Lam_pA, Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
            Mat<TC> &CoccA, Mat<TC> &CvirtA, Mat<TC> &CoccB, Mat<TC> &CvirtB,
            Mat<TC> &t1a, Mat<TC> &t1b, Col<double> &eA, Col<double> &eB,
            array_view<TC> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<double> &m_b3,
            double c_os, double c_ss);

        // CU
        void unrestricted_energy(
            complex<double> &Eos, complex<double> &Essa, complex<double> &Essb,
            const size_t n_occa, const size_t n_vira,
            const size_t n_occb, const size_t n_virb,
            const size_t n_aux, const size_t n_orb,
            Mat<complex<double>> &BQvo_a,
            Mat<complex<double>> &BQov_a,
            Mat<complex<double>> &BQhp_a,
            Mat<complex<double>> &BQoo_a,
            Mat<complex<double>> &BQoh_a,
            Mat<complex<double>> &BQvo_b,
            Mat<complex<double>> &BQov_b,
            Mat<complex<double>> &BQhp_b,
            Mat<complex<double>> &BQoo_b,
            Mat<complex<double>> &BQoh_b,
            Mat<complex<double>> &V_Pab,
            Mat<complex<double> > &Lam_hA,
            Mat<complex<double> > &Lam_pA,
            Mat<complex<double> > &Lam_hB,
            Mat<complex<double> > &Lam_pB,
            Mat<complex<double> > &CoccA,
            Mat<complex<double> > &CvirtA,
            Mat<complex<double> > &CoccB,
            Mat<complex<double> > &CvirtB,
            Mat<complex<double> > &t1a,
            Mat<complex<double> > &t1b,
            Col<complex<double>> &eA, Col<complex<double>> &eB,
            array_view<TI> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
            double c_os, double c_ss);

        void unrestricted_energy_digestor(
            complex<double> &Eos, complex<double> &Essa, complex<double> &Essb,
            const size_t n_occa, const size_t n_vira,
            const size_t n_occb, const size_t n_virb,
            const size_t n_aux, const size_t n_orb,
            Mat<complex<double>> &BQvo_a,
            Mat<complex<double>> &BQov_a,
            Mat<complex<double>> &BQhp_a,
            Mat<complex<double>> &BQoo_a,
            Mat<complex<double>> &BQoh_a,
            Mat<complex<double>> &BQvo_b,
            Mat<complex<double>> &BQov_b,
            Mat<complex<double>> &BQhp_b,
            Mat<complex<double>> &BQoo_b,
            Mat<complex<double>> &BQoh_b,
            Mat<complex<double> > &Lam_hA,
            Mat<complex<double> > &Lam_pA,
            Mat<complex<double> > &Lam_hB,
            Mat<complex<double> > &Lam_pB,
            Mat<complex<double> > &CoccA,
            Mat<complex<double> > &CvirtA,
            Mat<complex<double> > &CoccB,
            Mat<complex<double> > &CvirtB,
            Mat<complex<double> > &t1a,
            Mat<complex<double> > &t1b,
            Col<complex<double>> &eA, Col<complex<double>> &eB,
            array_view<TI> av_pqinvhalf,
            const libqints::dev_omp &m_dev,
            const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3,
            double c_os, double c_ss);


#if 0
        // non-Haettig algorithm
        void restricted_energy(
             double &Eos, double &Ess,
             const size_t n_occ, const size_t n_vir,
             const size_t n_aux, const size_t n_orb,
             Mat<TC> &BQov_a, Mat<TC> &BQph_a, Mat<TC> &BQoh_a,
             Mat<TC> &BQvo_a, Mat<TC> &BQpv_a,
             Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
             Mat<TC> &H1_a, Mat<TC> &H2_a,
             Mat<TC> &t1, Col<double> &e_orb);


        void unrestricted_energy(
             double &Eos, double &Essa, double &Essb,
             const size_t n_occa, const size_t n_vira,
             const size_t n_occb, const size_t n_virb,
             const size_t n_aux, const size_t n_orb,
             Mat<TC> &BQov_a, Mat<TC> &BQph_a, Mat<TC> &BQoh_a,
             Mat<TC> &BQvo_a, Mat<TC> &BQpv_a,
             Mat<TC> &BQov_b, Mat<TC> &BQph_b, Mat<TC> &BQoh_b,
             Mat<TC> &BQvo_b, Mat<TC> &BQpv_b,
             Mat<TC> &Lam_hA, Mat<TC> &Lam_pA,
             Mat<TC> &Lam_hB, Mat<TC> &Lam_pB,
             Mat<TC> &H1_a, Mat<TC> &H2_a,
             Mat<TC> &H1_b, Mat<TC> &H2_b,
             Mat<TC> &t1a, Mat<TC> &t1b,
             Col<double> &eA, Col<double> &eB);
#endif


    };

};

#endif // LIBGMBPT_RICC2_H
