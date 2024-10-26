#ifndef LIBGMBPT_CC2_H
#define LIBGMBPT_CC2_H

#include <memory>
#include <armadillo>
#include <string>
#include <libqints/basis/basis_2e2c_cgto.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/basis/bagen_1e2c_shellpair_cgto.h>
#include <libgmbpt/cc2/regularization.h>

#include <complex>
using namespace std;
using namespace arma;
using namespace libqints;

namespace libgmbpt{

    template<typename TC>
    class cc2{
    public:
        regularization_i<double>& m_reg;
        null_regularization<double> m_reg0;
        //regularization_i<TC>& m_reg;
        //null_regularization<TC> m_reg0;
    public:
        cc2(regularization_i<double>& reg): m_reg (reg) {};
        //cc2(regularization_i<TC>& reg): m_reg (reg) {};
        cc2(): m_reg (m_reg0) {};

        void restricted_energy(
            double &Eos, double &Ess,
            const size_t& n_occ, const size_t& n_vir,
            Mat<TC> &V_vovo_u, Mat<TC> &V_vovo,
            Mat<TC> &V_ovvv, Mat<TC> &V_ovoo,
            Mat<TC> &V_vooo, Mat<TC> &H1_a, Mat<TC> &H2_a,
            Mat<TC> &t1, Col<double> &e_orb,
            double c_os, double c_ss);

        // CU
        void restricted_energy(
            complex<double> &Eos, complex<double> &Ess,
            const size_t& n_occ, const size_t& n_vir,
            Mat<TC> &V_vovo_u, Mat<TC> &V_vovo,
            Mat<TC> &V_ovvv, Mat<TC> &V_ovoo,
            Mat<TC> &V_vooo, Mat<TC> &H1_a, Mat<TC> &H2_a,
            Mat<TC> &t1, Col<complex<double>> &e_orb,
            double c_os, double c_ss);

        void unrestricted_energy(
            double &Eos, double &Essa, double &Essb,
            const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            Mat<TC> &V_vovo_u_a, Mat<TC> &V_vovo_u_b, Mat<TC> &V_vovo_u_ab, 
            Mat<TC> &V_vovo_a, Mat<TC> &V_vovo_b, Mat<TC> &V_vovo_ab,
            Mat<TC> &V_ovvv_a, Mat<TC> &V_ovvv_b, Mat<TC> &V_ovvv_ab, Mat<TC> &V_ovvv_ba,
            Mat<TC> &V_ovoo_a, Mat<TC> &V_ovoo_b, Mat<TC> &V_ovoo_ab, Mat<TC> &V_ovoo_ba,
            Mat<TC> &V_vooo_a, Mat<TC> &V_vooo_b, Mat<TC> &V_vooo_ab, Mat<TC> &V_vooo_ba,
            Mat<TC> &H1_a, Mat<TC> &H2_a, Mat<TC> &H1_b, Mat<TC> &H2_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Col<double> &eA, Col<double> &eB, double c_os, double c_ss);

        // CU
        void unrestricted_energy(
            complex<double> &Eos, complex<double> &Essa, complex<double> &Essb,
            const size_t& n_occa, const size_t& n_vira, 
            const size_t& n_occb, const size_t& n_virb, 
            Mat<TC> &V_vovo_u_a, Mat<TC> &V_vovo_u_b, Mat<TC> &V_vovo_u_ab, 
            Mat<TC> &V_vovo_a, Mat<TC> &V_vovo_b, Mat<TC> &V_vovo_ab,
            Mat<TC> &V_ovvv_a, Mat<TC> &V_ovvv_b, Mat<TC> &V_ovvv_ab, Mat<TC> &V_ovvv_ba,
            Mat<TC> &V_ovoo_a, Mat<TC> &V_ovoo_b, Mat<TC> &V_ovoo_ab, Mat<TC> &V_ovoo_ba,
            Mat<TC> &V_vooo_a, Mat<TC> &V_vooo_b, Mat<TC> &V_vooo_ab, Mat<TC> &V_vooo_ba,
            Mat<TC> &H1_a, Mat<TC> &H2_a, Mat<TC> &H1_b, Mat<TC> &H2_b,
            Mat<TC> &t1a, Mat<TC> &t1b, Col<complex<double>> &eA, Col<complex<double>> &eB, double c_os, double c_ss);
    };

};

#endif // LIBGMBPT_CC2_H
