#ifndef LIBGMBPT_CC2_ENERGY_H
#define LIBGMBPT_CC2_ENERGY_H

#include <armadillo>
#include <libqints/basis/basis_2e4c_shellpair_cgto.h>
#include <libqints/basis/basis_1e1c_cgto.h>
#include <libqints/exec/dev_omp.h>
#include <libgscf/fock/scf_type.h>
#include <libqints/arrays/multi_array.h>
#include <libgmbpt/cc2/regularization.h>
#include <libgmbpt/mo/mo_spaces.h>
#include <libgscf/fock/fock_builder.h>
#include <libgscf/fock/fock_desc.h>

#include <complex>
using namespace std;
using namespace arma;
using namespace libqints;

namespace libgmbpt {

/** \brief Evaluates the CC2 energy

    \ingroup libgmbpt_cc2
 **/
template<typename TC, typename TI>
class cc2_energy {
private:
    regularization_i<double>& m_reg;
    null_regularization<double> m_reg0;
    const libqints::dev_omp &m_dev; //!< Thread private memory
    const libgscf::scf_type m_scf_type; //!< Type of SCF
    const libqints::basis_1e1c_cgto<TI> &m_b1;      // added
    const libqints::sym_basis_2e4c_shellpair_cgto<TI> &m_b4; //!< Basis
    const libqints::multi_array<TC> m_ma_aofock; //!< AO Fock matrix
    const libqints::array_view<TI> m_av_aos; //!< AO Overlap matrix
    const mo_spaces m_ns; //!< Sizes of MO spaces

public:
    libgscf::fock_desc aop; //!< Parameters defining AO basis + energy functional and derivatives.
    libgscf::fock_builder<complex<double>, complex<double>> Fa,Fb; //!< Fock matrix builder

    /** \brief Constructor of cc2_energy
        \param[in] dev dev
        \param[in] scf_type Type of SCF
        \param[in] b4 Basis
        \param[in] av_aos AO overlap matrix
        \param[in] av_pqinvhalf (P|Q)^(-1/2)
        \param[in] av_orb Size of orbital spaces including frozen core/virtual
        \param[in] ma_aofock AO Fock matrix
        \param[in] scf_thresh SCF convergence threshold
     **/

    cc2_energy(
        const libqints::dev_omp &dev,
        regularization_i<double>& reg0,
        libgscf::scf_type scf_type,
        const libqints::basis_1e1c_cgto<TI> &b1,
        const libqints::sym_basis_2e4c_shellpair_cgto<TI> &b4, //!< Basis
        const libqints::array_view<TI> av_aos, //!< AO overlap matrix
        const libqints::array_view<size_t> av_orb, //!< Size of orbital spaces
        const libqints::multi_array<TC> ma_aofock) :
        m_dev(dev), m_reg(m_reg0), m_scf_type(scf_type), m_b1(b1), m_b4(b4), m_av_aos(av_aos),
        m_ns(av_orb), m_ma_aofock(ma_aofock) { }


    /** \brief Compute CC2 energy
        \param[in] av_buff Shared memory buffer
        \param[in,out] ma_c MO coeff
        \param[out] av_ene CC2 energies
     **/
    void perform(
        libqints::array_view<TC> av_buff,
        libqints::multi_array<TC> ma_c,
        libqints::multi_array<TC> ma_oe,
        Mat<TC> &t1a, Mat<TC> &t1b,
        libqints::array_view<TC> av_ene,
        arma::Col<TC> Ea, arma::Col<TC> Eb,
        double c_os, double c_ss);


    /** \brief Return how much memory required (in Byte / sizeof(TC))
     **/
    size_t required_memory();


}; // class cc2_energy
} // namespace libgmbpt

#endif // LIBGMBPT_CC2_ENERGY_H
