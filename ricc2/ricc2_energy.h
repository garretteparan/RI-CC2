#ifndef LIBGMBPT_RICC2_ENERGY_H
#define LIBGMBPT_RICC2_ENERGY_H

#include <armadillo>
#include <complex>
#include <libqints/basis/basis_1e1c_cgto.h>
#include <libqints/basis/basis_2e4c_shellpair_cgto.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/exec/dev_omp.h>
#include <libgscf/fock/scf_type.h>
#include <libgmbpt/mo/mo_spaces.h>
#include <libgmbpt/ricc2/regularization.h>

using namespace std;
using namespace arma;
using namespace libqints;

namespace libgmbpt {

/** \brief Evaluates the RICC2 energy
    \ingroup libgmbpt_ricc2
 **/
template<typename TC, typename TI>
class ricc2_energy {
private:
    regularization_i<double>& m_reg;
    null_regularization<double> m_reg0;
    const libqints::dev_omp &m_dev; //!< Thread private memory
    const libgscf::scf_type m_scf_type; //!< Type of SCF
    const libqints::basis_2e3c_shellpair_cgto<TI> &m_b3; //!< Basis
    const libqints::array_view<TI> m_av_pqinvhalf; //!< (P|Q)^(-1/2)
    const mo_spaces m_ns; //!< Sizes of MO spaces


public:
    /** \brief Constructor of ricc2_energy
        \param[in] dev dev
        \param[in] scf_type Type of SCF
        \param[in] b4 Basis
        \param[in] av_aos AO overlap matrix
        \param[in] av_pqinvhalf (P|Q)^(-1/2)
        \param[in] av_orb Size of orbital spaces including frozen core/virtual
        \param[in] ma_aofock AO Fock matrix
        \param[in] scf_thresh SCF convergence threshold
     **/

    ricc2_energy(
        const libqints::dev_omp &dev,
        regularization_i<double>& reg0,
        libgscf::scf_type scf_type,
        const libqints::basis_2e3c_shellpair_cgto<TI> &b3, //!< Basis
        const libqints::array_view<TI> av_pqinvhalf, //!< (P|Q)^(-1/2)
        const libqints::array_view<size_t> av_orb) : //!< Size of orbital spaces
        m_dev(dev), m_reg(m_reg0), m_scf_type(scf_type), m_b3(b3),
        m_av_pqinvhalf(av_pqinvhalf), m_ns(av_orb) { }


    /** \brief Compute RICC2 energy
        \param[in] av_buff Shared memory buffer
        \param[in,out] ma_c MO coeff
        \param[out] av_ene RICC2 energies
     **/
    void perform(
        libqints::array_view<TC> av_buff,
        libqints::multi_array<TC> ma_c,
        Mat<TC> &t1a, Mat<TC> &t1b,
        libqints::array_view<TC> av_ene,
        arma::Col<TC> Ea, arma::Col<TC> Eb,
        double c_os, double c_ss, int dig);


    /** \brief Return how much memory required (in Byte / sizeof(TC))
     **/
    size_t required_memory();

    /** \brief Return size of av_buff for perform()
     **/
    size_t required_buffer_size();

    /** \brief Return size of av_buff for perform()
     **/
    size_t required_buffer_size_digestor();

    /** \brief Return minimum dynamic memory required (mainly for motran)
     **/
    size_t required_min_dynamic_memory();

private:
    /** \brief Compute BQai
        \param[out] BQai
        \param[in] av_buff Buffer for VaiP
        \param[in] Cocc active occ MO
        \param[in] Cvir active vir MO
     **/
    void compute_BQmn(
        arma::Mat<TC> &BQmn,
        libqints::array_view<TC> av_buff,
        arma::Mat<TC> Cocc,
        arma::Mat<TC> Cvir);

}; // class ricc2_energy
} // namespace libgmbpt

#endif // LIBGMBPT_ricc2_energy_H

