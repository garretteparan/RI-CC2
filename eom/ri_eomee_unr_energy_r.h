#ifndef LIBGMBPT_RI_EOMEE_UNR_ENERGY_R_H
#define LIBGMBPT_RI_EOMEE_UNR_ENERGY_R_H

#include <armadillo>
#include <libqints/basis/basis_1e1c_cgto.h>
#include <libqints/basis/basis_2e3c_shellpair_cgto.h>
#include <libqints/basis/basis_2e4c_shellpair_cgto.h>
#include <libqints/exec/dev_omp.h>
#include <libgscf/fock/scf_type.h>
#include <libgmbpt/mo/mo_spaces.h>
#include <libgmbpt/eom/regularization.h>

using namespace arma;
using namespace libqints;

namespace libgmbpt {
    
/** \brief Evaluates the EOM energy
 
    \ingroup libgmbpt_eom
 **/
template<typename TC, typename TI>
class ri_eomee_unr_energy_r {
private:
    regularization_i<TC>& m_reg;
    null_regularization<TC> m_reg0;
    const libqints::dev_omp &m_dev; //!< Thread private memory
    const libgscf::scf_type m_scf_type; //!< Type of SCF
    const libqints::basis_1e1c_cgto<double> &m_b1;      // added
    const libqints::basis_2e3c_shellpair_cgto<double> &m_b3; //!< Basis
    const libqints::sym_basis_2e4c_shellpair_cgto<double> &m_b4; //!< Basis
    const libqints::multi_array<TC> m_ma_aofock; //!< AO Fock matrix
    const libqints::array_view<double> m_av_aos; //!< AO Overlap matrix
    const libqints::array_view<double> m_av_pqinvhalf; //!< (P|Q)^(-1/2)
    const mo_spaces m_ns; //!< Sizes of MO spaces

public:

    /** \brief Constructor of eom_energy_r
        \param[in] dev dev
        \param[in] scf_type Type of SCF
        \param[in] b4 Basis
        \param[in] av_aos AO overlap matrix
        \param[in] av_pqinvhalf (P|Q)^(-1/2)
        \param[in] av_orb Size of orbital spaces including frozen core/virtual
        \param[in] ma_aofock AO Fock matrix
        \param[in] scf_thresh SCF convergence threshold
     **/
    
    ri_eomee_unr_energy_r(
        const libqints::dev_omp &dev,
        regularization_i<TC>& reg0,
        libgscf::scf_type scf_type,
        const libqints::basis_1e1c_cgto<double> &b1,
        const libqints::basis_2e3c_shellpair_cgto<double> &b3, //!< Basis
        const libqints::sym_basis_2e4c_shellpair_cgto<double> &b4, //!< Basis
        const libqints::array_view<double> av_aos, //!< AO overlap matrix
        const libqints::array_view<double> av_pqinvhalf, //!< (P|Q)^(-1/2)
        const libqints::array_view<size_t> av_orb, //!< Size of orbital spaces
        const libqints::multi_array<TC> ma_aofock) : 
        m_dev(dev), m_reg(m_reg0), m_scf_type(scf_type), m_b1(b1), m_b3(b3), m_b4(b4),
        m_av_aos(av_aos), m_av_pqinvhalf(av_pqinvhalf), m_ns(av_orb), m_ma_aofock(ma_aofock) { }
    
    /** \brief Compute EOM energy
        \param[in] av_buff Shared memory buffer
        \param[in,out] ma_c MO coeff
        \param[out] av_ene EOM energies
     **/
    void perform(
        libqints::array_view<TC> av_buff,
        libqints::multi_array<TC> ma_c,
        libqints::multi_array<TC> ma_oe,
        Mat<TC> &t1a, Mat<TC> &t1b,
        Mat<TC> &r1a, Mat<TC> &r1b,
        double &ene_exci, arma::vec Ea, arma::vec Eb,
        Mat<TC> &sigma_a, Mat<TC> &sigma_b, 
        unsigned algo, int dig);    
    
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


}; // class ri_eomee_unr_energy_r
} // namespace libgmbpt

#endif // LIBGMBPT_RI_EOMEE_UNR_ENERGY_R_H
