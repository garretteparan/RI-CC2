#include <libqints/arrays/memory_pool.h>
#include <libgscf/scf/posthf_kernels.h>
#include <libposthf/motran/motran.h>
#include <libposthf/motran/moints_2e4c_incore.h>
#include "cc2.h"
#include "cc2_energy.h"
#include <complex>
using namespace std;
namespace libgmbpt {
using namespace libqints;
using namespace libposthf;
using namespace libgscf;
using libham::scf_type;

template<typename TC, typename TI>
void cc2_energy<TC, TI>::perform(array_view<TC> av_buff,  multi_array<TC> ma_c,
                             multi_array<TC> ma_oe, Mat<TC> &t1a, Mat<TC> &t1b,
                             array_view<TC> av_ene, arma::Col<TC> Ea, arma::Col<TC> Eb,
                             double c_os, double c_ss)
{

    // Check prereqs
    if(m_scf_type != scf_type::RSCF && m_scf_type != scf_type::USCF)
        throw std::runtime_error(" cc2_energy(): Unknown SCF type!");

    const size_t nbsf = m_b1.get_nbsf();

    const size_t n_vovo_a = m_ns.vaca * m_ns.oaca * m_ns.vaca * m_ns.oaca;
    const size_t n_vovo_b = m_ns.vacb * m_ns.oacb * m_ns.vacb * m_ns.oacb;
    const size_t n_vovo_ab = m_ns.vaca * m_ns.oaca * m_ns.vacb * m_ns.oacb;

    const size_t n_ovvv_a = m_ns.oaca * m_ns.vaca * m_ns.vaca * m_ns.vaca;
    const size_t n_ovvv_b = m_ns.oacb * m_ns.vacb * m_ns.vacb * m_ns.vacb;
    const size_t n_ovvv_ab = m_ns.oaca * m_ns.vaca * m_ns.vacb * m_ns.vacb;
    const size_t n_ovvv_ba = m_ns.oacb * m_ns.vacb * m_ns.vaca * m_ns.vaca;

    const size_t n_ovoo_a = m_ns.oaca * m_ns.vaca * m_ns.oaca * m_ns.oaca;
    const size_t n_ovoo_b = m_ns.oacb * m_ns.vacb * m_ns.oacb * m_ns.oacb;
    const size_t n_ovoo_ab = m_ns.oaca * m_ns.vaca * m_ns.oacb * m_ns.oacb;
    const size_t n_ovoo_ba = m_ns.oacb * m_ns.vacb * m_ns.oaca * m_ns.oaca;

    if(av_buff.size() < required_memory())
        throw std::runtime_error(" cc2_energy(): requires more memory!");

    // Initialize objects
    memory_pool<TC> mem(av_buff);
    unsigned unr = (m_scf_type == scf_type::USCF);
    arma::Mat<TC>
        // Molecular Orbital Coefficents
        Ca(arrays<TC>::ptr(ma_c[0]), nbsf, m_ns.nmo, false, true),
        Cb(arrays<TC>::ptr(ma_c[unr]), nbsf, m_ns.nmo, false, true),
        // The One-electron Matrix
        OEa(arrays<TC>::ptr(ma_oe[0]), nbsf, nbsf, false, true),
        OEb(arrays<TC>::ptr(ma_oe[unr]), nbsf, nbsf, false, true);
    TC ene_aaaa(0.), ene_bbbb(0.), ene_aabb(0.);


    //Alpha virtual and occupied
    arma::Mat<TC> CvirtA = Ca.cols(m_ns.occa, m_ns.occa + m_ns.vaca - 1);
    arma::Mat<TC> CoccA = Ca.cols(m_ns.ofza, m_ns.occa - 1);

    // particle and hole
    arma::Mat<TC> Lam_pA (nbsf, m_ns.vaca, fill::zeros);
    Lam_pA = CvirtA - (CoccA * t1a.st());
    arma::Mat<TC> Lam_hA (nbsf, m_ns.occa, fill::zeros);
    Lam_hA = CoccA + (CvirtA * t1a);


    // Initialize memory buffer for V
    multi_array<TC> V(18);
    // RSCF
        V.set(0, mem.alloc(n_vovo_a)); // unmodified (ai|bj) - Energy
        V.set(1, mem.alloc(n_vovo_a)); // modified (ai|bj) - t2
        V.set(2, mem.alloc(n_ovvv_a)); // modified (jb|ca) - G
        V.set(3, mem.alloc(n_ovoo_a)); // modified (jb|ik)/(ia|kk) - H/F1
        V.set(4, mem.alloc(n_ovoo_a)); // modified (ai|kk) - F2

    // unmodified (ai|bj) - vovo w0
    arma::Mat<TC> V_vovo_u_a(arrays<TC>::ptr(V[0]), m_ns.vaca * m_ns.oaca, m_ns.vaca * m_ns.oaca, false, true);
    // modified (ai|bj) - VOVO w1
    arma::Mat<TC> V_vovo_a(arrays<TC>::ptr(V[1]), m_ns.vaca * m_ns.oaca, m_ns.vaca * m_ns.oaca, false, true);
    // modified (jb|ca) - ovVv w2
    arma::Mat<TC> V_ovvv_a(arrays<TC>::ptr(V[2]), m_ns.oaca * m_ns.vaca, m_ns.vaca * m_ns.vaca, false, true);
    // modified (jb|ik) & (ia|kk) - ovoO w3
    arma::Mat<TC> V_ovoo_a(arrays<TC>::ptr(V[3]), m_ns.oaca * m_ns.vaca, m_ns.oaca * m_ns.oaca, false, true);
    // modified (ai|kk) - VOoO w4
    arma::Mat<TC> V_vooo_a(arrays<TC>::ptr(V[4]), m_ns.vaca * m_ns.oaca, m_ns.oaca * m_ns.oaca, false, true);

    // Initialize to zero
    V_vovo_u_a.zeros(n_vovo_a);
    V_vovo_a.zeros(n_vovo_a);
    V_ovvv_a.zeros(n_ovvv_a);
    V_ovoo_a.zeros(n_ovoo_a);
    V_vooo_a.zeros(n_ovoo_a);

    typename memory_pool<TC>::checkpoint chkpt = mem.save_state();


    // Transform AO to Modified MOs using moints_2e4c

    // unmodified (ai|bj) - vovo w0
    moints_2e4c_incore<TC, TI>(m_b1,m_b4,CvirtA,CoccA,CvirtA,CoccA,V_vovo_u_a,m_dev,gto::lex);
    mem.load_state(chkpt);

    // modified (ai|bj) - VOVO w1
    moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pA,Lam_hA,Lam_pA,Lam_hA,V_vovo_a,m_dev,gto::lex);
    mem.load_state(chkpt);

    // modified (jb|ca) - ovVv w2
    moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccA,CvirtA,Lam_pA,CvirtA,V_ovvv_a,m_dev,gto::lex);
    mem.load_state(chkpt);

    // modified (jb|ik) (ia|kk) - ovoO w3
    moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccA,CvirtA,CoccA,Lam_hA,V_ovoo_a,m_dev,gto::lex);
    mem.load_state(chkpt);

    // modified (ai|kk) - VOoO w4
    moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pA,Lam_hA,CoccA,Lam_hA,V_vooo_a,m_dev,gto::lex);
    mem.load_state(chkpt);

    // Convert the one-electron matrix from AO to MO basis
    arma::Mat<TC> H1_a (m_ns.oaca, m_ns.vaca, fill::zeros);
    arma::Mat<TC> H2_a (m_ns.vaca, m_ns.oaca, fill::zeros);
    H1_a = CoccA.st() * OEa * CvirtA;
    H2_a = Lam_pA.st() * OEa * Lam_hA;


    // Compute CC2 energy from different spin blocks
    arma::Col<TC> Eacta = Ea.subvec(m_ns.ofza, m_ns.occa + m_ns.vaca - 1);
    if(m_scf_type == scf_type::RSCF) { // RHF is 2 * alpha part

        cc2<TC>(m_reg).restricted_energy(ene_aabb, ene_aaaa, m_ns.oaca, m_ns.vaca,
                                         V_vovo_u_a, V_vovo_a, V_ovvv_a, V_ovoo_a, V_vooo_a,
                                         H1_a, H2_a, t1a, Eacta, c_os, c_ss);
        ene_bbbb = ene_aaaa / 2.0;
        ene_aaaa = ene_aaaa / 2.0;
        t1b = t1a;

    }


    // Unrestricted
    else if(m_scf_type == scf_type::USCF) {


        //Beta virtual and occupied
        arma::Mat<TC> CvirtB = Cb.cols(m_ns.occb, m_ns.occb + m_ns.vacb - 1);
        arma::Mat<TC> CoccB = Cb.cols(m_ns.ofzb, m_ns.occb - 1);

        // Beta particle and hole
        arma::Mat<TC> Lam_pB (nbsf, m_ns.vacb, fill::zeros);
        Lam_pB = CvirtB - (CoccB * t1b.st());
        arma::Mat<TC> Lam_hB (nbsf, m_ns.occb, fill::zeros);
        Lam_hB = CoccB + (CvirtB * t1b);

        // declare variables
        V.set(5, mem.alloc(n_vovo_b));
        V.set(6, mem.alloc(n_vovo_ab));
        V.set(7, mem.alloc(n_vovo_b));
        V.set(8, mem.alloc(n_vovo_ab));
        V.set(9, mem.alloc(n_ovvv_b));
        V.set(10, mem.alloc(n_ovvv_ab));
        V.set(11, mem.alloc(n_ovvv_ba));
        V.set(12, mem.alloc(n_ovoo_b));
        V.set(13, mem.alloc(n_ovoo_ab));
        V.set(14, mem.alloc(n_ovoo_ba));
        V.set(15, mem.alloc(n_ovoo_b));
        V.set(16, mem.alloc(n_ovoo_ab));
        V.set(17, mem.alloc(n_ovoo_ba));

        // unmodified (ai|bj) - vovo w0
        arma::Mat<TC> V_vovo_u_b(arrays<TC>::ptr(V[5]), m_ns.vacb * m_ns.oacb, m_ns.vacb * m_ns.oacb, false, true);
        arma::Mat<TC> V_vovo_u_ab(arrays<TC>::ptr(V[6]), m_ns.vaca * m_ns.oaca, m_ns.vacb * m_ns.oacb, false, true);
        // modified (ai|bj) - VOVO w1
        arma::Mat<TC> V_vovo_b(arrays<TC>::ptr(V[7]), m_ns.vacb * m_ns.oacb, m_ns.vacb * m_ns.oacb, false, true);
        arma::Mat<TC> V_vovo_ab(arrays<TC>::ptr(V[8]), m_ns.vaca * m_ns.oaca, m_ns.vacb * m_ns.oacb, false, true);
        // modified (jb|ca) - ovVv w2
        arma::Mat<TC> V_ovvv_b(arrays<TC>::ptr(V[9]), m_ns.oacb * m_ns.vacb, m_ns.vacb * m_ns.vacb, false, true);
        arma::Mat<TC> V_ovvv_ab(arrays<TC>::ptr(V[10]), m_ns.oaca * m_ns.vaca, m_ns.vacb * m_ns.vacb, false, true);
        arma::Mat<TC> V_ovvv_ba(arrays<TC>::ptr(V[11]), m_ns.oacb * m_ns.vacb, m_ns.vaca * m_ns.vaca, false, true);
        // modified (jb|ik) & (ia|kk) - ovoO w3
        arma::Mat<TC> V_ovoo_b(arrays<TC>::ptr(V[12]), m_ns.oacb * m_ns.vacb, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> V_ovoo_ab(arrays<TC>::ptr(V[13]), m_ns.oaca * m_ns.vaca, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> V_ovoo_ba(arrays<TC>::ptr(V[14]), m_ns.oacb * m_ns.vacb, m_ns.oaca * m_ns.oaca, false, true);
        // modified (ai|kk) - VOoO w4
        arma::Mat<TC> V_vooo_b(arrays<TC>::ptr(V[15]), m_ns.vacb * m_ns.oacb, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> V_vooo_ab(arrays<TC>::ptr(V[16]), m_ns.vaca * m_ns.oaca, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> V_vooo_ba(arrays<TC>::ptr(V[17]), m_ns.vacb * m_ns.oacb, m_ns.oaca * m_ns.oaca, false, true);

        // Initialize to zero
        V_vovo_u_b.zeros(n_vovo_b);
        V_vovo_u_ab.zeros(n_vovo_ab);
        V_vovo_b.zeros(n_vovo_b);
        V_vovo_ab.zeros(n_vovo_ab);
        V_ovvv_b.zeros(n_ovvv_b);
        V_ovvv_ab.zeros(n_ovvv_ab);
        V_ovvv_ba.zeros(n_ovvv_ba);
        V_ovoo_b.zeros(n_ovoo_b);
        V_ovoo_ab.zeros(n_ovoo_ab);
        V_ovoo_ba.zeros(n_ovoo_ba);
        V_vooo_b.zeros(n_ovoo_b);
        V_vooo_ab.zeros(n_ovoo_ab);
        V_vooo_ba.zeros(n_ovoo_ba);

        typename memory_pool<TC>::checkpoint chkpt = mem.save_state();

        // Transform AO to Modified MOs using moints_2e4c

        // unmodified (ai|bj) - vovo w0
        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CvirtB,CoccB,CvirtB,CoccB,V_vovo_u_b,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CvirtA,CoccA,CvirtB,CoccB,V_vovo_u_ab,m_dev,gto::lex);
        mem.load_state(chkpt);

        // modified (ai|bj) - VOVO w1
        moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pB,Lam_hB,Lam_pB,Lam_hB,V_vovo_b,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pA,Lam_hA,Lam_pB,Lam_hB,V_vovo_ab,m_dev,gto::lex);
        mem.load_state(chkpt);

        // modified (jb|ca) - ovVv w2
        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccB,CvirtB,Lam_pB,CvirtB,V_ovvv_b,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccA,CvirtA,Lam_pB,CvirtB,V_ovvv_ab,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccB,CvirtB,Lam_pA,CvirtA,V_ovvv_ba,m_dev,gto::lex);
        mem.load_state(chkpt);

        // modified (jb|ik) (ia|kk) - ovoO w3
        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccB,CvirtB,CoccB,Lam_hB,V_ovoo_b,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccA,CvirtA,CoccB,Lam_hB,V_ovoo_ab,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,CoccB,CvirtB,CoccA,Lam_hA,V_ovoo_ba,m_dev,gto::lex);
        mem.load_state(chkpt);

        // modified (ai|kk) - VOoO w4
        moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pB,Lam_hB,CoccB,Lam_hB,V_vooo_b,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pA,Lam_hA,CoccB,Lam_hB,V_vooo_ab,m_dev,gto::lex);
        mem.load_state(chkpt);

        moints_2e4c_incore<TC, TI>(m_b1,m_b4,Lam_pB,Lam_hB,CoccA,Lam_hA,V_vooo_ba,m_dev,gto::lex);
        mem.load_state(chkpt);


        // Convert the one-electron matrix from AO to MO basis
        arma::Mat<TC> H1_b (m_ns.oaca, m_ns.vaca, fill::zeros);
        arma::Mat<TC> H2_b (m_ns.vaca, m_ns.oaca, fill::zeros);
        H1_b = CoccB.st() * OEa * CvirtB;
        H2_b = Lam_pB.st() * OEa * Lam_hB;

        arma::Col<TC> Eactb = Eb.subvec(m_ns.ofzb, m_ns.occb + m_ns.vacb - 1);
        cc2<TC>(m_reg).unrestricted_energy(ene_aabb, ene_aaaa, ene_bbbb,
                                           m_ns.oaca, m_ns.vaca, m_ns.oacb, m_ns.vacb,
                                           V_vovo_u_a, V_vovo_u_b, V_vovo_u_ab,
                                           V_vovo_a, V_vovo_b, V_vovo_ab,
                                           V_ovvv_a, V_ovvv_b, V_ovvv_ab, V_ovvv_ba,
                                           V_ovoo_a, V_ovoo_b, V_ovoo_ab, V_ovoo_ba,
                                           V_vooo_a, V_vooo_b, V_vooo_ab, V_vooo_ba,
                                           H1_a, H2_a, H1_b, H2_b, t1a, t1b, Eacta, Eactb, c_os, c_ss);
    }

    // Save result
    av_ene[0] = ene_aaaa;
    av_ene[1] = ene_bbbb;
    av_ene[2] = ene_aabb;
    av_ene[3] = ene_aaaa + ene_bbbb + ene_aabb;

}

template<typename TC, typename TI>
size_t cc2_energy<TC, TI>::required_memory() {

    const size_t n_vovo_a = aligned_length(m_ns.vaca * m_ns.oaca * m_ns.vaca * m_ns.oaca);
    const size_t n_vovo_b = aligned_length(m_ns.vacb * m_ns.oacb * m_ns.vacb * m_ns.oacb);
    const size_t n_vovo_ab = aligned_length(m_ns.vaca * m_ns.oaca * m_ns.vacb * m_ns.oacb);
    const size_t n_ovvv_a = aligned_length(m_ns.oaca * m_ns.vaca * m_ns.vaca * m_ns.vaca);
    const size_t n_ovvv_b = aligned_length(m_ns.oacb * m_ns.vacb * m_ns.vacb * m_ns.vacb);
    const size_t n_ovvv_ab = aligned_length(m_ns.oaca * m_ns.vaca * m_ns.vacb * m_ns.vacb);
    const size_t n_ovvv_ba = aligned_length(m_ns.oacb * m_ns.vacb * m_ns.vaca * m_ns.vaca);
    const size_t n_ovoo_a = aligned_length(m_ns.oaca * m_ns.vaca * m_ns.oaca * m_ns.oaca);
    const size_t n_ovoo_b = aligned_length(m_ns.oacb * m_ns.vacb * m_ns.oacb * m_ns.oacb);
    const size_t n_ovoo_ab = aligned_length(m_ns.oaca * m_ns.vaca * m_ns.oacb * m_ns.oacb);
    const size_t n_ovoo_ba = aligned_length(m_ns.oacb * m_ns.vacb * m_ns.oaca * m_ns.oaca);

    if(m_scf_type == scf_type::USCF)
        return 2*n_vovo_a + 2*n_vovo_b + 2*n_vovo_ab + n_ovvv_a + n_ovvv_b + n_ovvv_ab + n_ovvv_ba
                + 2*n_ovoo_a + 2*n_ovoo_b + 2*n_ovoo_ab + 2*n_ovoo_ba;
    return 2*n_vovo_a + n_ovvv_a + 2*n_ovoo_a;

}


//template class cap_cc2_energy<double>;
template class cc2_energy<double, double>;
template class cc2_energy<complex<double>, double>;
template class cc2_energy<complex<double>, complex<double>>;

} // namespace libgmbpt

