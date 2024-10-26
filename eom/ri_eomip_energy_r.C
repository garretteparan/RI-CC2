#include <libqints/arrays/memory_pool.h>
#include <libgscf/scf/posthf_kernels.h>
#include <libposthf/motran/motran_2e3c.h>
#include "ri_eomip_r.h"
#include "ri_eomip_energy_r.h"


namespace libgmbpt {
using namespace libqints;
using namespace libposthf;
using namespace libgscf;
using libham::scf_type;

template<typename TC, typename TI>
void ri_eomip_energy_r<TC,TI>::perform(array_view<TC> av_buff, multi_array<TC> ma_c, multi_array<TC> ma_oe,
                                  Mat<TC> &t1a, Mat<TC> &t1b, Row<TC> &r1, Row<TC> &res_a, Row<TC> &res_b,
                                  double &ene_exci, arma::vec Ea, arma::vec Eb, double c_os, double c_ss,
                                  rowvec &sigma, unsigned algo) {
    // Check prereqs
    if(m_scf_type != scf_type::RSCF && m_scf_type != scf_type::USCF)
        throw std::runtime_error(" ri_eomip_energy(): Unknown SCF type!");
    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    const size_t nrrx = nbsf * nbsf * naux;
    
    const size_t novx_a = m_ns.oaca * m_ns.vaca * naux;
    const size_t noox_a = m_ns.oaca * m_ns.oaca * naux;
    const size_t nvvx_a = m_ns.vaca * m_ns.vaca * naux;
    
    // USCF
    const size_t novx_b = m_ns.oacb * m_ns.vacb * naux;
    const size_t noox_b = m_ns.oacb * m_ns.oacb * naux;
    const size_t nvvx_b = m_ns.vacb * m_ns.vacb * naux;
    
    
    if(av_buff.size() < required_memory()) 
        throw std::runtime_error(" ri_eomip_energy(): requires more memory!");
    
    // Initialize objects
    memory_pool<TC> mem(av_buff);
    unsigned unr = (m_scf_type == scf_type::USCF);
    arma::Mat<TC>
        // The Fock matrix
        Fa(arrays<TC>::ptr(m_ma_aofock[0]), nbsf, nbsf, false, true),
        Fb(arrays<TC>::ptr(m_ma_aofock[unr]), nbsf, nbsf, false, true),
        // Molecular Orbital Coefficents
        Ca(arrays<TC>::ptr(ma_c[0]), nbsf, m_ns.nmo, false, true),
        Cb(arrays<TC>::ptr(ma_c[unr]), nbsf, m_ns.nmo, false, true),
        // The One-electron Matrix
        OEa(arrays<TC>::ptr(ma_oe[0]), nbsf, nbsf, false, true),
        OEb(arrays<TC>::ptr(ma_oe[unr]), nbsf, nbsf, false, true);
    
    
    /// GPP: RI-CC2 calculation Haettig's algorithm
    /// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
    
    /// step 1: create coefficients and lambas (particle and hole)
    
    // Alpha virtual and occupied
    arma::Mat<TC> CvirtA = Ca.cols(m_ns.occa, m_ns.occa + m_ns.vaca - 1); //(orb,vir)
    arma::Mat<TC> CoccA = Ca.cols(m_ns.ofza, m_ns.occa - 1); //(orb,occ)
    
    // lambda particle and hole
    arma::Mat<TC> Lam_pA (nbsf, m_ns.vaca, fill::zeros);
    Lam_pA = CvirtA - (CoccA * t1a.t());
    arma::Mat<TC> Lam_hA (nbsf, m_ns.occa, fill::zeros);
    Lam_hA = CoccA + (CvirtA * t1a);
    
    // Convert the Fock matrix from AO to MO basis
    arma::Mat<TC> f_vv_a (m_ns.vaca, m_ns.vaca, fill::zeros);
    arma::Mat<TC> f_oo_a (m_ns.oaca, m_ns.oaca, fill::zeros);
    arma::Mat<TC> f_ov_a (m_ns.oaca, m_ns.vaca, fill::zeros);
    arma::Mat<TC> f_vo_a (m_ns.vaca, m_ns.oaca, fill::zeros);
    f_vv_a = CvirtA.st() * Fa * CvirtA;
    f_oo_a = CoccA.st() * Fa * CoccA;
    f_ov_a = CoccA.st() * Fa * CvirtA;
    f_vo_a = Lam_pA.st() * Fa * Lam_hA;


    /// step 2: form the different B matrices
    // Initialize memory buffer for BQai
    multi_array<TC> ma_BQmn(6);
    // RSCF
    ma_BQmn.set(0, mem.alloc(novx_a)); // B_ov
    ma_BQmn.set(1, mem.alloc(novx_a)); // B_vo
    ma_BQmn.set(2, mem.alloc(novx_a)); // B_hp
    ma_BQmn.set(3, mem.alloc(noox_a)); // B_oh
    ma_BQmn.set(4, mem.alloc(noox_a)); // B_ho
    ma_BQmn.set(5, mem.alloc(noox_a)); // B_oo

    // declare variables
    arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
    arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
    arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
    arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
    arma::Mat<TC> BQho_a(arrays<TC>::ptr(ma_BQmn[4]), naux, m_ns.oaca * m_ns.oaca, false, true);
    arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn[5]), naux, m_ns.oaca * m_ns.oaca, false, true);
    
    typename memory_pool<TC>::checkpoint chkpt = mem.save_state();

    // calculate integrals
    //BQov
    compute_BQmn(BQov_a, mem.alloc(novx_a), CoccA, CvirtA);
    mem.load_state(chkpt);
    //BQvo
    compute_BQmn(BQvo_a, mem.alloc(novx_a), CvirtA, CoccA);
    mem.load_state(chkpt);
    //BQhp
    compute_BQmn(BQhp_a, mem.alloc(novx_a), Lam_hA, Lam_pA);
    mem.load_state(chkpt);
    //BQoh
    compute_BQmn(BQoh_a, mem.alloc(noox_a), CoccA, Lam_hA);
    mem.load_state(chkpt);
    //BQho
    compute_BQmn(BQho_a, mem.alloc(noox_a), Lam_hA, CoccA);
    mem.load_state(chkpt);
    //BQoo
    compute_BQmn(BQoo_a, mem.alloc(noox_a), CoccA, CvirtA*t1a);
    mem.load_state(chkpt);
    
    // Compute RICC2 energy from different spin blocks
    arma::vec Eacta = Ea.subvec(m_ns.ofza, m_ns.occa + m_ns.vaca - 1);
    
    // if(m_scf_type == scf_type::RSCF) { // RHF is 2 * alpha part

        if (algo == -1) {
           ri_eomip_r<TC,TI>(m_reg).ccs_restricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                            BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, BQoo_a, 
                                            f_vv_a, f_oo_a, t1a, r1, res_a, Eacta,
                                            m_dev, m_b3, c_os, c_ss, sigma);
        } else if (algo == 0) {
           ri_eomip_r<TC,TI>(m_reg).davidson_restricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                            BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, BQoo_a, 
                                            f_vv_a, f_oo_a, t1a, r1, res_a, Eacta,
                                            m_dev, m_b3, c_os, c_ss, sigma);
        } else if (algo == 1) {
           ri_eomip_r<TC,TI>(m_reg).diis_restricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                            BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, BQoo_a, 
                                            f_vv_a, f_oo_a, t1a, r1, res_a, Eacta,
                                            m_dev, m_b3, c_os, c_ss, sigma);
        }
    // }
    
    /*
    // Unrestricted
    else if(m_scf_type == scf_type::USCF) {

        // Beta virtual and occupied
        arma::Mat<TC> CvirtB = Cb.cols(m_ns.occb, m_ns.occb + m_ns.vacb - 1);
        arma::Mat<TC> CoccB = Cb.cols(m_ns.ofzb, m_ns.occb - 1);

        // Beta particle and hole
        arma::Mat<TC> Lam_pB (nbsf, m_ns.vacb, fill::zeros);
        Lam_pB = CvirtB - (CoccB * t1b.t());
        arma::Mat<TC> Lam_hB (nbsf, m_ns.occb, fill::zeros);
        Lam_hB = CoccB + (CvirtB * t1b);
    
        // Beta lambda_bar particle and hole
        arma::Mat<TC> Lam_pB_bar (nbsf, m_ns.vacb, fill::zeros);
        Lam_pB_bar = - CoccB * r1b.t();
        arma::Mat<TC> Lam_hB_bar (nbsf, m_ns.occb, fill::zeros);
        Lam_hB_bar = CvirtB * r1b;
    
        // Convert the Fock matrix from AO to MO basis
        arma::Mat<TC> f_vv_b (m_ns.vacb, m_ns.vacb, fill::zeros);
        arma::Mat<TC> f_oo_b (m_ns.oacb, m_ns.oacb, fill::zeros);
        f_vv_b = CvirtB.st() * Fb * CvirtB;
        f_oo_b = CoccB.st() * Fb * CoccB;


        // declare variables
        ma_BQmn.set(11, mem.alloc(novx_b)); // B_ov
        ma_BQmn.set(12, mem.alloc(novx_b)); // B_vo
        ma_BQmn.set(13, mem.alloc(novx_b)); // B_ph
        ma_BQmn.set(14, mem.alloc(novx_b)); // B_hp
        ma_BQmn.set(15, mem.alloc(noox_b)); // B_oh
        ma_BQmn.set(16, mem.alloc(noox_b)); // B_ho
        ma_BQmn.set(17, mem.alloc(noox_b)); // B_oo
        ma_BQmn.set(18, mem.alloc(noox_b)); // B_ob
        ma_BQmn.set(19, mem.alloc(nvvx_b)); // B_pv
        ma_BQmn.set(20, mem.alloc(novx_b)); // B_po
        ma_BQmn.set(21, mem.alloc(novx_b)); // B_hb
        ma_BQmn.set(22, mem.alloc(novx_b)); // B_bp
        ma_BQmn.set(23, mem.alloc(novx_b)); // B_bh
        ma_BQmn.set(24, mem.alloc(novx_b)); // B_pb
        ma_BQmn.set(25, mem.alloc(novx_a)); // B_ph_a
        ma_BQmn.set(26, mem.alloc(novx_a)); // B_bh_a
        ma_BQmn.set(27, mem.alloc(novx_a)); // B_pb_a
        

    
        arma::Mat<TC> BQov_b(arrays<TC>::ptr(ma_BQmn[11]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQvo_b(arrays<TC>::ptr(ma_BQmn[12]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQph_b(arrays<TC>::ptr(ma_BQmn[13]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQhp_b(arrays<TC>::ptr(ma_BQmn[14]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQoh_b(arrays<TC>::ptr(ma_BQmn[15]), naux, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> BQho_b(arrays<TC>::ptr(ma_BQmn[16]), naux, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> BQoo_b(arrays<TC>::ptr(ma_BQmn[17]), naux, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> BQob_b(arrays<TC>::ptr(ma_BQmn[18]), naux, m_ns.oacb * m_ns.oacb, false, true);
        arma::Mat<TC> BQpv_b(arrays<TC>::ptr(ma_BQmn[19]), naux, m_ns.vacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQpo_b(arrays<TC>::ptr(ma_BQmn[20]), naux, m_ns.vacb * m_ns.oacb, false, true);
        arma::Mat<TC> BQhb_b(arrays<TC>::ptr(ma_BQmn[21]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQbp_b(arrays<TC>::ptr(ma_BQmn[22]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQbh_b(arrays<TC>::ptr(ma_BQmn[23]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQpb_b(arrays<TC>::ptr(ma_BQmn[24]), naux, m_ns.oacb * m_ns.vacb, false, true);
        arma::Mat<TC> BQph_a(arrays<TC>::ptr(ma_BQmn[25]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQbh_a(arrays<TC>::ptr(ma_BQmn[26]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQpb_a(arrays<TC>::ptr(ma_BQmn[27]), naux, m_ns.oaca * m_ns.vaca, false, true);


        typename memory_pool<TC>::checkpoint chkpt = mem.save_state();
    
        // calculate integrals
        //BQov
        compute_BQmn(BQov_b, mem.alloc(novx_b), CoccB, CvirtB);
        mem.load_state(chkpt);
        //BQvo
        compute_BQmn(BQvo_b, mem.alloc(novx_b), CvirtB, CoccB);
        mem.load_state(chkpt);
        //BQph
        compute_BQmn(BQph_b, mem.alloc(novx_b), Lam_pB, Lam_hB);
        mem.load_state(chkpt);
        //BQhp
        compute_BQmn(BQhp_b, mem.alloc(novx_b), Lam_hB, Lam_pB);
        mem.load_state(chkpt);
        //BQoh
        compute_BQmn(BQoh_b, mem.alloc(noox_b), CoccB, Lam_hB);
        mem.load_state(chkpt);
        //BQho
        compute_BQmn(BQho_b, mem.alloc(noox_b), Lam_hB, CoccB);
        mem.load_state(chkpt);
        //BQoo
        compute_BQmn(BQoo_b, mem.alloc(noox_b), CoccB, CvirtB*t1b);
        mem.load_state(chkpt);
        //BQob
        compute_BQmn(BQob_b, mem.alloc(noox_b), CoccB, Lam_hB_bar);
        mem.load_state(chkpt);
        //BQpv
        compute_BQmn(BQpv_b, mem.alloc(nvvx_b), Lam_pB, CvirtB);
        mem.load_state(chkpt);
        //BQpo
        compute_BQmn(BQpo_b, mem.alloc(novx_b), Lam_pB, CvirtB*t1b);
        mem.load_state(chkpt);
        //BQhb
        compute_BQmn(BQhb_b, mem.alloc(novx_b), Lam_hB, Lam_pB_bar);
        mem.load_state(chkpt);
        //BQbp
        compute_BQmn(BQbp_b, mem.alloc(novx_b), Lam_hB_bar, Lam_pB);
        mem.load_state(chkpt);
        //BQbh
        compute_BQmn(BQbh_b, mem.alloc(novx_b), Lam_pB_bar, Lam_hB);
        mem.load_state(chkpt);
        //BQpb
        compute_BQmn(BQpb_b, mem.alloc(novx_b), Lam_pB, Lam_hB_bar);
        mem.load_state(chkpt);

        //BQph
        compute_BQmn(BQph_a, mem.alloc(novx_a), Lam_pA, Lam_hA);
        mem.load_state(chkpt);
        // //BQbh
        compute_BQmn(BQbh_a, mem.alloc(novx_a), Lam_pA_bar, Lam_hA);
        mem.load_state(chkpt);
        // //BQpb
        compute_BQmn(BQpb_a, mem.alloc(novx_a), Lam_pA, Lam_hA_bar);
        mem.load_state(chkpt);


        arma::vec Eactb = Eb.subvec(m_ns.ofzb, m_ns.occb + m_ns.vacb - 1);

        ri_eom_r<TC,TI>(m_reg).unrestricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, 
                                                m_ns.oacb, m_ns.vacb, naux, nbsf,
                                                BQov_a, BQvo_a, BQph_a, BQhp_a, BQoh_a, BQho_a, 
                                                BQoo_a, BQob_a, BQpv_a, BQpo_a, BQhb_a, BQbp_a, BQbh_a, BQpb_a,
                                                BQov_b, BQvo_b, BQph_b, BQhp_b, BQoh_b, BQho_b, 
                                                BQoo_b, BQob_b, BQpv_b, BQpo_b, BQhb_b, BQbp_b, BQbh_b, BQpb_b,
                                                Lam_hA, Lam_pA, Lam_hB, Lam_pB, Lam_hA_bar, 
                                                Lam_pA_bar, Lam_hB_bar, Lam_pB_bar, 
                                                CoccA, CvirtA, CoccB, CvirtB,
                                                f_vv_a, f_oo_a, f_vv_b, f_oo_b, 
                                                t1a, t1b, r1a, r1b, res_a, res_b, Eacta, Eactb,
                                                av_buff_ao, m_av_pqinvhalf, m_dev, m_b3);

    }
    */

}
    
template<typename TC, typename TI>
void ri_eomip_energy_r<TC,TI>::compute_BQmn(arma::Mat<TC> &BQmn,
    array_view<TC> av_buff, arma::Mat<TC> Cocc, arma::Mat<TC> Cvir)
{
    const size_t naux = m_b3.get_ket().get_nbsf();
    std::vector<size_t> vblst(1);
    idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
    blst.populate();
    op_coulomb op;
    {
        motran_2e3c_incore_result_container<TC> buf(av_buff);
        scr_null<bat_2e3c_shellpair_cgto<double>> scr;
        motran_2e3c<TC, double> mot(op, m_b3, scr, m_dev);
        mot.set_trn(Cocc, Cvir);
        mot.run(m_dev, blst, buf);
    }
    arma::Mat<TC> VaiP(arrays<TC>::ptr(av_buff),
        Cocc.n_cols * Cvir.n_cols, naux, false, true);
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    arma::mat PQinvhalf(arrays<double>::ptr(m_av_pqinvhalf),
        naux, naux, false, true);
    BQmn = PQinvhalf * VaiP.t();
}
    
template<typename TC, typename TI>
size_t ri_eomip_energy_r<TC,TI>::required_memory() {
    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);
    const size_t novx_b = aligned_length(m_ns.oacb * m_ns.vacb * naux);
    const size_t noox_b = aligned_length(m_ns.oacb * m_ns.oacb * naux);
    const size_t nvvx_b = aligned_length(m_ns.vacb * m_ns.vacb * naux);
    
    // if(m_scf_type == scf_type::USCF)
    //     return 12*novx_a + 9*novx_b + 4*noox_a + 4*noox_b + nvvx_a + nvvx_b + 2*nrrx;
    return 4*novx_a + 3*noox_a;
    
}
    
template<typename TC, typename TI>
size_t ri_eomip_energy_r<TC,TI>::required_buffer_size() {
    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();  
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);
    const size_t novx_b = aligned_length(m_ns.oacb * m_ns.vacb * naux);
    const size_t noox_b = aligned_length(m_ns.oacb * m_ns.oacb * naux);
    const size_t nvvx_b = aligned_length(m_ns.vacb * m_ns.vacb * naux);
    
    // if(m_scf_type == scf_type::USCF)
    //     return 12*novx_a + 9*novx_b + 4*noox_a + 4*noox_b + nvvx_a + nvvx_b + 2*nrrx;
    return 4*novx_a + 3*noox_a;
    
}

template<typename TC, typename TI>
size_t ri_eomip_energy_r<TC,TI>::required_min_dynamic_memory() {
    
    op_coulomb op;
    std::vector<size_t> vblst(1);
    idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
    blst.populate();
    scr_null<bat_2e3c_shellpair_cgto<double>> scr;
    return motran_2e3c<TC, TI>(op, m_b3, scr, m_dev).min_memreq(blst);
    
}


template class ri_eomip_energy_r<double, double>;
// template class ri_eomip_energy_r<std::complex<double> >;     // GPP: activate complex later

} // namespace libgmbpt

