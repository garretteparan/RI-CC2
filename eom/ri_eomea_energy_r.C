#include <libqints/arrays/memory_pool.h>
#include <libgscf/scf/posthf_kernels.h>
#include <libposthf/motran/motran_2e3c.h>
#include "ri_eomea_r.h"
#include "ri_eomea_energy_r.h"


namespace libgmbpt {
using namespace libqints;
using namespace libposthf;
using namespace libgscf;
using libham::scf_type;

template<typename TC, typename TI>
void ri_eomea_energy_r<TC,TI>::perform(array_view<TC> av_buff, multi_array<TC> ma_c, multi_array<TC> ma_oe,
                                        Mat<TC> &t1a, Col<TC> &r1, double &ene_exci, arma::vec Ea, 
                                        double c_os, double c_ss, Col<TC> &sigma, unsigned algo)
{
    // Check prereqs
    if(m_scf_type != scf_type::RSCF && m_scf_type != scf_type::USCF)
        throw std::runtime_error(" ri_eomea_energy(): Unknown SCF type!");

    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    const size_t nrrx = nbsf * nbsf * naux;
    
    const size_t novx_a = m_ns.oaca * m_ns.vaca * naux;
    const size_t noox_a = m_ns.oaca * m_ns.oaca * naux;
    const size_t nvvx_a = m_ns.vaca * m_ns.vaca * naux;
    

    if(av_buff.size() < required_memory()) 
        throw std::runtime_error(" ri_eomea_energy(): requires more memory!");
    
    // Initialize objects
    memory_pool<TC> mem(av_buff);
    unsigned unr = (m_scf_type == scf_type::USCF);
    arma::Mat<TC>
        // The Fock matrix
        Fa(arrays<TC>::ptr(m_ma_aofock[0]), nbsf, nbsf, false, true),
        Fb(arrays<TC>::ptr(m_ma_aofock[unr]), nbsf, nbsf, false, true),
        // Molecular Orbital Coefficents
        Ca(arrays<TC>::ptr(ma_c[0]), nbsf, m_ns.nmo, false, true),
        Cb(arrays<TC>::ptr(ma_c[unr]), nbsf, m_ns.nmo, false, true);
    
    
    /// GPP: RI-CC2 calculation Haettig's algorithm
    /// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)
    
    /// step 1: create coefficients and lambas (particle and hole)

    if(m_scf_type == scf_type::RSCF) { // RHF is 2 * alpha part

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
        multi_array<TC> ma_BQmn(7);
        // RSCF
        ma_BQmn.set(0, mem.alloc(novx_a)); // B_ov
        ma_BQmn.set(1, mem.alloc(novx_a)); // B_vo
        ma_BQmn.set(2, mem.alloc(novx_a)); // B_hp
        ma_BQmn.set(3, mem.alloc(novx_a)); // B_po
        ma_BQmn.set(4, mem.alloc(noox_a)); // B_oo
        ma_BQmn.set(5, mem.alloc(nvvx_a)); // B_vp
        
        // declare variables
        arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQpo_a(arrays<TC>::ptr(ma_BQmn[3]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn[4]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQvp_a(arrays<TC>::ptr(ma_BQmn[5]), naux, m_ns.vaca * m_ns.vaca, false, true);
        
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
        
        //BQpo
        compute_BQmn(BQpo_a, mem.alloc(novx_a), Lam_pA, CvirtA*t1a);
        mem.load_state(chkpt);
        
        //BQoo
        compute_BQmn(BQoo_a, mem.alloc(noox_a), CoccA, CvirtA*t1a);
        mem.load_state(chkpt); 
        
        //BQvp
        compute_BQmn(BQvp_a, mem.alloc(nvvx_a), CvirtA, Lam_pA);
        mem.load_state(chkpt);
        
        // memory allocation for VPab
        array_view<TC> av_buff_ao(mem.alloc(nrrx));
        

        // Compute RICC2 energy from different spin blocks
        arma::vec Eacta = Ea.subvec(m_ns.ofza, m_ns.occa + m_ns.vaca - 1);
        

        if (algo == -1) {
            ri_eomea_r<TC,TI>(m_reg).css_restricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                BQov_a, BQvo_a, BQhp_a, BQoo_a, BQpo_a, BQvp_a, 
                                                Lam_hA, Lam_pA, CoccA, CvirtA, f_vv_a, f_oo_a, t1a, r1, Eacta,
                                                av_buff_ao, m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
        } else if (algo == 0) {
            ri_eomea_r<TC,TI>(m_reg).davidson_restricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                BQov_a, BQvo_a, BQhp_a, BQoo_a, BQpo_a, BQvp_a, 
                                                Lam_hA, Lam_pA, CoccA, CvirtA, f_vv_a, f_oo_a, t1a, r1, Eacta,
                                                av_buff_ao, m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
        } else if (algo == 1) {
            ri_eomea_r<TC,TI>(m_reg).diis_restricted_energy(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                BQov_a, BQvo_a, BQhp_a, BQoo_a, BQpo_a, BQvp_a, 
                                                Lam_hA, Lam_pA, CoccA, CvirtA, f_vv_a, f_oo_a, t1a, r1, Eacta,
                                                av_buff_ao, m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
        }
    }

}
    
template<typename TC, typename TI>
void ri_eomea_energy_r<TC,TI>::compute_BQmn(arma::Mat<TC> &BQmn,
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
size_t ri_eomea_energy_r<TC,TI>::required_memory() {
    
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
    return 4*novx_a + noox_a + nvvx_a + nrrx;
    // return 4*novx_a + noox_a + 2*nvvx_a;
    
}
    
template<typename TC, typename TI>
size_t ri_eomea_energy_r<TC,TI>::required_buffer_size() {
    
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
    return 4*novx_a + noox_a + nvvx_a + nrrx;
    // return 4*novx_a + noox_a + 2*nvvx_a;
    
}

template<typename TC, typename TI>
size_t ri_eomea_energy_r<TC,TI>::required_min_dynamic_memory() {
    
    op_coulomb op;
    std::vector<size_t> vblst(1);
    idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
    blst.populate();
    scr_null<bat_2e3c_shellpair_cgto<double>> scr;
    return motran_2e3c<TC, TI>(op, m_b3, scr, m_dev).min_memreq(blst);
    
}


template class ri_eomea_energy_r<double, double>;
// template class ri_eomea_energy_r<std::complex<double> >;     // GPP: activate complex later

} // namespace libgmbpt

