#include <libqints/arrays/memory_pool.h>
#include <libgscf/scf/posthf_kernels.h>
#include <libposthf/motran/motran_2e3c.h>
#include "ri_eomee_sing_r.h"
#include "ri_eomee_trip_r.h"
#include "ri_eomee_energy_r.h"


namespace libgmbpt {
using namespace libqints;
using namespace libposthf;
using namespace libgscf;
using libham::scf_type;

template<typename TC, typename TI>
void ri_eomee_energy_r<TC,TI>::perform(array_view<TC> av_buff, multi_array<TC> ma_c, multi_array<TC> ma_oe,
                                       Mat<TC> &t1a, Mat<TC> &r1a, 
                                       size_t nsinglets, size_t ntriplets,
                                       double &ene_exci, arma::vec Ea, double c_os, double c_ss,
                                       Mat<TC> &sigma, unsigned algo, int dig) {

    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    const size_t nrrx = nbsf * nbsf * naux;
    
    const size_t novx_a = m_ns.oaca * m_ns.vaca * naux;
    const size_t noox_a = m_ns.oaca * m_ns.oaca * naux;
    const size_t nvvx_a = m_ns.vaca * m_ns.vaca * naux;
    
    if(av_buff.size() < required_memory()) 
        throw std::runtime_error(" ri_eomee_energy(): requires more memory!");
    
    // Initialize objects
    memory_pool<TC> mem(av_buff);
    arma::Mat<TC>
        // The Fock matrix
        Fa(arrays<TC>::ptr(m_ma_aofock[0]), nbsf, nbsf, false, true),
        // Molecular Orbital Coefficents
        Ca(arrays<TC>::ptr(ma_c[0]), nbsf, m_ns.nmo, false, true);
    
    
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
    
    // lambda_bar particle and hole
    arma::Mat<TC> Lam_pA_bar (nbsf, m_ns.vaca, fill::zeros);
    Lam_pA_bar = - CoccA * r1a.t(); // -(orb,occ)*(vir,occ).t
    arma::Mat<TC> Lam_hA_bar (nbsf, m_ns.occa, fill::zeros);
    Lam_hA_bar = CvirtA * r1a; //(orb,vir)*(vir,occ)

    arma::Mat<TC> Lam_hA_prime = CvirtA * t1a;
    
    // Convert the Fock matrix from AO to MO basis
    arma::Mat<TC> f_vv_a (m_ns.vaca, m_ns.vaca, fill::zeros);
    arma::Mat<TC> f_oo_a (m_ns.oaca, m_ns.oaca, fill::zeros);
    f_vv_a = CvirtA.st() * Fa * CvirtA;
    f_oo_a = CoccA.st() * Fa * CoccA;
     
    // Compute RICC2 energy from different spin blocks
    arma::vec Eacta = Ea.subvec(m_ns.ofza, m_ns.occa + m_ns.vaca - 1);


    // this is the default algorithm (without digestor)
    if (dig == 0) {    
        
        /// step 2: form the different B matrices
        // Initialize memory buffer for BQai
        multi_array<TC> ma_BQmn_res(12);
        // RSCF
        ma_BQmn_res.set(0, mem.alloc(novx_a)); // B_ov
        ma_BQmn_res.set(1, mem.alloc(novx_a)); // B_vo
        ma_BQmn_res.set(2, mem.alloc(novx_a)); // B_hp
        ma_BQmn_res.set(3, mem.alloc(noox_a)); // B_oh
        ma_BQmn_res.set(4, mem.alloc(noox_a)); // B_ho
        ma_BQmn_res.set(5, mem.alloc(noox_a)); // B_oo
        ma_BQmn_res.set(6, mem.alloc(noox_a)); // B_ob
        ma_BQmn_res.set(7, mem.alloc(novx_a)); // B_po
        ma_BQmn_res.set(8, mem.alloc(novx_a)); // B_hb
        ma_BQmn_res.set(9, mem.alloc(novx_a)); // B_bp
        ma_BQmn_res.set(10, mem.alloc(nvvx_a)); // B_pv
        
        // declare variables
        arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn_res[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn_res[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn_res[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn_res[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQho_a(arrays<TC>::ptr(ma_BQmn_res[4]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn_res[5]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQob_a(arrays<TC>::ptr(ma_BQmn_res[6]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQpo_a(arrays<TC>::ptr(ma_BQmn_res[7]), naux, m_ns.vaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQhb_a(arrays<TC>::ptr(ma_BQmn_res[8]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQbp_a(arrays<TC>::ptr(ma_BQmn_res[9]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQpv_a(arrays<TC>::ptr(ma_BQmn_res[10]), naux, m_ns.vaca * m_ns.vaca, false, true);
        
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
        //BQob
        compute_BQmn(BQob_a, mem.alloc(noox_a), CoccA, Lam_hA_bar);
        mem.load_state(chkpt);
        //BQpo
        compute_BQmn(BQpo_a, mem.alloc(novx_a), Lam_pA, CvirtA*t1a);
        mem.load_state(chkpt);
        //BQhb
        compute_BQmn(BQhb_a, mem.alloc(novx_a), Lam_hA, Lam_pA_bar);
        mem.load_state(chkpt);
        //BQbp
        compute_BQmn(BQbp_a, mem.alloc(novx_a), Lam_hA_bar, Lam_pA);
        mem.load_state(chkpt);
        //BQpv
        compute_BQmn(BQpv_a, mem.alloc(nvvx_a), Lam_pA, CvirtA);
        mem.load_state(chkpt);

        
        // memory allocation for VPab
        array_view<TC> av_buff_ao(mem.alloc(nrrx));
        arma::Mat<double> Unit(nbsf, nbsf, fill::eye);
        std::vector<size_t> vblst(1);
        idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
        blst.populate();
        op_coulomb op;
        {
            motran_2e3c_incore_result_container<TC> buf(av_buff_ao);
            scr_null<bat_2e3c_shellpair_cgto<double>> scr;
            motran_2e3c<TC, double> mot(op, m_b3, scr, m_dev);
            mot.set_trn(Unit);
            mot.run(m_dev, blst, buf);
        }

        arma::Mat<TC> V_Pab(arrays<TC>::ptr(av_buff_ao), Unit.n_cols * Unit.n_cols, naux, false, true);
        mem.load_state(chkpt);
        

        if (nsinglets > 0) {

            if (algo == -1) {
            ri_eomee_r<TC,TI>(m_reg).ccs_restricted_energy_singlets(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    BQpv_a, V_Pab, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } else if (algo == 0) {
            ri_eomee_r<TC,TI>(m_reg).davidson_restricted_energy_singlets(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    BQpv_a, V_Pab, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } else if (algo == 1) {
            ri_eomee_r<TC,TI>(m_reg).diis_restricted_energy_singlets(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    BQpv_a, V_Pab, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            }

        } 
        else if (ntriplets > 0) { 

            if (algo == -1) {
            ri_eomee_trip_r<TC,TI>(m_reg).ccs_restricted_energy_triplets(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    BQpv_a, V_Pab, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } else if (algo == 0) {
            ri_eomee_trip_r<TC,TI>(m_reg).davidson_restricted_energy_triplets(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    BQpv_a, V_Pab, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } else if (algo == 1) {
            ri_eomee_trip_r<TC,TI>(m_reg).diis_restricted_energy_triplets(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    BQpv_a, V_Pab, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } 
        }
    }
    
    // digestor activated
    else if (dig == 1) {

        /// step 2: form the different B matrices
        // Initialize memory buffer for BQai
        multi_array<TC> ma_BQmn_res(10);
        // RSCF
        ma_BQmn_res.set(0, mem.alloc(novx_a)); // B_ov
        ma_BQmn_res.set(1, mem.alloc(novx_a)); // B_vo
        ma_BQmn_res.set(2, mem.alloc(novx_a)); // B_hp
        ma_BQmn_res.set(3, mem.alloc(noox_a)); // B_oh
        ma_BQmn_res.set(4, mem.alloc(noox_a)); // B_ho
        ma_BQmn_res.set(5, mem.alloc(noox_a)); // B_oo
        ma_BQmn_res.set(6, mem.alloc(noox_a)); // B_ob
        ma_BQmn_res.set(7, mem.alloc(novx_a)); // B_po
        ma_BQmn_res.set(8, mem.alloc(novx_a)); // B_hb
        ma_BQmn_res.set(9, mem.alloc(novx_a)); // B_bp
        
        // declare variables
        arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn_res[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn_res[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn_res[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn_res[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQho_a(arrays<TC>::ptr(ma_BQmn_res[4]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn_res[5]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQob_a(arrays<TC>::ptr(ma_BQmn_res[6]), naux, m_ns.oaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQpo_a(arrays<TC>::ptr(ma_BQmn_res[7]), naux, m_ns.vaca * m_ns.oaca, false, true);
        arma::Mat<TC> BQhb_a(arrays<TC>::ptr(ma_BQmn_res[8]), naux, m_ns.oaca * m_ns.vaca, false, true);
        arma::Mat<TC> BQbp_a(arrays<TC>::ptr(ma_BQmn_res[9]), naux, m_ns.oaca * m_ns.vaca, false, true);
        
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
        //BQob
        compute_BQmn(BQob_a, mem.alloc(noox_a), CoccA, Lam_hA_bar);
        mem.load_state(chkpt);
        //BQpo
        compute_BQmn(BQpo_a, mem.alloc(novx_a), Lam_pA, CvirtA*t1a);
        mem.load_state(chkpt);
        //BQhb
        compute_BQmn(BQhb_a, mem.alloc(novx_a), Lam_hA, Lam_pA_bar);
        mem.load_state(chkpt);
        //BQbp
        compute_BQmn(BQbp_a, mem.alloc(novx_a), Lam_hA_bar, Lam_pA);
        mem.load_state(chkpt);
        

        if (nsinglets > 0) {

            if (algo == -1) {
            ri_eomee_r<TC,TI>(m_reg).ccs_restricted_energy_singlets_digestor(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } else if (algo == 0) {
            ri_eomee_r<TC,TI>(m_reg).davidson_restricted_energy_singlets_digestor(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            } else if (algo == 1) {
            ri_eomee_r<TC,TI>(m_reg).diis_restricted_energy_singlets_digestor(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                                    BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
                                                                    BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
                                                                    Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
                                                                    f_vv_a, f_oo_a, t1a, r1a, Eacta,
                                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
            }

        } 
        // else if (ntriplets > 0) { 

        //     if (algo == -1) {
        //     ri_eomee_trip_r<TC,TI>(m_reg).ccs_restricted_energy_triplets_digestor(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
        //                                                             BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
        //                                                             BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
        //                                                             Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
        //                                                             f_vv_a, f_oo_a, t1a, r1a, Eacta,
        //                                                             m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
        //     } else if (algo == 0) {
        //     ri_eomee_trip_r<TC,TI>(m_reg).davidson_restricted_energy_triplets_digestor(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
        //                                                             BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
        //                                                             BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
        //                                                             Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
        //                                                             f_vv_a, f_oo_a, t1a, r1a, Eacta,
        //                                                             m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
        //     } else if (algo == 1) {
        //     ri_eomee_trip_r<TC,TI>(m_reg).diis_restricted_energy_triplets_digestor(ene_exci, m_ns.oaca, m_ns.vaca, naux, nbsf,
        //                                                             BQov_a, BQvo_a, BQhp_a, BQoh_a, BQho_a, 
        //                                                             BQoo_a, BQob_a, BQpo_a, BQhb_a, BQbp_a, 
        //                                                             Lam_hA, Lam_pA, Lam_hA_bar, Lam_pA_bar, CoccA, CvirtA,
        //                                                             f_vv_a, f_oo_a, t1a, r1a, Eacta,
        //                                                             m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss, sigma);
        //     } 
        // }
    }

}
    
template<typename TC, typename TI>
void ri_eomee_energy_r<TC,TI>::compute_BQmn(arma::Mat<TC> &BQmn,
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
size_t ri_eomee_energy_r<TC,TI>::required_memory() {
    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);
    
    return 6*novx_a + 4*noox_a + nvvx_a + nrrx;
    
}
    
template<typename TC, typename TI>
size_t ri_eomee_energy_r<TC,TI>::required_buffer_size() {
    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();  
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);
    
    return 6*novx_a + 4*noox_a + nvvx_a + nrrx;
    
}

template<typename TC, typename TI>
size_t ri_eomee_energy_r<TC,TI>::required_buffer_size_digestor() {
    
    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();  
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);
    
    return 7*novx_a + 4*noox_a;
    
}

template<typename TC, typename TI>
size_t ri_eomee_energy_r<TC,TI>::required_min_dynamic_memory() {
    
    op_coulomb op;
    std::vector<size_t> vblst(1);
    idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
    blst.populate();
    scr_null<bat_2e3c_shellpair_cgto<double>> scr;
    return motran_2e3c<TC, TI>(op, m_b3, scr, m_dev).min_memreq(blst);    
}


template class ri_eomee_energy_r<double, double>;
// template class ri_eomee_energy_r<std::complex<double> >;     // GPP: activate complex later

} // namespace libgmbpt

