#include <libqints/arrays/memory_pool.h>
#include <libgscf/scf/posthf_kernels.h>
#include <libposthf/motran/motran_2e3c.h>
#include "ricc2.h"
#include "ricc2_energy.h"

namespace libgmbpt {
using namespace libqints;
using namespace libposthf;
using namespace libgscf;
using libham::scf_type;

template<typename TC, typename TI>
void ricc2_energy<TC, TI>::perform(array_view<TC> av_buff,
                               multi_array<TC> ma_c, Mat<TC> &t1a,
                               Mat<TC> &t1b, array_view<TC> av_ene,
                               arma::Col<TC> Ea, arma::Col<TC> Eb,
                               double c_os, double c_ss, int dig)
{
    // Check prereqs
    if(m_scf_type != scf_type::RSCF && m_scf_type != scf_type::USCF)
        throw std::runtime_error(" ricc2_energy(): Unknown SCF type!");

    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    const size_t nrrx = nbsf * nbsf * naux;

    const size_t novx_a = m_ns.oaca * m_ns.vaca * naux;
    const size_t noox_a = m_ns.oaca * m_ns.oaca * naux;
    const size_t novx_b = m_ns.oacb * m_ns.vacb * naux;
    const size_t noox_b = m_ns.oacb * m_ns.oacb * naux;

    // CU: not required
    //if(av_buff.size() < required_memory())
    //    throw std::runtime_error(" ricc2_energy(): requires more memory!");

    // Initialize objects
    memory_pool<TC> mem(av_buff);
    unsigned unr = (m_scf_type == scf_type::USCF);
    arma::Mat<TC> Ca(arrays<TC>::ptr(ma_c[0]), nbsf, m_ns.nmo, false, true),
                  Cb(arrays<TC>::ptr(ma_c[unr]), nbsf, m_ns.nmo, false, true);
    TC ene_aaaa(0.0), ene_bbbb(0.0), ene_aabb(0.0);

    /// GPP: RI-CC2 calculation Haettig's algorithm
    /// J. Chem. Phys. 113, 5154 (2000); doi: 10.1063/1.1290013 (see figure 1)

    /// step 1: create coefficients and lambas (particle and hole)

    // virtual and occupied
    arma::Mat<TC> CvirtA = Ca.cols(m_ns.occa, m_ns.occa + m_ns.vaca - 1); //(orb,vir)
    arma::Mat<TC> CoccA = Ca.cols(m_ns.ofza, m_ns.occa - 1); //(orb,occ)
    arma::Mat<TC> Unit(nbsf, nbsf, fill::eye);

    // particle and hole
    arma::Mat<TC> Lam_pA (nbsf, m_ns.vaca, fill::zeros);
    Lam_pA = CvirtA - (CoccA * t1a.st());
    arma::Mat<TC> Lam_hA (nbsf, m_ns.occa, fill::zeros);
    Lam_hA = CoccA + (CvirtA * t1a);

    // Compute RICC2 energy from different spin blocks
    arma::Col<TC> Eacta = Ea.subvec(m_ns.ofza, m_ns.occa + m_ns.vaca - 1);

    if(m_scf_type == scf_type::RSCF) { // RHF is 2 * alpha part

        // this is the default algorithm (without digestor)
        if (dig == 0) {
            
            /// step 2: form the different B matrices
            // Initialize memory buffer for BQai
            // GPP: for optimization:
            //      the first two integrals can be calculated only once and stored on the disk
            multi_array<TC> ma_BQmn_res(6);
            ma_BQmn_res.set(0, mem.alloc(novx_a)); // B_vo
            ma_BQmn_res.set(1, mem.alloc(novx_a)); // B_ov
            ma_BQmn_res.set(2, mem.alloc(novx_a)); // B_hp
            ma_BQmn_res.set(3, mem.alloc(noox_a)); // B_oo
            ma_BQmn_res.set(4, mem.alloc(noox_a)); // B_oh


            // declare variables
            arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn_res[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn_res[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn_res[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn_res[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
            arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn_res[4]), naux, m_ns.oaca * m_ns.oaca, false, true);


            typename memory_pool<TC>::checkpoint chkpt = mem.save_state();

            // calculate integrals

            //BQvo
            compute_BQmn(BQvo_a, mem.alloc(novx_a), CvirtA, CoccA);
            mem.load_state(chkpt);

            //BQov
            compute_BQmn(BQov_a, mem.alloc(novx_a), CoccA, CvirtA);
            mem.load_state(chkpt);

            //BQhp
            compute_BQmn(BQhp_a, mem.alloc(novx_a), Lam_hA, Lam_pA);
            mem.load_state(chkpt);

            //BQoo
            compute_BQmn(BQoo_a, mem.alloc(noox_a), CoccA, CvirtA*t1a);
            mem.load_state(chkpt);

            //BQoh
            compute_BQmn(BQoh_a, mem.alloc(noox_a), CoccA, Lam_hA);
            mem.load_state(chkpt);

            
            // memory allocation for VPab
            array_view<TC> av_buff(mem.alloc(nrrx));
            ma_BQmn_res.set(5, av_buff); // V_Pab
            std::vector<size_t> vblst(1);
            idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
            blst.populate();
            op_coulomb op;
            {
                motran_2e3c_incore_result_container<TC> buf(av_buff);
                scr_null<bat_2e3c_shellpair_cgto<TI>> scr;
                motran_2e3c<TC, TI> mot(op, m_b3, scr, m_dev);
                mot.set_trn(Unit, Unit);
                mot.run(m_dev, blst, buf);
            }
            arma::Mat<TC> V_Pab(arrays<TC>::ptr(av_buff), Unit.n_cols * Unit.n_cols, naux, false, true);
            mem.load_state(chkpt);
        

            ricc2<TC,TI>(m_reg).restricted_energy(ene_aabb, ene_aaaa, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                    BQvo_a, BQov_a, BQhp_a, BQoo_a, BQoh_a, V_Pab,
                                                    Lam_hA, Lam_pA, CoccA, CvirtA, t1a, Eacta,
                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss);
                                                    

            ene_bbbb = ene_aaaa / 2.0;
            ene_aaaa = ene_aaaa / 2.0;

        }
    
        // digestor activated
        else if (dig == 1) {

            /// step 2: form the different B matrices
            // Initialize memory buffer for BQai
            multi_array<TC> ma_BQmn_res(5);
            ma_BQmn_res.set(0, mem.alloc(novx_a)); // B_vo
            ma_BQmn_res.set(1, mem.alloc(novx_a)); // B_ov
            ma_BQmn_res.set(2, mem.alloc(novx_a)); // B_hp
            ma_BQmn_res.set(3, mem.alloc(noox_a)); // B_oo
            ma_BQmn_res.set(4, mem.alloc(noox_a)); // B_oh


            // declare variables
            arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn_res[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn_res[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn_res[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn_res[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
            arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn_res[4]), naux, m_ns.oaca * m_ns.oaca, false, true);


            typename memory_pool<TC>::checkpoint chkpt = mem.save_state();

            // calculate integrals

            //BQvo
            compute_BQmn(BQvo_a, mem.alloc(novx_a), CvirtA, CoccA);
            mem.load_state(chkpt);

            //BQov
            compute_BQmn(BQov_a, mem.alloc(novx_a), CoccA, CvirtA);
            mem.load_state(chkpt);

            //BQhp
            compute_BQmn(BQhp_a, mem.alloc(novx_a), Lam_hA, Lam_pA);
            mem.load_state(chkpt);

            //BQoo
            compute_BQmn(BQoo_a, mem.alloc(noox_a), CoccA, CvirtA*t1a);
            mem.load_state(chkpt);

            //BQoh
            compute_BQmn(BQoh_a, mem.alloc(noox_a), CoccA, Lam_hA);
            mem.load_state(chkpt);

            ricc2<TC,TI>(m_reg).restricted_energy_digestor(ene_aabb, ene_aaaa, m_ns.oaca, m_ns.vaca, naux, nbsf,
                                                    BQvo_a, BQov_a, BQhp_a, BQoo_a, BQoh_a,
                                                    Lam_hA, Lam_pA, CoccA, CvirtA, t1a, Eacta,
                                                    m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss);
                                         
            ene_bbbb = ene_aaaa / 2.0;
            ene_aaaa = ene_aaaa / 2.0;

        }

    }

    // The same as step 1 and 2 above but with the beta spin case
    else if(m_scf_type == scf_type::USCF) {

        /// step 1: create coefficients and lambas (particle and hole)

        // virtual and occupied
        arma::Mat<TC> CvirtB = Cb.cols(m_ns.occb, m_ns.occb + m_ns.vacb - 1);
        arma::Mat<TC> CoccB = Cb.cols(m_ns.ofzb, m_ns.occb - 1);

        // particle and hole
        arma::Mat<TC> Lam_pB (nbsf, m_ns.vacb, fill::zeros);
        Lam_pB = CvirtB - (CoccB * t1b.st());
        arma::Mat<TC> Lam_hB (nbsf, m_ns.occb, fill::zeros);
        Lam_hB = CoccB + (CvirtB * t1b);

        // this is the default algorithm (without digestor)
        if (dig == 0) {           
            
            /// step 2: form the different B matrices
            // Initialize memory buffer for BQai
            multi_array<TC> ma_BQmn_unr(11);

            // declare variables
            ma_BQmn_unr.set(0, mem.alloc(novx_a)); // B_vo
            ma_BQmn_unr.set(1, mem.alloc(novx_a)); // B_ov
            ma_BQmn_unr.set(2, mem.alloc(novx_a)); // B_hp
            ma_BQmn_unr.set(3, mem.alloc(noox_a)); // B_oo
            ma_BQmn_unr.set(4, mem.alloc(noox_a)); // B_oh
            ma_BQmn_unr.set(5, mem.alloc(novx_b)); // B_vo
            ma_BQmn_unr.set(6, mem.alloc(novx_b)); // B_ov
            ma_BQmn_unr.set(7, mem.alloc(novx_b)); // B_hp
            ma_BQmn_unr.set(8, mem.alloc(noox_b)); // B_oo
            ma_BQmn_unr.set(9, mem.alloc(noox_b)); // B_oh

            // declare variables
            arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn_unr[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn_unr[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn_unr[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn_unr[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
            arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn_unr[4]), naux, m_ns.oaca * m_ns.oaca, false, true);
            arma::Mat<TC> BQvo_b(arrays<TC>::ptr(ma_BQmn_unr[5]), naux, m_ns.oacb * m_ns.vacb, false, true);
            arma::Mat<TC> BQov_b(arrays<TC>::ptr(ma_BQmn_unr[6]), naux, m_ns.oacb * m_ns.vacb, false, true);
            arma::Mat<TC> BQhp_b(arrays<TC>::ptr(ma_BQmn_unr[7]), naux, m_ns.oacb * m_ns.vacb, false, true);
            arma::Mat<TC> BQoo_b(arrays<TC>::ptr(ma_BQmn_unr[8]), naux, m_ns.oacb * m_ns.oacb, false, true);
            arma::Mat<TC> BQoh_b(arrays<TC>::ptr(ma_BQmn_unr[9]), naux, m_ns.oacb * m_ns.oacb, false, true);

            typename memory_pool<TC>::checkpoint chkpt = mem.save_state();

            // calculate integrals
            
            //BQvo
            compute_BQmn(BQvo_a, mem.alloc(novx_a), CvirtA, CoccA);
            mem.load_state(chkpt);
            
            //BQov
            compute_BQmn(BQov_a, mem.alloc(novx_a), CoccA, CvirtA);
            mem.load_state(chkpt);
            
            //BQhp
            compute_BQmn(BQhp_a, mem.alloc(novx_a), Lam_hA, Lam_pA);
            mem.load_state(chkpt);
            
            //BQoo
            compute_BQmn(BQoo_a, mem.alloc(noox_a), CoccA, CvirtA*t1a);
            mem.load_state(chkpt);
            
            //BQoh
            compute_BQmn(BQoh_a, mem.alloc(noox_a), CoccA, Lam_hA);
            mem.load_state(chkpt);
            
            //BQvo
            compute_BQmn(BQvo_b, mem.alloc(novx_b), CvirtB, CoccB);
            mem.load_state(chkpt);
            
            //BQov
            compute_BQmn(BQov_b, mem.alloc(novx_b), CoccB, CvirtB);
            mem.load_state(chkpt);
            
            //BQhp
            compute_BQmn(BQhp_b, mem.alloc(novx_b), Lam_hB, Lam_pB);
            mem.load_state(chkpt);
            
            //BQoo
            compute_BQmn(BQoo_b, mem.alloc(noox_b), CoccB, CvirtB*t1b);
            mem.load_state(chkpt);
            
            //BQoh
            compute_BQmn(BQoh_b, mem.alloc(noox_b), CoccB, Lam_hB);
            mem.load_state(chkpt);

            //V_Pab
            array_view<TC> av_buff(mem.alloc(nrrx));
            ma_BQmn_unr.set(10, av_buff); // V_Pab
            std::vector<size_t> vblst(1);
            idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
            blst.populate();
            op_coulomb op;
            {
                motran_2e3c_incore_result_container<TC> buf(av_buff);
                scr_null<bat_2e3c_shellpair_cgto<TI>> scr;
                motran_2e3c<TC, TI> mot(op, m_b3, scr, m_dev);
                mot.set_trn(Unit, Unit);
                mot.run(m_dev, blst, buf);
            }
            arma::Mat<TC> V_Pab(arrays<TC>::ptr(av_buff), Unit.n_cols * Unit.n_cols, naux, false, true);
            mem.load_state(chkpt);

            arma::Col<TC> Eactb = Eb.subvec(m_ns.ofzb, m_ns.occb + m_ns.vacb - 1);

            ricc2<TC,TI>(m_reg).unrestricted_energy(ene_aabb, ene_aaaa, ene_bbbb,
                                                m_ns.oaca, m_ns.vaca, m_ns.oacb, m_ns.vacb, naux, nbsf,
                                                BQvo_a, BQov_a, BQhp_a, BQoo_a, BQoh_a,
                                                BQvo_b, BQov_b, BQhp_b, BQoo_b, BQoh_b, 
                                                V_Pab, Lam_hA, Lam_pA, Lam_hB, Lam_pB,
                                                CoccA, CvirtA, CoccB, CvirtB, t1a, t1b, Eacta, Eactb,
                                                m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss);


        }


        // digestor activated
        else if (dig == 1) {

            /// step 2: form the different B matrices
            // Initialize memory buffer for BQai
            multi_array<TC> ma_BQmn_unr(10);

            // declare variables
            ma_BQmn_unr.set(0, mem.alloc(novx_a)); // B_vo
            ma_BQmn_unr.set(1, mem.alloc(novx_a)); // B_ov
            ma_BQmn_unr.set(2, mem.alloc(novx_a)); // B_hp
            ma_BQmn_unr.set(3, mem.alloc(noox_a)); // B_oo
            ma_BQmn_unr.set(4, mem.alloc(noox_a)); // B_oh
            ma_BQmn_unr.set(5, mem.alloc(novx_b)); // B_vo
            ma_BQmn_unr.set(6, mem.alloc(novx_b)); // B_ov
            ma_BQmn_unr.set(7, mem.alloc(novx_b)); // B_hp
            ma_BQmn_unr.set(8, mem.alloc(noox_b)); // B_oo
            ma_BQmn_unr.set(9, mem.alloc(noox_b)); // B_oh

            // declare variables
            arma::Mat<TC> BQvo_a(arrays<TC>::ptr(ma_BQmn_unr[0]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQov_a(arrays<TC>::ptr(ma_BQmn_unr[1]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQhp_a(arrays<TC>::ptr(ma_BQmn_unr[2]), naux, m_ns.oaca * m_ns.vaca, false, true);
            arma::Mat<TC> BQoo_a(arrays<TC>::ptr(ma_BQmn_unr[3]), naux, m_ns.oaca * m_ns.oaca, false, true);
            arma::Mat<TC> BQoh_a(arrays<TC>::ptr(ma_BQmn_unr[4]), naux, m_ns.oaca * m_ns.oaca, false, true);
            arma::Mat<TC> BQvo_b(arrays<TC>::ptr(ma_BQmn_unr[5]), naux, m_ns.oacb * m_ns.vacb, false, true);
            arma::Mat<TC> BQov_b(arrays<TC>::ptr(ma_BQmn_unr[6]), naux, m_ns.oacb * m_ns.vacb, false, true);
            arma::Mat<TC> BQhp_b(arrays<TC>::ptr(ma_BQmn_unr[7]), naux, m_ns.oacb * m_ns.vacb, false, true);
            arma::Mat<TC> BQoo_b(arrays<TC>::ptr(ma_BQmn_unr[8]), naux, m_ns.oacb * m_ns.oacb, false, true);
            arma::Mat<TC> BQoh_b(arrays<TC>::ptr(ma_BQmn_unr[9]), naux, m_ns.oacb * m_ns.oacb, false, true);

            typename memory_pool<TC>::checkpoint chkpt = mem.save_state();

            // calculate integrals
            
            //BQvo
            compute_BQmn(BQvo_a, mem.alloc(novx_a), CvirtA, CoccA);
            mem.load_state(chkpt);
            
            //BQov
            compute_BQmn(BQov_a, mem.alloc(novx_a), CoccA, CvirtA);
            mem.load_state(chkpt);
            
            //BQhp
            compute_BQmn(BQhp_a, mem.alloc(novx_a), Lam_hA, Lam_pA);
            mem.load_state(chkpt);
            
            //BQoo
            compute_BQmn(BQoo_a, mem.alloc(noox_a), CoccA, CvirtA*t1a);
            mem.load_state(chkpt);
            
            //BQoh
            compute_BQmn(BQoh_a, mem.alloc(noox_a), CoccA, Lam_hA);
            mem.load_state(chkpt);
            
            //BQvo
            compute_BQmn(BQvo_b, mem.alloc(novx_b), CvirtB, CoccB);
            mem.load_state(chkpt);
            
            //BQov
            compute_BQmn(BQov_b, mem.alloc(novx_b), CoccB, CvirtB);
            mem.load_state(chkpt);
            
            //BQhp
            compute_BQmn(BQhp_b, mem.alloc(novx_b), Lam_hB, Lam_pB);
            mem.load_state(chkpt);
            
            //BQoo
            compute_BQmn(BQoo_b, mem.alloc(noox_b), CoccB, CvirtB*t1b);
            mem.load_state(chkpt);
            
            //BQoh
            compute_BQmn(BQoh_b, mem.alloc(noox_b), CoccB, Lam_hB);
            mem.load_state(chkpt);

            arma::Col<TC> Eactb = Eb.subvec(m_ns.ofzb, m_ns.occb + m_ns.vacb - 1);

            ricc2<TC,TI>(m_reg).unrestricted_energy_digestor(ene_aabb, ene_aaaa, ene_bbbb,
                                                m_ns.oaca, m_ns.vaca, m_ns.oacb, m_ns.vacb, naux, nbsf,
                                                BQvo_a, BQov_a, BQhp_a, BQoo_a, BQoh_a,
                                                BQvo_b, BQov_b, BQhp_b, BQoo_b, BQoh_b,
                                                Lam_hA, Lam_pA, Lam_hB, Lam_pB,
                                                CoccA, CvirtA, CoccB, CvirtB,
                                                t1a, t1b, Eacta, Eactb,
                                                m_av_pqinvhalf, m_dev, m_b3, c_os, c_ss);


        }

    }

    // Save result
    av_ene[0] = ene_aaaa;
    av_ene[1] = ene_bbbb;
    av_ene[2] = ene_aabb;
    av_ene[3] = ene_aaaa + ene_bbbb + ene_aabb;

}

template<typename TC, typename TI>
void ricc2_energy<TC, TI>::compute_BQmn(arma::Mat<TC> &BQmn,
    array_view<TC> av_buff, arma::Mat<TC> Cocc, arma::Mat<TC> Cvir)
{
    const size_t naux = m_b3.get_ket().get_nbsf();
    std::vector<size_t> vblst(1);
    idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
    blst.populate();
    op_coulomb op;
    {
        motran_2e3c_incore_result_container<TC> buf(av_buff);
        scr_null<bat_2e3c_shellpair_cgto<TI>> scr;
        motran_2e3c<TC, TI> mot(op, m_b3, scr, m_dev);
        mot.set_trn(Cocc, Cvir);
        mot.run(m_dev, blst, buf);
    }
    arma::Mat<TC> VaiP(arrays<TC>::ptr(av_buff),
        Cocc.n_cols * Cvir.n_cols, naux, false, true);
    const size_t nbsf = m_b3.get_bra().get_nbsfa();
    arma::Mat<TI> PQinvhalf(arrays<TI>::ptr(m_av_pqinvhalf),
        naux, naux, false, true);
    BQmn = PQinvhalf.st() * VaiP.st();
}

// CU: not used
template<typename TC, typename TI>
size_t ricc2_energy<TC, TI>::required_memory() {

    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();

    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t novx_b = aligned_length(m_ns.oacb * m_ns.vacb * naux);
    const size_t noox_b = aligned_length(m_ns.oacb * m_ns.oacb * naux);
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);

    if(m_scf_type == scf_type::USCF)
        return 4*novx_a + 4*novx_b + 2*noox_a + 2*noox_b + nrrx;
    return 3*novx_a + 2*noox_a + nrrx;

}

template<typename TC, typename TI>
size_t ricc2_energy<TC, TI>::required_buffer_size() {

    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();

    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t novx_b = aligned_length(m_ns.oacb * m_ns.vacb * naux);
    const size_t noox_b = aligned_length(m_ns.oacb * m_ns.oacb * naux);
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);

    const size_t nov_a = aligned_length(m_ns.oaca * m_ns.vaca);
    const size_t nor_a = aligned_length(m_ns.oaca * nbsf);
    const size_t norx_a = aligned_length(m_ns.oaca * nbsf * naux);
    const size_t nx = aligned_length(naux);

    if(m_scf_type == scf_type::USCF)
        return 4*novx_a + 4*novx_b + 2*noox_a + 2*noox_b + nrrx;
    return 3*novx_a + 2*noox_a + nrrx;
    // GPP: the correct memory allocation still needs to be determined depending on the 
    //      different integrals formed in ricc2.C - check the local variables created
    // return 8*novx_a + 2*noox_a + nrrx + 6*nov_a + 2*nor_a + 5*norx_a + nx;

}

template<typename TC, typename TI>
size_t ricc2_energy<TC, TI>::required_buffer_size_digestor() {

    const size_t naux = m_b3.get_ket().get_nbsf();
    const size_t nbsf = m_b3.get_bra().get_nbsfa();

    const size_t novx_a = aligned_length(m_ns.oaca * m_ns.vaca * naux);
    const size_t noox_a = aligned_length(m_ns.oaca * m_ns.oaca * naux);
    const size_t novx_b = aligned_length(m_ns.oacb * m_ns.vacb * naux);
    const size_t noox_b = aligned_length(m_ns.oacb * m_ns.oacb * naux);
    const size_t nrrx = aligned_length(nbsf * nbsf * naux);
    const size_t nvvx_a = aligned_length(m_ns.vaca * m_ns.vaca * naux);
    
    const size_t nov_a = aligned_length(m_ns.oaca * m_ns.vaca);
    const size_t nor_a = aligned_length(m_ns.oaca * nbsf);
    const size_t norx_a = aligned_length(m_ns.oaca * nbsf * naux);
    const size_t nx = aligned_length(naux);

    if(m_scf_type == scf_type::USCF)
        return 4*novx_a + 4*novx_b + 2*noox_a + 2*noox_b;
    return 4*novx_a + 2*noox_a;
    // GPP: the correct memory allocation still needs to be determined depending on the 
    //      different integrals formed in ricc2.C - check the local variables created
    // return 8*novx_a + 2*noox_a + 6*nov_a + 2*nor_a + 5*norx_a + nx;

}

template<typename TC, typename TI>
size_t ricc2_energy<TC, TI>::required_min_dynamic_memory() {

    op_coulomb op;
    std::vector<size_t> vblst(1);
    idx2_list blst(1, 1, 1, array_view<size_t>(&vblst[0], vblst.size()));
    blst.populate();
    scr_null<bat_2e3c_shellpair_cgto<TI>> scr;
    return motran_2e3c<TC, TI>(op, m_b3, scr, m_dev).min_memreq(blst);

}

template class ricc2_energy<double, double>;
template class ricc2_energy<complex<double>, double>;
template class ricc2_energy<complex<double>, complex<double>>;

} // namespace libgmbpt



