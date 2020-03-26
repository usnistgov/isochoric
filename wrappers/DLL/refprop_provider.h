#ifndef REFPROP_PROVIDER_H
#define REFPROP_PROVIDER_H

#define REFPROP_LIB_NAMESPACE REFPROP_lib
#include "REFPROP_lib.h"
#undef REFPROP_LIB_NAMESPACE

class REFPROPNativeDerivProvider : public AbstractNativeDerivProvider<> {
private:
    Eigen::ArrayXd z;
    GenHelmDerivDerivs<> m_vals, m_valsr, m_vals0;
    DerivativeMatrices<> m_mats;
    Eigen::ArrayXd m_dTr_drhoi, m_drhor_drhoi;
    Eigen::MatrixXd m_d2Tr_drhoidrhoj, m_d2rhor_drhoidrhoj;
    Eigen::ArrayXd m_dTr_dxi, m_drhor_dxi;
    Eigen::MatrixXd m_d2Tr_dxidxj, m_d2rhor_dxidxj;
    double m_Tr, m_rhor, m_R;
public:
    REFPROPNativeDerivProvider(const double T, const Eigen::ArrayXd &rhovec)
        : AbstractNativeDerivProvider<>(rhovec), z(rhovec / rhovec.sum())
    {
        using namespace REFPROP_lib;
        m_vals0.resize(3, 3); m_valsr.resize(3, 3);

        REDXdll(z.data(), m_Tr, m_rhor); m_rhor *= 1000;
        m_vals.tau = m_Tr / T;
        m_vals.delta = rhovec.sum() / m_rhor;
        double rho = m_vals.delta*m_rhor, rho_molL = rho / 1000;

        for (int itau = 0; itau < 3; ++itau) {
            for (int idelta = 0; idelta < 3; ++idelta) {
                double Arval, A0val;
                PHI0dll(itau, idelta, const_cast<double&>(T), rho_molL, z.data(), A0val);
                PHIXdll(itau, idelta, m_vals.tau, m_vals.delta, z.data(), Arval);
                m_valsr.setA(itau, idelta, Arval);
                m_vals0.setA(itau, idelta, A0val);
            }
        }
        auto N = z.size();
        m_mats.resize(N);

        Eigen::ArrayXd rhovec_molL = rhovec / 1000;
        double Tr_rho, rhor_rho; int ierr = 0; char herr[255] = "";

        /*
        subroutine RDXHMX (ix,icmp,icmp2,z,Tred,Dred,ierr,herr)
        c
        c  Returns reducing parameters and their derivatives associated with
        c    the mixture Helmholtz EOS; these are used to calculate the 'tau' and 'del'
        c    that are the independent variables in the EOS.
        c
        c  Input:
        c     icmp--Component number for which derivative will be calculated
        c    icmp2--Second component number for which derivative will be calculated
        c        z--Composition array (array of mole fractions)
        c       ix--Flag specifying the order of the composition derivative to calculate,
        c           when ix = 0, compute T(red) and D(red)
        c         for icmp2=0
        c           when ix = 1, compute 1st derivative with respect to z(icmp) or z(icmp2)
        c           when ix = 2, compute 2nd derivative with respect to z(icmp) or z(icmp2)
        c         for icmp<>0 and icmp2<>0
        c           when ix = 11, compute cross derivative with respect
        c                         to z(icmp) and z(icmp2)
        c
        c  Outputs:
        c     Tred--Reducing temperature [K] or derivative
        c     Dred--Reducing molar density [mol/L] or derivative of reducing volume
        c           (ix=0 - Dc; ix=1 - dVc/dxi; ix=2 - d^2Vc/dxi^2; ix=11 - d^2Vc/dxidxj)
        c     ierr--Error flag:  0 - Successful
        c                      301 - Mixing rule not found for i,j
        c                      191 - Derivative not available
        c     herr--Error string (character*255)
        */

        double drhor_dvr = -POW2(m_rhor);

        // First partials for each component
        int ix = 1, jj = 0;
        m_dTr_dxi.resize(N); m_drhor_dxi.resize(N);
        m_dTr_drhoi.resize(N); m_drhor_drhoi.resize(N);
        for (int i = 1; i <= N; ++i) {
            double dvr_dxi = -1;
            RDXHMXdll(ix, i, jj, const_cast<Eigen::ArrayXd&>(z).data(), m_dTr_dxi[i - 1], dvr_dxi, ierr, herr, 255);
            dvr_dxi /= 1000.0; // REFPROP returns dv/dxi, convert to density derivative and convert from mol/L to mol/m^3
            m_drhor_dxi(i - 1) = dvr_dxi*drhor_dvr;
        }

        // Second partials for each component
        m_d2Tr_dxidxj.resize(N, N); m_d2rhor_dxidxj.resize(N, N);
        m_d2Tr_drhoidrhoj.resize(N, N); m_d2rhor_drhoidrhoj.resize(N, N);
        for (int i = 1; i <= N; ++i) {
            double d2vrdxi2 = -1;
            ix = 2; RDXHMXdll(ix, i, jj, const_cast<Eigen::ArrayXd&>(z).data(), m_d2Tr_dxidxj(i - 1, i - 1), d2vrdxi2, ierr, herr, 255);
            d2vrdxi2 /= 1000.0; // From [L/mol] to [m^3/mol]
            double dvr_dxi = m_drhor_dxi(i - 1) / (-m_rhor*m_rhor); // [m^3/mol]
            m_d2rhor_dxidxj(i - 1, i - 1) = d2vrdxi2*drhor_dvr + 2 * POW3(m_rhor)*POW2(dvr_dxi);
        }
        // Cross partials
        for (int i = 1; i <= N; ++i) {
            for (int j = i + 1; j <= N; ++j) {
                double d2vrdxidxj = -1;
                ix = 11; RDXHMXdll(ix, i, j, const_cast<Eigen::ArrayXd&>(z).data(), m_d2Tr_dxidxj(i - 1, j - 1), d2vrdxidxj, ierr, herr, 255);
                d2vrdxidxj /= 1000.0; // From [L/mol] to [m^3/mol]
                double dvr_dxi = m_drhor_dxi(i - 1) / (-m_rhor*m_rhor); // [m^3/mol]
                double dvr_dxj = m_drhor_dxi(j - 1) / (-m_rhor*m_rhor); // [m^3/mol]
                m_d2rhor_dxidxj(i - 1, j - 1) = d2vrdxidxj*drhor_dvr + 2 * POW3(m_rhor)*dvr_dxi*dvr_dxj;
                m_d2Tr_dxidxj(j - 1, i - 1) = m_d2Tr_dxidxj(i - 1, j - 1);
                m_d2rhor_dxidxj(j - 1, i - 1) = m_d2rhor_dxidxj(i - 1, j - 1);
            }
        }

        ix = -100; int ii = 1; jj = 1;
        RDXHMXdll(ix, ii, jj, const_cast<Eigen::ArrayXd&>(rhovec_molL).data(), Tr_rho, rhor_rho, ierr, herr, 255);

        m_dTr_drhoi.resize(N); m_drhor_drhoi.resize(N);
        double dT_drhoival = -1, drhorrho_drhoival = -1, d2T_drhoi2val = -1, d2rhorrho2_drhoi2val = -1; int dummy = 0;

        for (int i = 1; i <= N; ++i) {
            ix = -101;
            RDXHMXdll(ix, i, dummy, const_cast<Eigen::ArrayXd&>(rhovec_molL).data(), dT_drhoival, drhorrho_drhoival, ierr, herr, 255);
            m_drhor_drhoi(i - 1) = 2 * rho_molL*rhor_rho + POW2(rho_molL)*drhorrho_drhoival;
            m_dTr_drhoi(i - 1) = (dT_drhoival / POW2(rho_molL) - 2 * Tr_rho / POW3(rho_molL)) / 1000.0; // division by 1000 is for conversion from [K/(mol/L)] to [K/(mol/m^3)]
            ix = -102;
            RDXHMXdll(ix, i, dummy, const_cast<Eigen::ArrayXd&>(rhovec_molL).data(), d2T_drhoi2val, d2rhorrho2_drhoi2val, ierr, herr, 255);
            m_d2Tr_drhoidrhoj(i - 1, i - 1) = ((rho_molL*d2T_drhoi2val - 4 * dT_drhoival) / POW3(rho_molL) + 6 * Tr_rho / POW4(rho_molL)) / 1e6; // division by 1e6 is for conversion from [K/(mol/L)]^2 to [K/(mol/m^3)]^2
            m_d2rhor_drhoidrhoj(i - 1, i - 1) = (4 * rho_molL*drhorrho_drhoival + 2 * rhor_rho + POW2(rho_molL)*d2rhorrho2_drhoi2val) / 1000; // division by 1000 is for conversion from [K/(mol/L)] to [K/(mol/m^3)]
            for (int j = i + 1; j <= N; ++j) {
                double dT_drhojval = -1, drhorrho_drhojval = -1, d2T_drhoidrhojval = -1, d2rhorrho2_drhoidrhojval = -1;
                ix = -101;
                RDXHMXdll(ix, j, dummy, const_cast<Eigen::ArrayXd&>(rhovec_molL).data(), dT_drhojval, drhorrho_drhojval, ierr, herr, 255);
                ix = -111;
                RDXHMXdll(ix, i, j, const_cast<Eigen::ArrayXd&>(rhovec_molL).data(), d2T_drhoidrhojval, d2rhorrho2_drhoidrhojval, ierr, herr, 255);
                m_d2Tr_drhoidrhoj(i - 1, j - 1) = ((rho_molL*d2T_drhoidrhojval - 2 * dT_drhoival - 2 * dT_drhojval) / POW3(rho_molL) + 6 * Tr_rho / POW4(rho_molL)) / 1e6; // division by 1e6 is for conversion from [K/(mol/L)]^2 to [K/(mol/m^3)]^2
                m_d2rhor_drhoidrhoj(i - 1, j - 1) = (2 * rho_molL*(drhorrho_drhoival + drhorrho_drhojval) + 2 * rhor_rho + POW2(rho_molL)*d2rhorrho2_drhoidrhojval) / 1000; // division by 1000 is for conversion from [K/(mol/L)] to [K/(mol/m^3)]
                m_d2Tr_drhoidrhoj(j - 1, i - 1) = m_d2Tr_drhoidrhoj(i - 1, j - 1);
                m_d2rhor_drhoidrhoj(j - 1, i - 1) = m_d2rhor_drhoidrhoj(i - 1, j - 1);
            }
        }
        {
            // Call PHIDERVdll and force derivatives to get cached
            int iderv = 2; double _T = T; int ierr = 0; char herr[255];
            Eigen::ArrayXd dadn(20), dnadn(20);
            PHIDERVdll(iderv, _T, rho_molL, z.data(), dadn.data(), dnadn.data(), ierr, herr, 255);
        }
        {
            double dbl; char hstr[255], herr[255]; int iset = 0 /*0:get,1:set*/, icomp = 1, jcomp = 1, ilng = 1, ierr = 0;
            Eigen::ArrayXd arr(100);
            {
                char hvr[255] = "dadxi";
                PASSCMNdll(hvr, iset, icomp, jcomp, hstr, ilng, dbl, arr.data(), ierr, herr, 255, 255, 255);
                m_mats.dalphar_dxi__taudeltaxj.head(N) = arr.head(N);
            }
            {
                char hvr[255] = "dadtx";
                PASSCMNdll(hvr, iset, icomp, jcomp, hstr, ilng, dbl, arr.data(), ierr, herr, 255, 255, 255);
                m_mats.d2alphar_dxi_dtau__constdeltaxj.head(N) = arr.head(N) / m_vals.tau;
            }
            {
                char hvr[255] = "daddx";
                PASSCMNdll(hvr, iset, icomp, jcomp, hstr, ilng, dbl, arr.data(), ierr, herr, 255, 255, 255);
                m_mats.d2alphar_dxi_ddelta__consttauxj.head(N) = arr.head(N) / m_vals.delta;
            }
            {
                char hvr[255] = "dadxij";
                for (int icomp = 1; icomp <= N; ++icomp) {
                    Eigen::ArrayXd buffer(30);
                    PASSCMNdll(hvr, iset, icomp, jcomp, hstr, ilng, dbl, buffer.data(), ierr, herr, 255, 255, 255);
                    m_mats.d2alphar_dxidxj__consttaudelta.row(icomp - 1).head(N) = buffer.head(N);
                }
            }
        }
        RMIX2dll(z.data(), m_R);
    };
    virtual double R() const override { return m_R; }
    virtual double tau() const override { return  m_vals.tau; };
    virtual double delta() const  override { return  m_vals.delta; };
    virtual double A(std::size_t itau, std::size_t idelta) const override { return m_vals0.A(itau, idelta) + m_valsr.A(itau, idelta); };
    virtual double Ar(std::size_t itau, std::size_t idelta) const override { return m_valsr.A(itau, idelta); };
    virtual double A0(std::size_t itau, std::size_t idelta) const override { return m_vals0.A(itau, idelta); };
    virtual double Tr() const override { return m_Tr; };
    virtual double dTr_drhoi(std::size_t i) const override { return m_dTr_drhoi(i); };
    virtual double d2Tr_drhoidrhoj(std::size_t i, std::size_t j) const  override { return m_d2Tr_drhoidrhoj(i, j); };
    virtual double dTr_dxi__constxj(std::size_t i) const override { return m_dTr_dxi(i); };
    virtual double d2Tr_dxidxj(std::size_t i, std::size_t j) const  override { return m_d2Tr_dxidxj(i, j); };
    virtual double rhor() const override { return m_rhor; };
    virtual double drhor_drhoi(std::size_t i) const override { return m_drhor_drhoi(i); };
    virtual double d2rhor_drhoidrhoj(std::size_t i, std::size_t j) const override { return m_d2rhor_drhoidrhoj(i, j); };
    virtual double drhor_dxi__constxj(std::size_t i) const override { return m_drhor_dxi(i); };
    virtual double d2rhor_dxidxj(std::size_t i, std::size_t j) const override { return m_d2rhor_dxidxj(i, j); };

    virtual double dalpha_dxi__taudeltaxj(std::size_t i) const override {
        return m_mats.dalpha_dxi__taudeltaxj(i);
    };
    virtual double d2alpha_dxi_ddelta__consttauxj(std::size_t i) const override {
        return m_mats.d2alpha_dxi_ddelta__consttauxj(i);
    };
    virtual double d2alpha_dxi_dtau__constdeltaxj(std::size_t i) const override { return m_mats.d2alpha_dxi_dtau__constdeltaxj(i); };
    virtual double d2alpha_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override { return m_mats.d2alpha_dxidxj__consttaudelta(i, j); };

    virtual double dalphar_dxi__taudeltaxj(std::size_t i) const override { return m_mats.dalphar_dxi__taudeltaxj(i); };
    virtual double d2alphar_dxi_ddelta__consttauxj(std::size_t i) const override { return m_mats.d2alphar_dxi_ddelta__consttauxj(i); };
    virtual double d2alphar_dxi_dtau__constdeltaxj(std::size_t i) const override { return m_mats.d2alphar_dxi_dtau__constdeltaxj(i); };
    virtual double d2alphar_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override { return m_mats.d2alphar_dxidxj__consttaudelta(i, j); };
};

#endif