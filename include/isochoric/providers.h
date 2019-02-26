#ifndef PROVIDERS_H
#define PROVIDERS_H

#include "mixderiv.h"

#include "AbstractState.h"
#include "Backends/Helmholtz/HelmholtzEOSMixtureBackend.h"
#include "Backends/Helmholtz/MixtureDerivatives.h"

using namespace CoolProp;
class CoolPropNativeDerivProvider : public AbstractNativeDerivProvider<double> {
private:
    CoolProp::HelmholtzEOSMixtureBackend & m_HEOS;
    Eigen::ArrayXd z;
    GenHelmDerivDerivs<double> m_vals, m_valsr, m_vals0;
    DerivativeMatrices<double> m_mats;
    Eigen::ArrayXd m_dTr_drhoi, m_drhor_drhoi;
    Eigen::MatrixXd m_d2Tr_drhoidrhoj, m_d2rhor_drhoidrhoj;
    Eigen::ArrayXd m_dTr_dxi, m_drhor_dxi;
    Eigen::MatrixXd m_d2Tr_dxidxj, m_d2rhor_dxidxj;
    double m_Tr, m_rhor, m_R;
public:
    CoolPropNativeDerivProvider(const CoolPropNativeDerivProvider&other) = delete;
    CoolPropNativeDerivProvider(HelmholtzEOSMixtureBackend &HEOS, double T, const Eigen::ArrayXd &rhovec)
        : AbstractNativeDerivProvider(rhovec), m_HEOS(HEOS), z(rhovec / rhovec.sum())
    {
        auto N = z.size(); std::vector<double> zz(z.data(), z.data() + N);
        m_vals0.resize(3, 3); m_valsr.resize(3, 3);

        m_mats.resize(N);

        m_dTr_dxi.resize(N); m_drhor_dxi.resize(N);
        m_dTr_drhoi.resize(N); m_drhor_drhoi.resize(N);
        m_d2Tr_dxidxj.resize(N, N); m_d2rhor_dxidxj.resize(N, N); 
        m_d2Tr_drhoidrhoj.resize(N, N); m_d2rhor_drhoidrhoj.resize(N, N);

        m_HEOS.set_mole_fractions(zz);
        double rhomolar = rhovec.sum();
        m_HEOS.specify_phase(iphase_gas);
        m_HEOS.update(DmolarT_INPUTS, rhomolar, T);
        m_HEOS.unspecify_phase();
        m_R = m_HEOS.gas_constant();

        m_vals.T = T;
        auto red = m_HEOS.get_reducing_state();
        m_rhor = red.rhomolar;
        m_Tr = red.T;
        m_vals.Tr = red.T;
        m_vals.tau = m_vals.Tr / T;
        m_vals.delta = rhovec.sum() / m_rhor;
        double rho = m_vals.delta*m_rhor;

        auto resid = HEOS.residual_helmholtz->all(m_HEOS, zz, m_vals.tau, m_vals.delta);
        for (long itau = 0; itau < 3; ++itau) {
            for (long idelta = 0; idelta < 3; ++idelta) {
                m_valsr.setA(itau, idelta, resid.get(itau, idelta)*pow(delta(), idelta)*pow(tau(), itau));
            }
        }
        if (m_HEOS.backend_name() == get_backend_string(HEOS_BACKEND_MIX)){
            auto GERG = static_cast<CoolProp::GERG2008ReducingFunction*>(m_HEOS.Reducing.get());

            // First partials for each component
            const std::vector<double> _rhovec(rhovec.data(), rhovec.data() + N);
            for (long i = 0; i < N; ++i) {
                m_dTr_dxi(i) = MixtureDerivatives::dTrdxi__constxj(m_HEOS, i, XN_INDEPENDENT);
                m_drhor_dxi(i) = MixtureDerivatives::drhormolardxi__constxj(m_HEOS, i, XN_INDEPENDENT);

                double rhor_rho = GERG->rhormolar(_rhovec);
                double Tr_rho = GERG->Tr(_rhovec);
                double drhorrho_drhoival = GERG->drhormolardxi__constxj(_rhovec, i, XN_INDEPENDENT);
                double dTrrho_drhoival = GERG->dTrdxi__constxj(_rhovec, i, XN_INDEPENDENT);

                m_drhor_drhoi(i) = 2 * rho*rhor_rho + POW2(rho)*drhorrho_drhoival;
                m_dTr_drhoi(i) = dTrrho_drhoival / POW2(rho) - 2 * Tr_rho / POW3(rho);
                for (long j = 0; j < N; ++j) {
                    double drhorrho_drhojval = GERG->drhormolardxi__constxj(_rhovec, j, XN_INDEPENDENT);
                    double d2rhorrho_drhoidrhojval = GERG->d2rhormolardxidxj(_rhovec, i, j, XN_INDEPENDENT);
                    double dTrrho_drhojval = GERG->dTrdxi__constxj(_rhovec, j, XN_INDEPENDENT);
                    double d2Trrho_drhoidrhojval = GERG->d2Trdxidxj(_rhovec, i, j, XN_INDEPENDENT);
                    m_d2rhor_drhoidrhoj(i, j) = 2 * (rho*drhorrho_drhoival + rhor_rho) + (rho*rho*d2rhorrho_drhoidrhojval + 2 * rho*drhorrho_drhojval);
                    m_d2Tr_drhoidrhoj(i, j) = (rho*d2Trrho_drhoidrhojval - 2 * dTrrho_drhoival - 2 * dTrrho_drhojval) / POW3(rho) + 6 * Tr_rho / POW4(rho);
                }
                // Second partials for each component
                for (long j = i; j < N; ++j) {
                    m_d2rhor_dxidxj(i, j) = MixtureDerivatives::d2rhormolardxidxj(m_HEOS, i, j, XN_INDEPENDENT);
                    m_d2Tr_dxidxj(i, j) = MixtureDerivatives::d2Trdxidxj(m_HEOS, i, j, XN_INDEPENDENT);
                    m_d2rhor_dxidxj(j, i) = m_d2rhor_dxidxj(i, j);
                    m_d2Tr_dxidxj(j, i) = m_d2Tr_dxidxj(i, j);
                }
            }
        }
        else {
            m_dTr_dxi.fill(0); m_drhor_dxi.fill(0);
            m_dTr_drhoi.fill(0); m_drhor_drhoi.fill(0);
            m_d2Tr_dxidxj.fill(0); m_d2rhor_dxidxj.fill(0);
            m_d2Tr_drhoidrhoj.fill(0); m_d2rhor_drhoidrhoj.fill(0);
        }
        
        for (long i = 0; i < N; ++i) {
            m_mats.dalphar_dxi__taudeltaxj(i) = MixtureDerivatives::dalphar_dxi(m_HEOS, i, XN_INDEPENDENT);
            m_mats.d2alphar_dxi_ddelta__consttauxj(i) = MixtureDerivatives::d2alphar_dxi_dDelta(m_HEOS, i, XN_INDEPENDENT);
            m_mats.d2alphar_dxi_dtau__constdeltaxj(i) = MixtureDerivatives::d2alphar_dxi_dTau(m_HEOS, i, XN_INDEPENDENT);
            for (long j = i; j < N; ++j) {
                m_mats.d2alphar_dxidxj__consttaudelta(i, j) = MixtureDerivatives::d2alphardxidxj(m_HEOS, i, j, XN_INDEPENDENT);
                m_mats.d2alphar_dxidxj__consttaudelta(j, i) = m_mats.d2alphar_dxidxj__consttaudelta(i, j);
            }
        }
    };
    virtual double R() const override{ return m_R; };
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

    virtual double dalpha_dxi__taudeltaxj(std::size_t i) const override { return m_mats.dalpha_dxi__taudeltaxj(i); };
    virtual double d2alpha_dxi_ddelta__consttauxj(std::size_t i) const override { return m_mats.d2alpha_dxi_ddelta__consttauxj(i); };
    virtual double d2alpha_dxi_dtau__constdeltaxj(std::size_t i) const override { return m_mats.d2alpha_dxi_dtau__constdeltaxj(i); };
    virtual double d2alpha_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override { return m_mats.d2alpha_dxidxj__consttaudelta(i, j); };

    virtual double dalphar_dxi__taudeltaxj(std::size_t i) const override { return m_mats.dalphar_dxi__taudeltaxj(i); };
    virtual double d2alphar_dxi_ddelta__consttauxj(std::size_t i) const override { return m_mats.d2alphar_dxi_ddelta__consttauxj(i); };
    virtual double d2alphar_dxi_dtau__constdeltaxj(std::size_t i) const override { return m_mats.d2alphar_dxi_dtau__constdeltaxj(i); };
    virtual double d2alphar_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override { return m_mats.d2alphar_dxidxj__consttaudelta(i, j); };
};

#endif
