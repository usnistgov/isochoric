#ifndef CUBIC_PROVIDER_H
#define CUBIC_PROVIDER_H

#include "isochoric/generalizedcubic.h"
#include "isochoric/mixderiv.h"

//
template<typename TYPE>
class CubicNativeDerivProvider : public AbstractNativeDerivProvider<TYPE> {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
private:
    EigenArray z;
    GenHelmDerivDerivs<TYPE> m_vals, m_valsr, m_vals0;
    DerivativeMatrices<TYPE> m_mats;
    TYPE m_Tr, m_rhor, m_R;
public:
    CubicNativeDerivProvider(
     const std::unique_ptr<const AbstractCubic<TYPE>> &cubic, const TYPE T, const EigenArray &rhovec, const bool only_psir = false
    ) : AbstractNativeDerivProvider<TYPE>(rhovec), z(rhovec / rhovec.sum()) {

        auto N = z.size(); std::vector<TYPE> zz(z.data(), z.data() + N);
        m_vals0.resize(3, 3); m_valsr.resize(3, 3); m_mats.resize(N); 

        m_vals.T = T;
        m_rhor = cubic->rho_r;
        m_Tr = cubic->T_r;
        m_vals.Tr = m_Tr;
        m_vals.tau = m_vals.Tr / T;
        m_vals.delta = rhovec.sum()/m_rhor;
        m_R = 8.3144598;

        for (auto itau = 0; itau < 3; ++itau) {
            for (auto idelta = 0; idelta < 3; ++idelta) {
                auto Axy = cubic->alphar(m_vals.tau, m_vals.delta, zz, itau, idelta)*pow(m_vals.delta, idelta)*pow(m_vals.tau, itau);
                m_valsr.setA(itau, idelta, Axy);
                if (only_psir){ return; }
            }
        }
        bool XN_INDEPENDENT = true;
        for (auto i = 0; i < N; ++i) {
            TYPE tau = m_vals.tau, delta = m_vals.delta;
            m_mats.dalphar_dxi__taudeltaxj(i) = cubic->d_alphar_dxi(tau,delta,zz,0,0,i, XN_INDEPENDENT); // 0,0 are itau, idelta
            m_mats.d2alphar_dxi_ddelta__consttauxj(i) = cubic->d_alphar_dxi(tau, delta, zz, 0, 1, i, XN_INDEPENDENT); // 0,1 are itau,idelta
            m_mats.d2alphar_dxi_dtau__constdeltaxj(i) = cubic->d_alphar_dxi(tau, delta, zz, 1, 0, i, XN_INDEPENDENT); // 1,0 are itau,idelta
            for (auto j = i; j < N; ++j) {
                m_mats.d2alphar_dxidxj__consttaudelta(i, j) = cubic->d2_alphar_dxidxj(tau, delta, zz, 0, 0, i, j, XN_INDEPENDENT);
                m_mats.d2alphar_dxidxj__consttaudelta(j, i) = m_mats.d2alphar_dxidxj__consttaudelta(i, j);
            }
        }
    };
    //virtual ~CubicNativeDerivProvider(){};
    virtual TYPE R() const override { return m_R; };
    virtual TYPE tau() const override { return m_vals.tau; };
    virtual TYPE delta() const  override { return m_vals.delta; };
    virtual TYPE A(std::size_t itau, std::size_t idelta) const override { throw std::exception(); return Ar(itau,idelta) + A0(itau,idelta); };
    virtual TYPE Ar(std::size_t itau, std::size_t idelta) const override { 
        return m_valsr.A(itau, idelta); };
    virtual TYPE A0(std::size_t itau, std::size_t idelta) const override { return m_vals0.A(itau, idelta); };
    virtual TYPE Tr() const override { return m_Tr; };
    virtual TYPE dTr_drhoi(std::size_t i) const override { return 0.0; };
    virtual TYPE dTr_dxi__constxj(std::size_t i) const override { return 0.0; };
    virtual TYPE d2Tr_drhoidrhoj(std::size_t i, std::size_t j) const override { return 0.0; };
    virtual TYPE d2Tr_dxidxj(std::size_t i, std::size_t j) const override { return 0.0; };
    virtual TYPE rhor() const override { return m_rhor; };
    virtual TYPE drhor_drhoi(std::size_t i) const override { return 0.0; };
    virtual TYPE drhor_dxi__constxj(std::size_t i) const override { return 0.0; };
    virtual TYPE d2rhor_drhoidrhoj(std::size_t i, std::size_t j) const override { return 0.0; };
    virtual TYPE d2rhor_dxidxj(std::size_t i, std::size_t j) const override { return 0.0; };
    virtual TYPE dalpha_dxi__taudeltaxj(std::size_t i) const override { throw std::exception(); return m_mats.dalpha_dxi__taudeltaxj(i); };
    virtual TYPE d2alpha_dxi_ddelta__consttauxj(std::size_t i) const override { throw std::exception(); return m_mats.d2alpha_dxi_ddelta__consttauxj(i); };
    virtual TYPE d2alpha_dxi_dtau__constdeltaxj(std::size_t i) const override { throw std::exception(); return m_mats.d2alpha_dxi_dtau__constdeltaxj(i); };
    virtual TYPE d2alpha_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override { throw std::exception(); return m_mats.d2alpha_dxidxj__consttaudelta(i, j); };
    virtual TYPE dalphar_dxi__taudeltaxj(std::size_t i) const override { return m_mats.dalphar_dxi__taudeltaxj(i); };
    virtual TYPE d2alphar_dxi_ddelta__consttauxj(std::size_t i) const override { return m_mats.d2alphar_dxi_ddelta__consttauxj(i); };
    virtual TYPE d2alphar_dxi_dtau__constdeltaxj(std::size_t i) const override { return m_mats.d2alphar_dxi_dtau__constdeltaxj(i); };
    virtual TYPE d2alphar_dxidxj__consttaudelta(std::size_t i, std::size_t j) const override { return m_mats.d2alphar_dxidxj__consttaudelta(i, j); };
};


#endif
