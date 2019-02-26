//
//  mixderiv.h
//  HelmDeriv
//
//  Created by Ian on 9/30/17.
//

#ifndef mixderiv_h
#define mixderiv_h

#include "isochoric/Helm.h"
#include "isochoric/complex_derivatives.h"
#include <memory>
#include <iostream>
#include <chrono>

inline bool Kronecker(std::size_t i, std::size_t j) { return i == j; }

// This class defines the interface that must be provided by the implementer.  This could be
// calculations coming from a cubic EOS, CoolProp's HEOS backend, REFPROP, etc.
//
// In general, the intention is that the class is immutable and does not do any calculations itself;
// all derivatives should be cached internally
template<typename TYPE = double>
class AbstractNativeDerivProvider {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
private:
    const EigenArray &m_rhovec;
public:
    AbstractNativeDerivProvider(const EigenArray &rhovec) : m_rhovec(rhovec) { };
    virtual ~AbstractNativeDerivProvider() {}
    const EigenArray &rhovec() const { 
        return m_rhovec; };
    virtual TYPE R() const = 0;
    virtual TYPE tau() const = 0;
    virtual TYPE delta() const = 0;
    virtual TYPE A(std::size_t i, std::size_t j) const = 0;
    virtual TYPE Ar(std::size_t i, std::size_t j) const = 0;
    virtual TYPE A0(std::size_t i, std::size_t j) const = 0;
    virtual TYPE Tr() const = 0;
    virtual TYPE dTr_drhoi(std::size_t i) const = 0;
    virtual TYPE dTr_dxi__constxj(std::size_t i) const = 0;
    virtual TYPE d2Tr_drhoidrhoj(std::size_t i, std::size_t j) const = 0;
    virtual TYPE d2Tr_dxidxj(std::size_t i, std::size_t j) const = 0;
    virtual TYPE rhor() const = 0;
    virtual TYPE drhor_drhoi(std::size_t i) const = 0;
    virtual TYPE drhor_dxi__constxj(std::size_t i) const = 0;
    virtual TYPE d2rhor_drhoidrhoj(std::size_t i, std::size_t j) const = 0;
    virtual TYPE d2rhor_dxidxj(std::size_t i, std::size_t j) const = 0;
    virtual TYPE dalpha_dxi__taudeltaxj(std::size_t i) const = 0;
    virtual TYPE d2alpha_dxi_ddelta__consttauxj(std::size_t i) const = 0;
    virtual TYPE d2alpha_dxi_dtau__constdeltaxj(std::size_t i) const = 0;
    virtual TYPE d2alpha_dxidxj__consttaudelta(std::size_t i, std::size_t j) const = 0;
    virtual TYPE dalphar_dxi__taudeltaxj(std::size_t i) const = 0;
    virtual TYPE d2alphar_dxi_ddelta__consttauxj(std::size_t i) const = 0;
    virtual TYPE d2alphar_dxi_dtau__constdeltaxj(std::size_t i) const = 0;
    virtual TYPE d2alphar_dxidxj__consttaudelta(std::size_t i, std::size_t j) const = 0;
};

template<typename TYPE = double>
class DerivativeMatrices {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
public:
    EigenArray dalpha_dxi__taudeltaxj, d2alpha_dxi_ddelta__consttauxj, d2alpha_dxi_dtau__constdeltaxj;
    EigenMatrix d2alpha_dxidxj__consttaudelta;
    EigenArray dalphar_dxi__taudeltaxj, d2alphar_dxi_ddelta__consttauxj, d2alphar_dxi_dtau__constdeltaxj;
    EigenMatrix d2alphar_dxidxj__consttaudelta;
    void resize(std::size_t N) {
        std::function<void(EigenArray &A)> init = [N](EigenArray &A) { A.resize(N); A.fill(0.0); };
        std::function<void(EigenMatrix &A)> initmat = [N](EigenMatrix &A) { A.resize(N, N); A.fill(0.0); };
        init(dalpha_dxi__taudeltaxj);
        init(d2alpha_dxi_ddelta__consttauxj);
        init(d2alpha_dxi_dtau__constdeltaxj);
        initmat(d2alpha_dxidxj__consttaudelta);
        init(dalphar_dxi__taudeltaxj);
        init(d2alphar_dxi_ddelta__consttauxj);
        init(d2alphar_dxi_dtau__constdeltaxj);
        initmat(d2alphar_dxidxj__consttaudelta);
    }
};

template<typename TYPE = double>
class MixDerivs {
public:
    template<typename T> T POW2(T x) const{ return x*x; }
    template<typename T> T POW4(T x) const { return POW2(x)*POW2(x); }
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    typedef std::function<std::unique_ptr<AbstractNativeDerivProvider<TYPE> >(TYPE T, const Eigen::Array<TYPE, Eigen::Dynamic, 1>&)> ProviderFactory;
public:
    std::unique_ptr<const AbstractNativeDerivProvider<TYPE> > p_ders;
    const AbstractNativeDerivProvider<TYPE> &m_ders;
    const EigenArray m_rhovec;
    const TYPE R;
    const TYPE delta, tau, rhor, Tr, T, rho, rhorTr;
    
    MixDerivs(std::unique_ptr<const AbstractNativeDerivProvider<TYPE> > &&ders)
    : p_ders(std::move(ders)),
      m_ders(*p_ders.get()),
      m_rhovec(m_ders.rhovec()), R(m_ders.R()), delta(m_ders.delta()), tau(m_ders.tau()), rhor(m_ders.rhor()), Tr(m_ders.Tr()), T(Tr / tau), rho(rhor*delta), rhorTr(rhor*Tr)
    {
        //m_ders = *p_ders;
    };
    MixDerivs(const MixDerivs &) = delete;
        ~MixDerivs() {
    }
    const AbstractNativeDerivProvider<TYPE> & get_native() const {
        return m_ders;
    }
    TYPE p() const {
        TYPE summer = rho*R*T - psir();
        for (auto i = 0; i < m_rhovec.size(); ++i) {
            summer += m_rhovec(i)*dpsir_drhoi__constTrhoj(i);
        }
        return summer;
    }
    TYPE dtau_dT() const {return -tau / T; }

    EigenMatrix get_Hessian() const{
        EigenMatrix H(2,2);
        const auto N = 2;
        // Construct Hessian matrices of Helmholtz energy density
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = i; j < N; ++j) {
                H(i, j) = d2psir_drhoidrhoj__constT(i, j) + ((i == j) ? R*T/m_rhovec[i] : 0.0);
                H(j, i) = H(i, j);
            }
        }
        return H;
    }
    void get_Hessian_and_eigs(EigenMatrix &H, EigenArray &eigs, EigenMatrix &U_T) const {
        // Get the Hessian matrix with concentrations as independent variables
        H = get_Hessian();
        // Solve for eigenvalues and eigenvectors
        Eigen::ComplexEigenSolver<Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic>> es(H);
        eigs = es.eigenvalues();
        U_T = es.eigenvectors(); // eigenvectors as columns
    }

    std::tuple<EigenArray,EigenArray,EigenArray> get_dnPsi_dsigma1n(const ProviderFactory &provider_factory, short Nderivs = 4, int i = 0, EigenArray v = EigenArray(0)) const
    {
        // Allocate the output array and fill with zero
        EigenArray out(Nderivs+1); out.fill(0.0);
        
        // If no eigenvector is provided, determine the eigenvector from the Hessian
        if (v.size() == 0){
            // Get the Hessian matrix with concentrations as independent variables, eigenvalues and corresponding eigenvectors (as columns)
            EigenMatrix H, U_T; EigenArray eigs;
            get_Hessian_and_eigs(H, eigs, U_T);
            //auto check = eigs[0] * U_T.col(0) - H*U_T.col(0);
            v = U_T.col(i);
        }
        
        // Evaluate the derivatives of \Psi^r numerically with complex Cauchy derivatives
        const std::complex<double> z0 = 0.0;
        const TYPE &r_T = T;
        const EigenArray &r_m_rhovec = m_rhovec;
        std::function<std::complex<double>(std::complex<double>)> f = [&provider_factory, &v, &r_T, &r_m_rhovec, i](const std::complex<double> sigmai) {
            // Get the transformed concentration values as a perturbation from the critical point
            EigenMatrix rhovec = r_m_rhovec + v*sigmai;
            std::unique_ptr<const MixDerivs<TYPE> > md(new MixDerivs<TYPE>(provider_factory(r_T, rhovec)));
            return md->psir();
        };
        const EigenArray psir_derivs = complex_derivative(f, Nderivs, z0, 0.75*std::abs(m_rhovec[0]), 21);

        // The contribution from the ideal-gas part
        EigenArray psi0_derivs(psir_derivs.size());
        psi0_derivs[2] = R*T*(v.array().pow(2.0) / m_rhovec).sum();
        psi0_derivs[3] = -R*T*((v.array().pow(3.0)) / (m_rhovec.array().pow(2.0))).sum();
        psi0_derivs[4] = 2.0*R*T*((v.array().pow(4.0)) / (m_rhovec.array().pow(3.0))).sum();
        
        // The total derivative is then equal to a contribution from ideal gas plus the contribution
        // from the residual part
        EigenArray total = psir_derivs+psi0_derivs;
        return std::make_tuple(total, psir_derivs, psi0_derivs);
    }

    TYPE d2tau_drhoidT(const std::size_t i) const { return -1/POW2(T)*m_ders.dTr_drhoi(i); }
    TYPE dpdT__constrhovec() const {
        TYPE dtau_dT = -tau/T;
        TYPE summer = rho*R - dpsir_dtau()*dtau_dT;
        for (auto i = 0; i < m_rhovec.size(); ++i) {
            summer += m_rhovec(i)*d2psir_dTdrhoi__constrhoj(i);
        }
        return summer;
    }
    TYPE dpdrhoi__constTrhoj(std::size_t i) const {
        TYPE summer = R*T;
        for (auto j = 0; j < m_rhovec.size(); ++j) {
            summer += m_rhovec(j)*d2psir_drhoidrhoj__constT(i, j);
        }
        return summer;
    }
    TYPE d2psi_dTdrhoi__constrhoj(std::size_t i) const {
        TYPE dtau_dT = -tau / T;
        return d_dpsi_dtau_drhoi__constrhoj(i)*dtau_dT + dpsi_dtau()*d2tau_drhoidT(i);
    }
    TYPE d2psir_dTdrhoi__constrhoj(std::size_t i) const {
        TYPE dtau_dT = -tau / T;
        return d_dpsir_dtau_drhoi__constrhoj(i)*dtau_dT + dpsir_dtau()*d2tau_drhoidT(i);
    }
    TYPE ddelta_drhoi__constTrhoj(std::size_t i) const {
        return (rhor - rho*m_ders.drhor_drhoi(i)) / (rhor*rhor);
    }
    TYPE dTr_drhoi__constrhoj(std::size_t i) const {
        return m_ders.dTr_drhoi(i);
    }
    TYPE drhor_drhoi__constrhoj(std::size_t i) const {
        return m_ders.drhor_drhoi(i);
    }
    TYPE d2Tr_drhoidrhoj(std::size_t i, std::size_t j) const {
        return m_ders.d2Tr_drhoidrhoj(i, j);
    }
    TYPE d2rhor_drhoidrhoj(std::size_t i, std::size_t j) const {
        return m_ders.d2rhor_drhoidrhoj(i,j);
    }
    TYPE d2delta_drhoidrhoj__constT(std::size_t i, std::size_t j) const {
        return (POW2(rhor)*(m_ders.drhor_drhoi(j) - m_ders.drhor_drhoi(i) - rho*m_ders.d2rhor_drhoidrhoj(i, j)) - (rhor - rho*m_ders.drhor_drhoi(i))*(2.0 * rhor*m_ders.drhor_drhoi(j))) / POW4(rhor);
    }
    TYPE dtau_drhoi__constTrhoj(std::size_t i) const {
        const TYPE one_over_T = tau / m_ders.Tr();
        return m_ders.dTr_drhoi(i)*one_over_T;
    }
    TYPE d2tau_drhoidrhoj__constT(std::size_t i, std::size_t j) const {
        const TYPE one_over_T = tau / m_ders.Tr();
        return m_ders.d2Tr_drhoidrhoj(i, j)*one_over_T;
    }
    TYPE dxj_drhoi__constTrhoj(std::size_t j, std::size_t i) const {
        return (rho*static_cast<TYPE>(Kronecker(i, j)) - m_rhovec(j)) / (rho*rho);
    }
    TYPE d2xk_drhoidrhoj__constT(std::size_t k, std::size_t i, std::size_t j) const {
        return (-rho*static_cast<TYPE>(Kronecker(i, k) + Kronecker(j, k)) + 2.0*m_rhovec(k))/(rho*rho*rho);
    }
    TYPE drhorTr_dxi__xj(std::size_t i) const {
        return m_ders.Tr()*m_ders.drhor_dxi__constxj(i) + m_ders.rhor()*m_ders.dTr_dxi__constxj(i);
    }
    TYPE d2rhorTr_dxidxj__consttaudelta(std::size_t i, std::size_t j) const {
        return (m_ders.Tr()*m_ders.d2rhor_dxidxj(i, j)
            + m_ders.dTr_dxi__constxj(j)*m_ders.drhor_dxi__constxj(i)
            + m_ders.rhor()*m_ders.d2Tr_dxidxj(i, j)
            + m_ders.drhor_dxi__constxj(j)*m_ders.dTr_dxi__constxj(i)
            );
    }
    TYPE dpsi_dxi__consttaudeltaxj(std::size_t i) const {
        return delta*R/tau*(A(0, 0)*drhorTr_dxi__xj(i) + rhorTr*m_ders.dalpha_dxi__taudeltaxj(i));
    }
    TYPE dpsir_dxi__consttaudeltaxj(std::size_t i) const {
        return delta*R/tau*(Ar(0, 0)*drhorTr_dxi__xj(i) + rhorTr*m_ders.dalphar_dxi__taudeltaxj(i));
    }
    TYPE d2psi_dxi_ddelta__consttauxj(std::size_t i) const {
        return R / tau*(drhorTr_dxi__xj(i)*(A(0, 1) + A(0, 0)) + rhorTr*(delta*m_ders.d2alpha_dxi_ddelta__consttauxj(i) + m_ders.dalpha_dxi__taudeltaxj(i)));
    }
    TYPE d2psir_dxi_ddelta__consttauxj(std::size_t i) const {
        return R / tau*(drhorTr_dxi__xj(i)*(Ar(0, 1) + Ar(0, 0)) + rhorTr*(delta*m_ders.d2alphar_dxi_ddelta__consttauxj(i) + m_ders.dalphar_dxi__taudeltaxj(i)));
    }
    TYPE d2psi_dxi_dtau__constdeltaxj(std::size_t i) const {
        return delta*R / POW2(tau)*(drhorTr_dxi__xj(i)*(A(1, 0) - A(0, 0)) + rhorTr*(tau*m_ders.d2alpha_dxi_dtau__constdeltaxj(i) - m_ders.dalpha_dxi__taudeltaxj(i)));
    }
    TYPE d2psir_dxi_dtau__constdeltaxj(std::size_t i) const {
        return delta*R / POW2(tau)*(drhorTr_dxi__xj(i)*(Ar(1, 0) - Ar(0, 0)) + rhorTr*(tau*m_ders.d2alphar_dxi_dtau__constdeltaxj(i) - m_ders.dalphar_dxi__taudeltaxj(i)));
    }
    TYPE d2psi_dxidxj__consttaudelta(std::size_t i, std::size_t j) const {
        return delta*R / tau*(A(0, 0)*d2rhorTr_dxidxj__consttaudelta(i, j)
            + m_ders.dalpha_dxi__taudeltaxj(i)*drhorTr_dxi__xj(j)
            + rhorTr*m_ders.d2alpha_dxidxj__consttaudelta(i, j)
            + m_ders.dalpha_dxi__taudeltaxj(j)*drhorTr_dxi__xj(i)
            );
    }
    TYPE d2psir_dxidxj__consttaudelta(std::size_t i, std::size_t j) const {
        return delta*R / tau*(Ar(0, 0)*d2rhorTr_dxidxj__consttaudelta(i, j)
            + m_ders.dalphar_dxi__taudeltaxj(i)*drhorTr_dxi__xj(j)
            + rhorTr*m_ders.d2alphar_dxidxj__consttaudelta(i, j)
            + m_ders.dalphar_dxi__taudeltaxj(j)*drhorTr_dxi__xj(i)
            );
    }
    /** Combined ideal-gas and residual contributions into A
    */
    inline TYPE A(std::size_t itau, std::size_t idelta) const {
        return m_ders.A(itau, idelta);
    };
    inline TYPE Ar(std::size_t itau, std::size_t idelta) const {
        return m_ders.Ar(itau, idelta);
    };
    TYPE psir() const {
        return rho*R*T*Ar(0, 0);
    };
    TYPE dpsi_ddelta() const {
        return (m_ders.rhor()*R*T)*(A(0, 1) + A(0, 0));
    };
    TYPE dpsir_ddelta() const {
        return (m_ders.rhor()*R*T)*(Ar(0, 1) + Ar(0, 0));
    };
    TYPE dpsi_dtau() const {
        return (m_ders.rhor()*delta*R*Tr / (tau*tau))*(A(1, 0) - A(0, 0));
    }
    TYPE dpsir_dtau() const {
        return (m_ders.rhor()*delta*R*Tr / (tau*tau))*(Ar(1, 0) - Ar(0, 0));
    }
    TYPE d2psi_ddelta2() const {
        return (m_ders.rhor()*R*T / delta)*(A(0, 2) + 2.0*A(0, 1));
    }
    TYPE d2psir_ddelta2() const {
        return (m_ders.rhor()*R*T / delta)*(Ar(0, 2) + 2.0*Ar(0, 1));
    }
    TYPE d2psi_ddelta_dtau() const {
        return (m_ders.rhor()*R*Tr / (tau*tau))*(A(1, 0) - A(0, 0) - A(0, 1) + A(1, 1));
    }
    TYPE d2psir_ddelta_dtau() const {
        return (m_ders.rhor()*R*Tr / (tau*tau))*(Ar(1, 0) - Ar(0, 0) - Ar(0, 1) + Ar(1, 1));
    }
    TYPE d2psi_dtau2() const {
        return (m_ders.rhor()*delta*R*Tr / (tau*tau*tau))*(A(2, 0) - 2.0*A(1, 0) + 2*A(0, 0));
    }
    TYPE d2psir_dtau2() const {
        return (m_ders.rhor()*delta*R*Tr / (tau*tau*tau))*(Ar(2, 0) - 2.0*Ar(1, 0) + 2.0*Ar(0, 0));
    }
    TYPE d_dpsi_ddelta_drhoi__constrhoj(std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += d2psi_dxi_ddelta__consttauxj(m)*dxj_drhoi__constTrhoj(m, i);
        }
        return d2psi_ddelta2()*ddelta_drhoi__constTrhoj(i) + d2psi_ddelta_dtau()*dtau_drhoi__constTrhoj(i) + summer;
    }
    TYPE d_dpsir_ddelta_drhoi__constrhoj(std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += d2psir_dxi_ddelta__consttauxj(m)*dxj_drhoi__constTrhoj(m, i);
        }
        return d2psir_ddelta2()*ddelta_drhoi__constTrhoj(i) + d2psir_ddelta_dtau()*dtau_drhoi__constTrhoj(i) + summer;
    }
    TYPE d_dpsi_dtau_drhoi__constrhoj(std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += d2psi_dxi_dtau__constdeltaxj(m)*dxj_drhoi__constTrhoj(m, i);
        }
        return d2psi_ddelta_dtau()*ddelta_drhoi__constTrhoj(i) + d2psi_dtau2()*dtau_drhoi__constTrhoj(i) + summer;
    }
    TYPE d_dpsir_dtau_drhoi__constrhoj(std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += d2psir_dxi_dtau__constdeltaxj(m)*dxj_drhoi__constTrhoj(m, i);
        }
        return d2psir_ddelta_dtau()*ddelta_drhoi__constTrhoj(i) + d2psir_dtau2()*dtau_drhoi__constTrhoj(i) + summer;
    }
    TYPE d_dpsi_dxm_drhoi__constTrhoi(std::size_t m, std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto n = 0; n < N; ++n) {
            summer += d2psi_dxidxj__consttaudelta(m, n)*dxj_drhoi__constTrhoj(n, i);
        }
        return (d2psi_dxi_ddelta__consttauxj(m)*ddelta_drhoi__constTrhoj(i)
            + d2psi_dxi_dtau__constdeltaxj(m)*dtau_drhoi__constTrhoj(i)
            + summer);
    }
    TYPE d_dpsir_dxm_drhoi__constTrhoi(std::size_t m, std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto n = 0; n < N; ++n) {
            summer += d2psir_dxidxj__consttaudelta(m, n)*dxj_drhoi__constTrhoj(n, i);
        }
        return (d2psir_dxi_ddelta__consttauxj(m)*ddelta_drhoi__constTrhoj(i)
            + d2psir_dxi_dtau__constdeltaxj(m)*dtau_drhoi__constTrhoj(i)
            + summer);
    }
    TYPE dpsi_drhoi__constTrhoj(std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += dpsi_dxi__consttaudeltaxj(m)*dxj_drhoi__constTrhoj(m, i);
        }
        return (dpsi_ddelta()*ddelta_drhoi__constTrhoj(i)
            + dpsi_dtau()*dtau_drhoi__constTrhoj(i)
            + summer);
    }
    TYPE dpsir_drhoi__constTrhoj(std::size_t i) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += dpsir_dxi__consttaudeltaxj(m)*dxj_drhoi__constTrhoj(m, i);
        }
        return (dpsir_ddelta()*ddelta_drhoi__constTrhoj(i)
            + dpsir_dtau()*dtau_drhoi__constTrhoj(i)
            + summer);
    }
    TYPE d2psi_drhoidrhoj__constT(std::size_t i, std::size_t j) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            summer += (dpsi_dxi__consttaudeltaxj(m)*d2xk_drhoidrhoj__constT(m, i, j)
                + d_dpsi_dxm_drhoi__constTrhoi(m, j)*dxj_drhoi__constTrhoj(m, i));
        }
        return (dpsi_ddelta()*d2delta_drhoidrhoj__constT(i, j)
            + d_dpsi_ddelta_drhoi__constrhoj(j)*ddelta_drhoi__constTrhoj(i)
            + dpsi_dtau()*d2tau_drhoidrhoj__constT(i, j)
            + d_dpsi_dtau_drhoi__constrhoj(j)*dtau_drhoi__constTrhoj(i)
            + summer);
    }
    TYPE d2psir_drhoidrhoj__constT(std::size_t i, std::size_t j) const {
        TYPE summer = 0;
        auto N = m_rhovec.size();
        for (auto m = 0; m < N; ++m) {
            //std::cout << "*new* " << m << " " << dpsir_dxi__consttaudeltaxj(m) << " " << d2xk_drhoidrhoj__constT(m, i, j) << " " << d_dpsir_dxm_drhoi__constTrhoi(m, j) << " " << dxj_drhoi__constTrhoj(m, i) << std::endl;
            summer += (dpsir_dxi__consttaudeltaxj(m)*d2xk_drhoidrhoj__constT(m, i, j)
                + d_dpsir_dxm_drhoi__constTrhoi(m, j)*dxj_drhoi__constTrhoj(m, i));
        }
        return (dpsir_ddelta()*d2delta_drhoidrhoj__constT(i, j)
            + d_dpsir_ddelta_drhoi__constrhoj(j)*ddelta_drhoi__constTrhoj(i)
            + dpsir_dtau()*d2tau_drhoidrhoj__constT(i, j)
            + d_dpsir_dtau_drhoi__constrhoj(j)*dtau_drhoi__constTrhoj(i)
            + summer);
    }
};

template<typename TYPE>
typename MixDerivs<TYPE>::EigenMatrix calc_drhovec_dT_crit(const typename MixDerivs<TYPE>::ProviderFactory &provider_factory,
                                                           const typename MixDerivs<TYPE>::ProviderFactory &Cauchy_provider_factory,
                                                           const TYPE T,
                                                           const typename MixDerivs<TYPE>::EigenArray &rhovec)
{
    // A couple of local typedefs to avoid verbose typenames (Eigen doesn't like auto)
    typedef typename MixDerivs<TYPE>::EigenMatrix EigenMatrix;
    typedef typename MixDerivs<TYPE>::EigenArray EigenArray;
    
    auto derivs_func = [&provider_factory,&Cauchy_provider_factory](const TYPE T, const EigenArray& rhovec, const EigenArray& v0 = EigenArray(0)) {
        auto derivs = std::make_unique<const MixDerivs<TYPE>>(provider_factory(T, rhovec));
        return std::get<0>(derivs->get_dnPsi_dsigma1n(Cauchy_provider_factory, 4,0,v0));
    };
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    auto md = std::make_unique<const MixDerivs<TYPE>>(provider_factory(T, rhovec));
    
    EigenMatrix H, U_T;
    EigenArray eigs;
    md->get_Hessian_and_eigs(H,eigs,U_T);
    
    auto base_derivs = md->get_dnPsi_dsigma1n(Cauchy_provider_factory, 4);
    EigenArray derivstot, derivsr, basederivs0;
    std::tie(derivstot, derivsr, basederivs0) = base_derivs;
//    std::cout << H - U_T.transpose()*eigs.matrix().asDiagonal()*U_T.transpose().inverse();
//    std::cout << "H" << H << std::endl;
//    std::cout << "U^T" <<  U_T << std::endl;
//    std::cout << "eigs" << eigs << std::endl;
//    std::cout << "psir" << derivsr << std::endl;
//    std::cout << "psitot" << derivstot << std::endl;
    
    auto dT = 1e-6;
    Eigen::ArrayXd Tderiv = ((derivs_func(T + dT, rhovec) - derivs_func(T - dT, rhovec)) / (2.0*dT)).eval().real();
    
    Eigen::ArrayXd v0 = U_T.real().col(0), v1 = U_T.real().col(1);
    double dsigmai = 1e-6*rhovec.real()[1];
    Eigen::ArrayXd drho0 = v0*dsigmai;
    Eigen::ArrayXd derivs0plus  = derivs_func(T, rhovec+drho0).real();
    Eigen::ArrayXd derivs0minus = derivs_func(T, rhovec-drho0).real();
    auto derivs0 = (derivs0plus-derivs0minus)/(2.0*dsigmai);
    Eigen::ArrayXd drho1 = v1*dsigmai;
    Eigen::ArrayXd derivs1plus  = derivs_func(T, rhovec+drho1).real();
    Eigen::ArrayXd derivs1minus = derivs_func(T, rhovec-drho1).real();
    auto derivs1 = (derivs1plus-derivs1minus)/(2.0*dsigmai);
    
    Eigen::MatrixXd b(2,2);
    b.col(0) << derivstot.real()(3), derivs1(2);
    b.col(1) << derivs0(3), derivs1(3);
    Eigen::MatrixXd RHS(2,1); RHS << -Tderiv[2], -Tderiv[3];
    //std::cout << "b" << b << std::endl;
//    std::cout << "RHS" << RHS << std::endl;
    
    Eigen::MatrixXd drhovec_dT = ((U_T.real()*b).transpose()).colPivHouseholderQr().solve(RHS);
    
//    std::cout << "(U_T*b).transpose()" << (U_T*b).transpose().real() << std::endl;
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elap_us = std::chrono::duration<double>(endTime - startTime).count()*1e6;
    //std::cout <<"elapsed [us]: " << elap_us << std::endl;
    
    return drhovec_dT;
}

#endif /* mixderiv_h */
