#ifndef helm_h
#define helm_h

#include "Eigen/Dense"
#include <cmath>
#include "isochoric/Helm.h"

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

const double one_over_M_LN2 = 1.0/M_LN2;

template<typename TYPE = double>
struct GenHelmDerivCoeffs{
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    EigenArray n,t,d,l,c,cd,ld,lt,ct,beta,epsilon,eta,gamma;
    Eigen::Array<int, Eigen::Dynamic, 1> ld_as_int;
};

template<typename TYPE = double>
class GenHelmDerivDerivs {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
private:
    EigenMatrix m_A;
public:
    GenHelmDerivDerivs(){
    };
    GenHelmDerivDerivs(const GenHelmDerivDerivs& that) = delete;
    ~GenHelmDerivDerivs() {
    }
    TYPE tau, delta, T, Tr;
    void resize(std::size_t itau, std::size_t idelta) {
        m_A.resize(itau, idelta);
    }
    void setA(std::size_t itau, std::size_t idelta, TYPE val) {
        m_A(itau, idelta) = val;
    }
    TYPE A(std::size_t itau, std::size_t idelta) const {
        return m_A(itau, idelta);
    }
};

// From https://stackoverflow.com/a/5625446/1360263
template <typename T>
T powi(T p, unsigned q)
{
    T r(1);
    
    while (q != 0) {
        if (q % 2 == 1) {    // q is odd
            r *= p;
            q--;
        }
        p *= p;
        q /= 2;
    }
    
    return r;
}

template<typename TYPE>
class GenHelmDeriv{

    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;

private:
    bool use_tau_lt = false, use_gauss_tau = true;
    bool use_delta_ld = true, use_gauss_delta = true;
    bool some_u;
    
    GenHelmDerivCoeffs<TYPE> coeffs;
    // Work arrays
    EigenArray tau_du_dtau, B1delta, delta_dB1delta_ddelta, B1tau, B2delta, B3delta, armat, umat, delta_minus_epsilon, tau_minus_gamma, delta_to_ld, tau_to_lt, delta_du_ddelta, delta2_d2u_ddelta2, delta3_d3u_ddelta3, delta2_d2B1delta_ddelta2, tau2_d2u_dtau2, tau3_d3u_dtau3, tau_dB1tau_dtau, B2tau, tau2_d2B1tau_dtau2, B3tau;
    
    // Boolean mask arrays
    Eigen::Array<bool,Eigen::Dynamic,1> mask_gauss_delta,mask_delta_ld,mask_gauss_tau,mask_tau_lt;
    
    void init(){
        some_u = use_tau_lt || use_gauss_tau || use_delta_ld || use_gauss_delta;
        mask_gauss_delta = Eigen::Array<bool,Eigen::Dynamic,1>::Constant(coeffs.n.size(), use_gauss_delta);
        mask_delta_ld = Eigen::Array<bool,Eigen::Dynamic,1>::Constant(coeffs.n.size(), use_delta_ld);
        mask_gauss_tau = Eigen::Array<bool,Eigen::Dynamic,1>::Constant(coeffs.n.size(), use_gauss_tau);
        mask_tau_lt = Eigen::Array<bool,Eigen::Dynamic,1>::Constant(coeffs.n.size(), use_tau_lt);
        delta_to_ld = Eigen::ArrayXd::Constant(coeffs.n.size(), 0);
        tau_to_lt = Eigen::ArrayXd::Constant(coeffs.n.size(), 0);
    };
    
public:
    bool all_ld_integers = true;
    GenHelmDeriv(GenHelmDerivCoeffs<TYPE> &&coeffs) : coeffs(coeffs){ init(); };
    GenHelmDeriv(const GenHelmDerivCoeffs<TYPE> &coeffs) : coeffs(coeffs){ init(); };
    
    void calc(const TYPE tau, const TYPE delta, GenHelmDerivDerivs<TYPE> &derivs, const std::size_t Ntau_max = 3, const std::size_t Ndelta_max = 3)
    {
        derivs.tau = tau; derivs.delta = delta;
        const TYPE log2tau = log2(tau), log2delta = log2(delta);
        derivs.resize(Ntau_max+1, Ndelta_max+1);
        
        // Gaussian difference vectors
        if (use_gauss_delta) { delta_minus_epsilon = delta - coeffs.epsilon; }
        if (use_gauss_tau){ tau_minus_gamma = tau - coeffs.gamma; }
        
        if (use_delta_ld) {
            if (false && all_ld_integers){
                TYPE _delta = delta;
                //delta_to_ld = coeffs.ld_as_int.unaryExpr( [_delta](int _i){return powi(_delta, _i);} ).cast<TYPE>();
            }
            else{
                delta_to_ld = Eigen::pow(delta, coeffs.ld);
            }
        }
        
        if (use_tau_lt) { tau_to_lt = Eigen::pow(tau, coeffs.lt); }
        
        if (some_u){
            umat = (
                mask_gauss_delta.select(-coeffs.eta*delta_minus_epsilon.square(),0)
                +
                mask_delta_ld.select(-coeffs.cd*delta_to_ld,0)
                +
                mask_gauss_tau.select(-coeffs.beta*tau_minus_gamma.square(),0)
                +
                mask_tau_lt.select(-coeffs.ct*tau_to_lt,0)
            );
        }
        else{
            umat = 0*coeffs.n;
        }
        
        // *******************************************************************
        // *******************************************************************
        //                               TAU TERMS
        // *******************************************************************
        // *******************************************************************
        
        if (Ntau_max >= 1){
            if(use_gauss_tau && use_tau_lt){ tau_du_dtau = -2*tau*coeffs.beta*tau_minus_gamma -coeffs.ct*coeffs.lt*tau_to_lt; }
            else if(use_tau_lt){ tau_du_dtau = -coeffs.ct*coeffs.lt*tau_to_lt; }
            else if(use_gauss_tau){ tau_du_dtau = -2*tau*coeffs.beta*tau_minus_gamma; }
            else{
                tau_du_dtau = 0*coeffs.n;
            }
            B1tau = tau_du_dtau + coeffs.t;
        }
        if (Ntau_max >= 2){
            if (use_gauss_tau && use_tau_lt){ tau2_d2u_dtau2 = -2*(tau*tau)*coeffs.beta -coeffs.ct*coeffs.lt*(coeffs.lt - 1)*tau_to_lt; }
            else if(use_tau_lt){ tau2_d2u_dtau2 = -coeffs.ct*coeffs.lt*(coeffs.lt - 1)*tau_to_lt; }
            else if(use_gauss_tau){ tau2_d2u_dtau2 = -2*(tau*tau)*coeffs.beta; }
            else{
                tau2_d2u_dtau2 = 0*coeffs.n;
            }
            tau_dB1tau_dtau = tau2_d2u_dtau2 + tau_du_dtau;
            B2tau = tau_dB1tau_dtau + B1tau.square() - B1tau;
        }
        if (Ntau_max >= 3){
            if (use_tau_lt){
                tau3_d3u_dtau3 = -coeffs.ct*coeffs.lt*(coeffs.lt - 1)*(coeffs.lt- 2)*tau_to_lt;
            }
            else{
                tau3_d3u_dtau3 = 0*coeffs.n;
            }
            tau2_d2B1tau_dtau2 = tau3_d3u_dtau3 + 2*tau2_d2u_dtau2;
            B3tau = tau2_d2B1tau_dtau2 + 3*B1tau*tau_dB1tau_dtau - 2*tau_dB1tau_dtau + B1tau.cube() - 3*B1tau.square() + 2*B1tau;
        }
        
        // *******************************************************************
        // *******************************************************************
        //                               DELTA TERMS
        // *******************************************************************
        // *******************************************************************
        
        if (Ndelta_max >= 1){
            if (use_gauss_delta && use_delta_ld){ delta_du_ddelta = -2*delta*coeffs.eta*delta_minus_epsilon -coeffs.cd*coeffs.ld*delta_to_ld; }
            else if (use_delta_ld){ delta_du_ddelta = -coeffs.cd*coeffs.ld*delta_to_ld; }
            else if (use_gauss_delta) { delta_du_ddelta = -2*delta*coeffs.eta*delta_minus_epsilon; }
            else{
                delta_du_ddelta = 0*coeffs.n;
            }
            B1delta = delta_du_ddelta + coeffs.d;
        }
        if (Ndelta_max >= 2){
            if (use_gauss_delta && use_delta_ld){ delta2_d2u_ddelta2 = -2*(delta*delta)*coeffs.eta -coeffs.cd*coeffs.ld*(coeffs.ld-1)*delta_to_ld; }
            else if (use_delta_ld) { delta2_d2u_ddelta2 = -coeffs.cd*coeffs.ld*(coeffs.ld-1)*delta_to_ld; }
            else if (use_gauss_delta) { delta2_d2u_ddelta2 = -2*(delta*delta)*coeffs.eta; }
            else{
                delta2_d2u_ddelta2 = 0*coeffs.n;
            }
            delta_dB1delta_ddelta = delta2_d2u_ddelta2 + delta_du_ddelta;
            B2delta = delta_dB1delta_ddelta + B1delta.square() - B1delta;
        }
        if (Ndelta_max >= 3){
            if (use_delta_ld){
                delta3_d3u_ddelta3 = -coeffs.cd*coeffs.ld*(coeffs.ld-1)*(coeffs.ld-2)*delta_to_ld;
            }
            else{
                delta3_d3u_ddelta3 = 0*coeffs.n;
            }
            delta2_d2B1delta_ddelta2 = delta3_d3u_ddelta3 + 2*delta2_d2u_ddelta2;
            B3delta = delta2_d2B1delta_ddelta2 + 3*B1delta*delta_dB1delta_ddelta - 2*delta_dB1delta_ddelta + B1delta.cube() - 3*B1delta.square() + 2*B1delta;
        }
        
        // Finally, carry out the calculations
        
        armat = coeffs.n*Eigen::pow(2, (coeffs.t*log2tau + coeffs.d*log2delta + umat*one_over_M_LN2).array());
        derivs.setA(0,0, armat.sum());
        derivs.setA(1,0, (Ntau_max >=1) ? (armat*B1tau).sum() : 0);
        derivs.setA(0,1, (Ndelta_max >= 1) ? (armat*B1delta).sum() : 0);
        derivs.setA(2,0, (Ntau_max >=2) ? (armat*B2tau).sum() : 0);
        derivs.setA(1,1, (Ntau_max >= 1 && Ndelta_max >= 1) ? (armat*B1tau*B1delta).sum() : 0);
        derivs.setA(0,2, (Ndelta_max >=2) ? (armat*B2delta).sum() : 0);
        derivs.setA(3,0, (Ntau_max >=3) ? (armat*B3tau).sum() : 0);
        derivs.setA(2,1, (Ntau_max >= 2 && Ndelta_max >=1) ? (armat*B2tau*B1delta).sum() : 0);
        derivs.setA(1,2, (Ntau_max >= 1 && Ndelta_max >=2) ? (armat*B1tau*B2delta).sum() : 0);
        derivs.setA(0,3, (Ndelta_max >=3) ? (armat*B3delta).sum() : 0);
    };
};


#endif
