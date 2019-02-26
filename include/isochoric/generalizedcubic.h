/** 
    This C++ code is the implementation of the analyses presented in the paper
    I.Bell and A. Jäger, "Helmholtz energy translations for common cubic equations of state 
    for use in one-fluid and multi-fluid mixture models", J. Res. NIST, 2016

    This code is in the public domain, though if used in academic work, we would appreciate
    a reference back to the paper given above.

 */

#ifndef GENERALIZED_CUBIC_H
#define GENERALIZED_CUBIC_H

#include <vector>
#include <cmath>

template<typename TYPE = double>
class AbstractCubic
{
protected:
    std::vector<TYPE> Tc, ///< Vector of critical temperatures (in K)
                      pc, ///< Vector of critical pressures (in Pa) 
                      acentric; ///< Vector of acentric factors (unitless)
    TYPE R_u; ///< The universal gas constant  in J/(mol*K)
    TYPE Delta_1, ///< The first cubic constant
           Delta_2; ///< The second cubic constant
    int N; ///< Number of components in the mixture
    bool simple_aii; ///< True if the Mathias-Copeman equation for a_ii is not being used
    std::vector<TYPE> C1,C2,C3;///< The Mathias-Copeman coefficients for a_ii
    std::vector< std::vector<TYPE> > k;///< The interaction parameters (k_ii = 0)
public:
    const TYPE rho_r = 1.0, T_r = 1.0;
    
    /**
     \brief The abstract base clase for the concrete implementations of the cubic equations of state

     This abstract base class describes the structure that must be implemented by concrete implementations
     of the cubic equations of state (SRK, PR, etc.).  The virtual functions must be implemented by the 
     derived classes, the remaining functions are generic and are not dependent on the equation of state,
     so long as it has the formulation given in this work.

     */
    AbstractCubic(std::vector<TYPE> Tc, 
                  std::vector<TYPE> pc, 
                  std::vector<TYPE> acentric,
                  TYPE R_u,
                  TYPE Delta_1,
                  TYPE Delta_2,
                  const std::vector<TYPE> &C1 = std::vector<TYPE>(),
                  const std::vector<TYPE> &C2 = std::vector<TYPE>(),
                  const std::vector<TYPE> &C3 = std::vector<TYPE>(),
                  const std::vector<std::vector<TYPE>> &k = std::vector<std::vector<TYPE>>()
                 ) 
        : Tc(Tc), pc(pc), acentric(acentric), R_u(R_u), Delta_1(Delta_1), Delta_2(Delta_2), C1(C1), C2(C2), C3(C3), k(k)
        {
            N = static_cast<int>(Tc.size());
            if (this->k.empty()){
                this->k.resize(N, std::vector<TYPE>(N, 0));
            }
            /// If no Mathias-Copeman coefficients are passed in (all empty vectors), use the predictive scheme for m_ii
            simple_aii = (C1.empty() && C2.empty() && C3.empty());
        };
    
    virtual ~AbstractCubic() {};
    
    /// Get the vector of critical temperatures (in K)
    const std::vector<TYPE> &get_Tc() const { return Tc; }
    /// Get the vector of critical pressures (in Pa)
    const std::vector<TYPE> &get_pc() const { return pc; }
    /// Get the vector of acentric factors
    const std::vector<TYPE> &get_acentric() const { return acentric; }
    /// Read-only accessor for value of Delta_1
    const TYPE get_Delta_1(){ return Delta_1; }
    /// Read-only accessor for value of Delta_2
    const TYPE get_Delta_2(){ return Delta_2; }
    /// Read-only accessor for value of R_u (universal gas constant)
    const TYPE get_R_u(){ return R_u; }
    /// Get a constant reference to a constant vector of Mathias-Copeman constants
    const std::vector<TYPE> &get_C_ref(int i) const{
        switch (i){
        case 1: return C1;
        case 2: return C2;
        case 3: return C3;
        default:
            throw -1;
        }
    }
    /// Set the value of kij parameter for given binary pair
    void set_kij(std::size_t i, std::size_t j, TYPE kij){
        k[i][j] = kij;
        k[j][i] = kij; // symmetric
    }

    /// Get the leading constant in the expression for the pure fluid attractive energy term 
    /// (must be implemented by derived classes)
    virtual TYPE a0_ii(std::size_t i) const = 0;
    /// Get the leading constant in the expression for the pure fluid covolume term 
    /// (must be implemented by derived classes)
    virtual TYPE b0_ii(std::size_t i) const = 0;
    /// Get the m_ii variable in the alpha term inculuded in the attractive part
    virtual TYPE m_ii(std::size_t i) const = 0;

    /// The residual non-dimensionalized Helmholtz energy \f$\alpha^r\f$
    TYPE alphar(TYPE tau, TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta) const
    {
        return psi_minus(delta, x, itau, idelta)-tau_times_a(tau,x,itau)/(R_u*T_r)*psi_plus(delta,x,idelta);
    }

    /// The first composition derivative of \f$\alpha^r\f$ as well as derivatives with respect to \f$\tau\f$ and \f$\delta\f$
    TYPE d_alphar_dxi(TYPE tau, TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta, std::size_t i, bool xN_independent) const
    {
        return (d_psi_minus_dxi(delta, x, itau, idelta, i, xN_independent)
                -1.0/(R_u*T_r)*(d_tau_times_a_dxi(tau,x,itau,i,xN_independent)*psi_plus(delta,x,idelta)
                            +tau_times_a(tau,x,itau)*d_psi_plus_dxi(delta,x,idelta,i,xN_independent)
                            )
                );
    }
    /// The second composition derivative of \f$\alpha^r\f$ as well as derivatives with respect to \f$\tau\f$ and \f$\delta\f$
    TYPE d2_alphar_dxidxj(TYPE tau, TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta, std::size_t i, std::size_t j, bool xN_independent) const
    {
        return (d2_psi_minus_dxidxj(delta, x, itau, idelta, i, j, xN_independent)
                -1.0/(R_u*T_r)*(d2_tau_times_a_dxidxj(tau,x,itau,i,j,xN_independent)*psi_plus(delta,x,idelta)
                            +d_tau_times_a_dxi(tau,x,itau,i,xN_independent)*d_psi_plus_dxi(delta,x,idelta,j,xN_independent)
                            +d_tau_times_a_dxi(tau,x,itau,j,xN_independent)*d_psi_plus_dxi(delta,x,idelta,i,xN_independent)
                            +tau_times_a(tau,x,itau)*d2_psi_plus_dxidxj(delta,x,idelta,i,j,xN_independent)
                            )
                );
    }

    
    /// The third composition derivative of \f$\alpha^r\f$ as well as derivatives with respect to \f$\tau\f$ and \f$\delta\f$
    TYPE d3_alphar_dxidxjdxk(TYPE tau, TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        return (d3_psi_minus_dxidxjdxk(delta, x, itau, idelta, i, j, k, xN_independent)
                -1.0/(R_u*T_r)*(d2_tau_times_a_dxidxj(tau,x,itau,i,j,xN_independent)*d_psi_plus_dxi(delta,x,idelta,k,xN_independent)
                            +d3_tau_times_a_dxidxjdxk(tau,x,itau,i,j,k,xN_independent)*psi_plus(delta,x,idelta)
                            
                            +d_tau_times_a_dxi(tau,x,itau,i,xN_independent)*d2_psi_plus_dxidxj(delta,x,idelta,j,k,xN_independent)
                            +d2_tau_times_a_dxidxj(tau,x,itau,i,k,xN_independent)*d_psi_plus_dxi(delta,x,idelta,j,xN_independent)
                
                            +d_tau_times_a_dxi(tau,x,itau,j,xN_independent)*d2_psi_plus_dxidxj(delta,x,idelta,i,k,xN_independent)
                            +d2_tau_times_a_dxidxj(tau,x,itau,j,k,xN_independent)*d_psi_plus_dxi(delta,x,idelta,i,xN_independent)
                
                            +tau_times_a(tau,x,itau)*d3_psi_plus_dxidxjdxk(delta,x,idelta,i,j,k,xN_independent)
                            +d_tau_times_a_dxi(tau,x,itau,k, xN_independent)*d2_psi_plus_dxidxj(delta,x,idelta,i,j,xN_independent)
                            )
                );
    }
    
    /** 
     * \brief The n-th derivative of \f$a_m\f$ with respect to \f$\tau\f$
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param x The vector of mole fractions
     * \param itau The number of derivatives of \f$a_m\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_m, itau=1 is d(a_m)/d(tau), etc.)
     */
    virtual TYPE am_term(TYPE tau, const std::vector<TYPE> &x, std::size_t itau) const
    {
        TYPE summer = 0;
        for(int i = N-1; i >= 0; --i)
        {
            for (int j = N-1; j >= 0; --j)
            {
                summer += x[i]*x[j]*aij_term(tau, i, j, itau);
            }
        }
        return summer;
    }



    /** 
     * \brief The first composition derivative of \f$a_m\f$ as well as derivatives with respect to \f$\tau\f$
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param x The vector of mole fractions
     * \param itau The number of derivatives of \f$a_m\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_m, itau=1 is d(a_m)/d(tau), etc.)
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    virtual TYPE d_am_term_dxi(TYPE tau, const std::vector<TYPE> &x, std::size_t itau, std::size_t i, bool xN_independent) const
    {
        if (xN_independent)
        {
            TYPE summer = 0;
            for (int j = N-1; j >= 0; --j)
            {
                summer += x[j]*aij_term(tau, i, j, itau);
            }
            return 2.0*summer;
        }
        else{
            TYPE summer = 0;
            for (int k = N-2; k >= 0; --k)
            {
                summer += x[k]*(aij_term(tau, i, k, itau)-aij_term(tau, k, N-1, itau));
            }
            return 2.0*(summer + x[N-1]*(aij_term(tau, N-1, i, itau) - aij_term(tau, N-1, N-1, itau)));
        }
    }
    /** 
     * \brief The second composition derivative of \f$a_m\f$ as well as derivatives with respect to \f$\tau\f$
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param x The vector of mole fractions
     * \param itau The number of derivatives of \f$a_m\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_m, itau=1 is d(a_m)/d(tau), etc.)
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    virtual TYPE d2_am_term_dxidxj(TYPE tau, const std::vector<TYPE> &x, std::size_t itau, std::size_t i, std::size_t j, bool xN_independent) const
    {
        if (xN_independent)
        {
            return 2.0*aij_term(tau, i, j, itau);
        }
        else{
            return 2.0*(aij_term(tau, i, j, itau)-aij_term(tau, j, N-1, itau)-aij_term(tau, N-1, i, itau)+aij_term(tau, N-1, N-1, itau));
        }
    }

    /** 
     * \brief The third composition derivative of \f$a_m\f$ as well as derivatives with respect to \f$\tau\f$
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param x The vector of mole fractions
     * \param itau The number of derivatives of \f$a_m\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_m, itau=1 is d(a_m)/d(tau), etc.)
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    virtual TYPE d3_am_term_dxidxjdxk(TYPE tau, const std::vector<TYPE> &x, std::size_t itau, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        return 0;
    }

    /** 
     * \brief The term \f$b_{\rm m}\f$ (mixture co-volume)
     * \param x The vector of mole fractions
     */
    virtual TYPE bm_term(const std::vector<TYPE> &x) const
    {
        TYPE summer = 0;
        for(int i = N-1; i >= 0; --i)
        {
            summer += x[i]*b0_ii(i);
        }
        return summer;
    }

    /** \brief The first composition derivative of \f$b_m\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    virtual TYPE d_bm_term_dxi(const std::vector<TYPE> &x, std::size_t i, bool xN_independent) const
    {
        if (xN_independent)
        {
            return b0_ii(i);
        }
        else{
            return b0_ii(i) - b0_ii(N-1);
        }
    }

    /** \brief The second composition derivative of \f$b_m\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    virtual TYPE d2_bm_term_dxidxj(const std::vector<TYPE> &x, std::size_t i, std::size_t j, bool xN_independent) const
    {
        return 0;
    }

    /** \brief The third composition derivative of \f$b_m\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    virtual TYPE d3_bm_term_dxidxjdxk(const std::vector<TYPE> &x, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        return 0;
    }

    /** 
     * \brief The n-th \f$\tau\f$ derivative of \f$a_{ij}(\tau)\f$
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param i The first index
     * \param j The second index
     * \param itau The number of derivatives of \f$a_{ij}\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_{ij}, itau=1 is d(a_ij)/d(tau), etc.)
     */
    TYPE aij_term(TYPE tau, std::size_t i, std::size_t j, std::size_t itau) const
    {
        TYPE u = u_term(tau, i, j, 0);
            
        switch (itau){
        case 0:
            return (1.0-k[i][j])*sqrt(u);
        case 1:
            return (1.0-k[i][j])/(2.0*sqrt(u))*u_term(tau, i, j, 1);
        case 2:
            return (1.0-k[i][j])/(4.0*pow(u,3.0/2.0))*(2.0*u*u_term(tau, i, j, 2)-pow(u_term(tau, i, j, 1), 2));
        case 3:
            return (1.0-k[i][j])/(8.0*pow(u,5.0/2.0))*(4.0*pow(u,2)*u_term(tau, i, j, 3)
                                                    -6.0*u*u_term(tau, i, j, 1)*u_term(tau, i, j, 2)
                                                    +3.0*pow(u_term(tau, i, j, 1),3));
        case 4:
            return (1.0-k[i][j])/(16.0*pow(u,7.0/2.0))*(-4.0*pow(u,2)*(4.0*u_term(tau, i, j, 1)*u_term(tau, i, j, 3) + 3.0*pow(u_term(tau, i, j, 2),2))
                                                    +8.0*pow(u,3)*u_term(tau, i, j, 4) + 36.0*u*pow(u_term(tau, i, j, 1),2)*u_term(tau, i, j, 2)
                                                    -15.0*pow(u_term(tau, i, j, 1), 4)
                                                    );
        default:
            throw -1;
        }
    }


    /** The n-th tau derivative of \f$u(\tau)\f$, the argument of sqrt in the cross aij term
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param i The first index
     * \param j The first index
     * \param itau The number of derivatives of \f$a_{ij}\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_{ij}, itau=1 is d(a_ij)/d(tau), etc.)
     */
    TYPE u_term(TYPE tau, std::size_t i, std::size_t j, std::size_t itau) const
    {
        TYPE aii = aii_term(tau, i, 0), ajj = aii_term(tau, j, 0);
        switch (itau){
        case 0:
            return aii*ajj;
        case 1:
            return aii*aii_term(tau, j, 1) + ajj*aii_term(tau, i, 1);
        case 2:
            return (aii*aii_term(tau, j, 2) 
                +2.0*aii_term(tau, i, 1)*aii_term(tau, j, 1) 
                +ajj*aii_term(tau, i, 2)
                );
        case 3:
            return (aii*aii_term(tau, j, 3) 
                    +3.0*aii_term(tau, i, 1)*aii_term(tau, j, 2) 
                    +3.0*aii_term(tau, i, 2)*aii_term(tau, j, 1) 
                    +ajj*aii_term(tau, i, 3)
                    );
        case 4:
            return (aii*aii_term(tau, j, 4) 
                    +4.0*aii_term(tau, i, 1)*aii_term(tau, j, 3) 
                    +6.0*aii_term(tau, i, 2)*aii_term(tau, j, 2) 
                    +4.0*aii_term(tau, i, 3)*aii_term(tau, j, 1) 
                    +ajj*aii_term(tau, i, 4)
                    );
        default:
            throw -1;
        }
    }

    /** Take the n-th tau derivative of the \f$a_{ii}(\tau)\f$ pure fluid contribution
     * \param tau The reciprocal reduced temperature \f$\tau=T_r/T\f$
     * \param i The index of the component
     * \param itau The number of derivatives of \f$u\f$ to take with respect to \f$\tau\f$ (itau=0 is just a_{ij}, itau=1 is d(a_ij)/d(tau), etc.)
     */
    TYPE aii_term(TYPE tau, std::size_t i, std::size_t itau) const
    {
        TYPE Tr_over_Tci = T_r/Tc[i];
        TYPE sqrt_Tr_Tci = sqrt(Tr_over_Tci);
        // If we are not using the full Mathias-Copeman formulation for a_ii, 
        // we just use the simple results from the supplemental information because 
        // they are much more computationally efficient
        if (simple_aii){
            // All derivatives have a common bracketed term, so we factor it out
            // and calculate it here
            TYPE m = m_ii(i);
            TYPE B = 1.0 + m*(1.0-sqrt_Tr_Tci*sqrt(1.0/tau));

            switch (itau){
            case 0:
                return a0_ii(i)*B*B;
            case 1:
                return a0_ii(i)*m*B/pow(tau, 3.0/2.0)*sqrt_Tr_Tci;
            case 2:
                return a0_ii(i)*m/2.0*(m/pow(tau, 3)*Tr_over_Tci - 3.0*B/pow(tau, 5.0/2.0)*sqrt_Tr_Tci);
            case 3:
                return (3.0/4.0)*a0_ii(i)*m*(-3.0*m/pow(tau, 4)*Tr_over_Tci + 5.0*B/pow(tau, 7.0/2.0)*sqrt_Tr_Tci);
            case 4:
                return (3.0/8.0)*a0_ii(i)*m*(29.0*m/pow(tau, 5)*Tr_over_Tci - 35.0*B/pow(tau, 9.0/2.0)*sqrt_Tr_Tci);
            default:
                throw -1;
            }
        }
        else{
            // Here we are using the full Mathias-Copeman formulation, introducing 
            // some additional computational effort, so we only evaluate the parameters that 
            // we actually need to evaluate, otherwise we just set their value to zero
            // See info on the conditional (ternary) operator : http://www.cplusplus.com/articles/1AUq5Di1/
            // Furthermore, this should help with branch prediction
            TYPE Di = 1.0-sqrt_Tr_Tci/sqrt(tau);
            TYPE dDi_dtau = (itau >= 1) ? (1.0/2.0)*sqrt_Tr_Tci/(pow(tau,1.5)) : 0;
            TYPE d2Di_dtau2 = (itau >= 2) ? -(3.0/4.0)*sqrt_Tr_Tci/(pow(tau,2.5)) : 0;
            TYPE d3Di_dtau3 = (itau >= 3) ? (15.0/8.0)*sqrt_Tr_Tci/(pow(tau,3.5)) : 0;
            TYPE d4Di_dtau4 = (itau >= 4) ? -(105.0/16.0)*sqrt_Tr_Tci/(pow(tau,4.5)) : 0;

            TYPE Bi = 1, dBi_dtau = 0, d2Bi_dtau2 = 0, d3Bi_dtau3 = 0, d4Bi_dtau4 = 0;
            for (double n = 1; n <= 3; ++n){
                const std::vector<TYPE> &C = get_C_ref(static_cast<int>(n));
                Bi += C[i]*pow(Di, n);
                dBi_dtau += (itau < 1) ? 0 : (n*C[i]*pow(Di,n-1)*dDi_dtau) ;
                d2Bi_dtau2 += (itau < 2) ? 0 : n*C[i]*((n-1)*pow(dDi_dtau,2) + Di*d2Di_dtau2)*pow(Di, n-2);
                d3Bi_dtau3 += (itau < 3) ? 0 : n*C[i]*(3.0*(n-1)*Di*dDi_dtau*d2Di_dtau2 + (n*n-3*n+2)*pow(dDi_dtau,3)+pow(Di,2)*d3Di_dtau3)*pow(Di, n-3);
                d4Bi_dtau4 += (itau < 4) ? 0 : n*C[i]*(6.0*(n*n-3.0*n+2)*Di*pow(dDi_dtau,2)*d2Di_dtau2 + (n*n*n-6.0*n*n+11.0*n-6)*pow(dDi_dtau,4)
                                                    +(4.0*n*dDi_dtau*d3Di_dtau3+3.0*n*pow(d2Di_dtau2,2)-4.0*dDi_dtau*d3Di_dtau3-3.0*pow(d2Di_dtau2,2) )*pow(Di,2) 
                                                    + pow(Di,3)*d4Di_dtau4 )*pow(Di, n-4);
            }
            switch (itau){
            case 0:
                return a0_ii(i)*Bi*Bi;
            case 1:
                return 2.0*a0_ii(i)*Bi*dBi_dtau;
            case 2:
                return 2.0*a0_ii(i)*(Bi*d2Bi_dtau2 + dBi_dtau*dBi_dtau);
            case 3:
                return 2.0*a0_ii(i)*(Bi*d3Bi_dtau3 + 3.0*dBi_dtau*d2Bi_dtau2);
            case 4:
                return 2.0*a0_ii(i)*(Bi*d4Bi_dtau4 + 4.0*dBi_dtau*d3Bi_dtau3 + 3.0*pow(d2Bi_dtau2, 2));
            default:
                throw -1;
            }
        }
    }

    /**
     * \brief The term \f$ \psi^{(-)}\f$ and its \f$\tau\f$ and \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     */
    TYPE psi_minus(TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta) const
    {
        if (itau > 0) return 0.0;
        TYPE b = bm_term(x);
        TYPE bracket = 1.0-b*delta*rho_r;

        switch(idelta){
        case 0:
            return -log(bracket);
        case 1:
            return b*rho_r/bracket;
        case 2:
            return pow(b*rho_r/bracket, 2);
        case 3:
            return 2.0*pow(b*rho_r/bracket, 3);
        case 4:
            return 6.0*pow(b*rho_r/bracket, 4);
        default:
            throw -1;
        }
    }

    /**
     * \brief The third composition derivative of \f$ \psi^{(-)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
     TYPE d_psi_minus_dxi(TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta, std::size_t i, bool xN_independent) const
     {
        if (itau > 0) return 0.0;
        TYPE b = bm_term(x);
        TYPE db_dxi = d_bm_term_dxi(x, i, xN_independent);
        TYPE bracket = 1.0-b*delta*rho_r;

        switch(idelta){
        case 0:
            return delta*rho_r*db_dxi/bracket;
        case 1:
            return rho_r*db_dxi/pow(bracket, 2);
        case 2:
            return 2.0*pow(rho_r,2)*b*db_dxi/pow(bracket, 3);
        case 3:
            return 6.0*pow(rho_r,3)*pow(b, 2)*db_dxi/pow(bracket, 4);
        case 4:
            return 24.0*pow(rho_r,4)*pow(b, 3)*db_dxi/pow(bracket, 5);
        default:
            throw -1;
        }
    }

    /**
     * \brief The second composition derivative of \f$ \psi^{(-)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
     TYPE d2_psi_minus_dxidxj(TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta, std::size_t i, std::size_t j, bool xN_independent) const
     {
        if (itau > 0) return 0.0;
        TYPE b = bm_term(x);
        TYPE db_dxi = d_bm_term_dxi(x, i, xN_independent), 
            db_dxj = d_bm_term_dxi(x, j, xN_independent),
            d2b_dxidxj = d2_bm_term_dxidxj(x, i, j, xN_independent);
        TYPE bracket = 1.0-b*delta*rho_r;

        switch(idelta){
        case 0:
            return pow(delta*rho_r, 2)*db_dxi*db_dxj/pow(bracket, 2) + delta*rho_r*d2b_dxidxj/bracket;
        case 1:
            return 2.0*delta*pow(rho_r, 2)*db_dxi*db_dxj/pow(bracket, 3) + rho_r*d2b_dxidxj/pow(bracket, 2);
        case 2:
            return 2.0*pow(rho_r,2)*db_dxi*db_dxj/pow(bracket, 4)*(2.0*delta*rho_r*b+1.0) + 2.0*pow(rho_r, 2)*b*d2b_dxidxj/pow(bracket,3);
        case 3:
            return 12.0*pow(rho_r,3)*b*db_dxi*db_dxj/pow(bracket, 5)*(delta*rho_r*b+1.0) + 6.0*pow(rho_r, 3)*pow(b,2)*d2b_dxidxj/pow(bracket,4);
        case 4:
            return 24.0*pow(rho_r,4)*pow(b, 2)*db_dxi*db_dxj/pow(bracket, 6)*(2.0*delta*rho_r*b + 3.0) + 24.0*pow(rho_r, 4)*pow(b,3)*d2b_dxidxj/pow(bracket,5);
        default:
            throw -1;
        }
    }

    /**
     * \brief The third composition derivative of \f$ \psi^{(-)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d3_psi_minus_dxidxjdxk(TYPE delta, const std::vector<TYPE> &x, std::size_t itau, std::size_t idelta, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        if (itau > 0) return 0.0;
        TYPE b = bm_term(x);
        TYPE db_dxi = d_bm_term_dxi(x, i, xN_independent), 
            db_dxj = d_bm_term_dxi(x, j, xN_independent),
            db_dxk = d_bm_term_dxi(x, k, xN_independent),
            d2b_dxidxj = d2_bm_term_dxidxj(x, i, j, xN_independent),
            d2b_dxidxk = d2_bm_term_dxidxj(x, i, k, xN_independent),
            d2b_dxjdxk = d2_bm_term_dxidxj(x, j, k, xN_independent),
            d3b_dxidxjdxk = d3_bm_term_dxidxjdxk(x, i, j, k, xN_independent);
        TYPE bracket = 1.0-b*delta*rho_r;

        switch(idelta){
        case 0:
            return delta*rho_r*d3b_dxidxjdxk/bracket 
                + 2.0*pow(delta*rho_r, 3)*db_dxi*db_dxj*db_dxk/pow(bracket, 3)
                + pow(delta*rho_r, 2)/pow(bracket, 2)*(db_dxi*d2b_dxjdxk
                                                        +db_dxj*d2b_dxidxk
                                                        +db_dxk*d2b_dxidxj);
        case 1:
            return rho_r*d3b_dxidxjdxk/pow(bracket, 2) 
                + 6.0*pow(delta, 2)*pow(rho_r, 3)*db_dxi*db_dxj*db_dxk/pow(bracket, 4)
                + 2.0*delta*pow(rho_r, 2)/pow(bracket, 3)*(db_dxi*d2b_dxjdxk
                                                            +db_dxj*d2b_dxidxk
                                                            +db_dxk*d2b_dxidxj);
        default:
            throw -1;
        }
    }


    /**
     * \brief The term \f$ \Pi_{12}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * 
     * \f[ \Pi_{12} = (1+\Delta_1\bm\rhor \delta)(1+\Delta_2\bm\rhor \delta) \f]
     */
    TYPE PI_12(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta) const
    {
        TYPE b = bm_term(x);
        switch(idelta){
        case 0:
            return (1.0+Delta_1*b*rho_r*delta)*(1.0+Delta_2*b*rho_r*delta);
        case 1:
            return b*rho_r*(2.0*Delta_1*Delta_2*b*delta*rho_r+Delta_1+Delta_2);
        case 2:
            return 2.0*Delta_1*Delta_2*pow(b*rho_r, 2);
        case 3:
            return 0;
        case 4:
            return 0;
        default:
            throw -1;
        }
    }

    /**
     * \brief The first composition derivative of \f$ \Pi_{12}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d_PI_12_dxi(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta, std::size_t i, bool xN_independent) const
    {
        TYPE b = bm_term(x);
        TYPE db_dxi = d_bm_term_dxi(x, i, xN_independent);
        switch(idelta){
        case 0:
            return delta*rho_r*db_dxi*(2.0*Delta_1*Delta_2*b*delta*rho_r+Delta_1+Delta_2);
        case 1:
            return rho_r*db_dxi*(4.0*Delta_1*Delta_2*b*delta*rho_r+Delta_1+Delta_2);
        case 2:
            return 4.0*Delta_1*Delta_2*pow(rho_r, 2)*b*db_dxi;
        case 3:
            return 0;
        case 4:
            return 0;
        default:
            throw -1;
        }
    }
    /**
     * \brief The second composition derivative of \f$ \Pi_{12}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d2_PI_12_dxidxj(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta, std::size_t i, std::size_t j, bool xN_independent) const
    {
        TYPE b = bm_term(x);
        TYPE db_dxi = d_bm_term_dxi(x, i, xN_independent),  
            db_dxj = d_bm_term_dxi(x, j, xN_independent),
            d2b_dxidxj = d2_bm_term_dxidxj(x, i, j, xN_independent);
        switch(idelta){
        case 0:
            return delta*rho_r*(2.0*Delta_1*Delta_2*delta*rho_r*db_dxi*db_dxj + (2.0*Delta_1*Delta_2*delta*rho_r*b+Delta_1+Delta_2)*d2b_dxidxj);
        case 1:
            return rho_r*(4.0*Delta_1*Delta_2*delta*rho_r*db_dxi*db_dxj + (4.0*Delta_1*Delta_2*delta*rho_r*b+Delta_1+Delta_2)*d2b_dxidxj);
        case 2:
            return 4.0*Delta_1*Delta_2*pow(rho_r,2)*(db_dxi*db_dxj + b*d2b_dxidxj);
        case 3:
            return 0;
        case 4:
            return 0;
        default:
            throw -1;
        }
    }

    /**
     * \brief The third composition derivative of \f$ \Pi_{12}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d3_PI_12_dxidxjdxk(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        TYPE b = bm_term(x);
        TYPE db_dxi = d_bm_term_dxi(x, i, xN_independent),  
            db_dxj = d_bm_term_dxi(x, j, xN_independent),
            db_dxk = d_bm_term_dxi(x, k, xN_independent),
            d2b_dxidxj = d2_bm_term_dxidxj(x, i, j, xN_independent),
            d2b_dxidxk = d2_bm_term_dxidxj(x, i, k, xN_independent),
            d2b_dxjdxk = d2_bm_term_dxidxj(x, j, k, xN_independent),
            d3b_dxidxjdxk = d3_bm_term_dxidxjdxk(x, i, j, k, xN_independent);
        switch(idelta){
        case 0:
            return delta*rho_r*((2.0*Delta_1*Delta_2*delta*rho_r*b+Delta_1+Delta_2)*d3b_dxidxjdxk 
                                + 2.0*Delta_1*Delta_2*delta*rho_r*(db_dxi*d2b_dxjdxk
                                +db_dxj*d2b_dxidxk
                                +db_dxk*d2b_dxidxj
                                ) 
                                );
        case 1:
            return rho_r*((4.0*Delta_1*Delta_2*delta*rho_r*b+Delta_1+Delta_2)*d3b_dxidxjdxk 
                        + 4.0*Delta_1*Delta_2*delta*rho_r*(db_dxi*d2b_dxjdxk
                        + db_dxj*d2b_dxidxk
                        + db_dxk*d2b_dxidxj
                            ) 
                        );
        default:
            throw -1;
        }
    }
    
    /**
     * \brief The term \f$ \tau\cdot a_m(\tau)\f$ and its \f$ \tau \f$ derivatives
     * \param tau The reciprocal reduced temperature \f$\tau = \frac{T_c}{T}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     */
    TYPE tau_times_a(TYPE tau, const std::vector<TYPE> &x, std::size_t itau) const
    {
        if (itau == 0){
            return tau*am_term(tau, x, 0);
        }
        else{
            return tau*am_term(tau,x,itau) + static_cast<double>(itau)*am_term(tau,x,itau-1);
        }
    }
    /**
     * \brief The first composition derivative of \f$ \tau\cdot a_m(\tau)\f$ and its \f$ \tau \f$ derivatives
     * \param tau The reciprocal reduced temperature \f$\tau = \frac{T_c}{T}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d_tau_times_a_dxi(TYPE tau, const std::vector<TYPE> &x, std::size_t itau, std::size_t i, bool xN_independent) const
    {
        if (itau == 0){
            return tau*d_am_term_dxi(tau, x, 0,i,xN_independent);
        }
        else{
            return tau*d_am_term_dxi(tau,x,itau,i,xN_independent) + static_cast<double>(itau)*d_am_term_dxi(tau,x,itau-1,i,xN_independent);
        }
    }

    /**
     * \brief The second composition derivative of \f$ \tau\cdot a_m(\tau)\f$ and its \f$ \tau \f$ derivatives
     * \param tau The reciprocal reduced temperature \f$\tau = \frac{T_c}{T}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d2_tau_times_a_dxidxj(TYPE tau, const std::vector<TYPE> &x, std::size_t itau, std::size_t i, std::size_t j, bool xN_independent) const
    {
        if (itau == 0){
            return tau*d2_am_term_dxidxj(tau, x, 0,i,j,xN_independent);
        }
        else{
            return tau*d2_am_term_dxidxj(tau,x,itau,i,j,xN_independent) + static_cast<double>(itau)*d2_am_term_dxidxj(tau,x,itau-1,i,j,xN_independent);
        }
    }

    /**
     * \brief The third composition derivative of \f$ \tau\cdot a_m(\tau)\f$ and its \f$ \tau \f$ derivatives
     * \param tau The reciprocal reduced temperature \f$\tau = \frac{T_c}{T}\f$
     * \param x The vector of mole fractions
     * \param itau How many derivatives to take with respect to \f$\tau\f$
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d3_tau_times_a_dxidxjdxk(TYPE tau, const std::vector<TYPE> &x, std::size_t itau, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        if (itau == 0){
            return tau*d3_am_term_dxidxjdxk(tau, x, 0,i,j,k,xN_independent);
        }
        else{
            return tau*d3_am_term_dxidxjdxk(tau,x,itau,i,j,k,xN_independent) + static_cast<TYPE>(itau)*d3_am_term_dxidxjdxk(tau,x,itau-1,i,j,k,xN_independent);
        }
    }


    /**
     * \brief The term \f$ \psi^{(+)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * 
     * \f[  \psi^{(+)} = \dfrac{\ln\left(\dfrac{\Delta_1\bm\rhor \delta+1}{\Delta_2\bm\rhor \delta+1}\right)}{\bm(\Delta_1-\Delta_2)}  \f]
     */
    TYPE psi_plus(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta) const
    {
        switch(idelta){
        case 0:
            return A_term(delta, x)*c_term(x)/(Delta_1-Delta_2);
        case 1:
            return rho_r/PI_12(delta,x,0);
        case 2:
            return -rho_r/pow(PI_12(delta,x,0),2)*PI_12(delta,x,1);
        case 3:
            return rho_r*(-PI_12(delta,x,0)*PI_12(delta,x,2)+2.0*pow(PI_12(delta,x,1),2))/pow(PI_12(delta,x,0),3);
        case 4:
            // Term -PI_12(delta,x,0)*PI_12(delta,x,3) in the numerator is zero (and removed) since PI_12(delta,x,3) = 0
            return rho_r*(6.0*PI_12(delta,x,0)*PI_12(delta,x,1)*PI_12(delta,x,2) - 6.0*pow(PI_12(delta,x,1),3))/pow(PI_12(delta,x,0),4);
        default:
            throw -1;
        }
    }

    /**
     * \brief The first composition derivative of \f$ \psi^{(+)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d_psi_plus_dxi(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta, std::size_t i, bool xN_independent) const
    {
        TYPE bracket = 0;
        if (idelta == 0){
            return (A_term(delta, x)*d_c_term_dxi(x, i, xN_independent) + c_term(x)*d_A_term_dxi(delta, x, i, xN_independent))/(Delta_1 - Delta_2);
        }
        // All the terms with at least one delta derivative are multiplied by a common term of -rhor/PI12^2
        // So we just evaluated the bracketed term and then multiply by the common factor in the front
        switch(idelta){
        case 1:
            bracket = d_PI_12_dxi(delta, x, 0, i, xN_independent); break;
        case 2:
            bracket = (d_PI_12_dxi(delta, x, 1, i, xN_independent)
                    + 2.0/rho_r*PI_12(delta,x,0)*PI_12(delta,x,1)*d_psi_plus_dxi(delta, x,1,i,xN_independent)
                    );
            break;
        case 3:{
            bracket = (d_PI_12_dxi(delta, x, 2, i, xN_independent)
                    + 2.0/rho_r*(pow(PI_12(delta,x,1), 2) + PI_12(delta,x,0)*PI_12(delta,x,2))*d_psi_plus_dxi(delta, x,1,i,xN_independent)
                    + 4.0/rho_r*PI_12(delta,x,0)*PI_12(delta,x,1)*d_psi_plus_dxi(delta, x,2,i,xN_independent)
                    );
            break;
        }
        case 4:
            // d_PI_12_dxi(delta, x, 3, i, xN_independent) = 0, and PI_12(delta,x,0)*PI_12(delta,x,3) = 0, so removed from sum
            bracket = (6.0/rho_r*PI_12(delta,x,1)*PI_12(delta,x,2)*d_psi_plus_dxi(delta, x,1,i,xN_independent)
                    + 6.0/rho_r*(pow(PI_12(delta,x,1), 2) + PI_12(delta,x,0)*PI_12(delta,x,2))*d_psi_plus_dxi(delta, x,2,i,xN_independent)
                    + 6.0/rho_r*PI_12(delta,x,0)*PI_12(delta,x,1)*d_psi_plus_dxi(delta, x,3,i,xN_independent)
                    );
            break;
        default:
            throw -1;
        }
        return -rho_r/pow(PI_12(delta,x,0), 2)*bracket;
    }

    /**
     * \brief The second composition derivative of \f$ \psi^{(+)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d2_psi_plus_dxidxj(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta, std::size_t i, std::size_t j, bool xN_independent) const
    {
        TYPE bracket = 0;
        TYPE PI12 = PI_12(delta, x, 0);
        if (idelta == 0){
            return (A_term(delta, x)*d2_c_term_dxidxj(x, i, j, xN_independent) 
                    +c_term(x)*d2_A_term_dxidxj(delta, x, i, j, xN_independent)
                    +d_A_term_dxi(delta, x, i, xN_independent)*d_c_term_dxi(x, j, xN_independent) 
                    +d_A_term_dxi(delta, x, j, xN_independent)*d_c_term_dxi(x, i, xN_independent) 
                    )/(Delta_1 - Delta_2);
        }
        // All the terms with at least one delta derivative have a common factor of -1/PI_12^2 out front
        // so we just calculate the bracketed term and then multiply later on
        switch(idelta){
        case 1:
            bracket = (rho_r*d2_PI_12_dxidxj(delta, x, 0, i, j, xN_independent) 
                    + 2.0*PI12*d_PI_12_dxi(delta, x, 0, j, xN_independent)*d_psi_plus_dxi(delta, x, 1, i, xN_independent)
                    ); 
            break;
        case 2:
            bracket = (rho_r*d2_PI_12_dxidxj(delta, x, 1, i, j, xN_independent) 
                    + 2.0*(PI12*d_PI_12_dxi(delta, x, 1, j, xN_independent) 
                            + PI_12(delta, x, 1)*d_PI_12_dxi(delta, x, 0, j, xN_independent)
                            )*d_psi_plus_dxi(delta, x, 1, i, xN_independent)
                    + 2.0*PI12*PI_12(delta, x, 1)*d2_psi_plus_dxidxj(delta, x, 1, i, j, xN_independent) 
                    + 2.0*PI12*d_PI_12_dxi(delta, x, 0, j, xN_independent)*d_psi_plus_dxi(delta, x, 2, i, xN_independent)
                    ); 
            break;
        case 3:{
            bracket = (rho_r*d2_PI_12_dxidxj(delta, x, 2, i, j, xN_independent) 
                    + 2.0*(PI12*PI_12(delta, x, 2) + pow(PI_12(delta, x, 1), 2))*d2_psi_plus_dxidxj(delta, x, 1, i, j, xN_independent)
                    + 4.0*(PI12*d_PI_12_dxi(delta, x, 1, j, xN_independent) 
                            + PI_12(delta, x, 1)*d_PI_12_dxi(delta, x, 0, j, xN_independent)
                            )*d_psi_plus_dxi(delta, x, 2, i, xN_independent)
                    + 2.0*(  PI12*d_PI_12_dxi(delta, x, 2, j, xN_independent) 
                            + 2.0*PI_12(delta, x, 1)*d_PI_12_dxi(delta, x, 1, j, xN_independent)
                            + d_PI_12_dxi(delta, x, 0, j, xN_independent)*PI_12(delta, x, 2)
                            )*d_psi_plus_dxi(delta, x, 1, i, xN_independent)
                    + 4.0*PI12*PI_12(delta, x, 1)*d2_psi_plus_dxidxj(delta, x, 2, i, j, xN_independent) 
                    + 2.0*PI12*d_PI_12_dxi(delta, x, 0, j, xN_independent)*d_psi_plus_dxi(delta, x, 3, i, xN_independent)
                    ); 
            break;
        }
        case 4:
            // rho_r*d2_PI_12_dxidxj(delta, x, 3, i, j, xN_independent)  = 0
            // PI_12(delta, x, 3) = 0
            // PI12*d_PI_12_dxi(delta, x, 3, j, xN_independent) = 0
            // d_PI_12_dxi(delta, x, 0, j, xN_independent)*PI_12(delta, x, 3) = 0
            bracket = (
                    + 6.0*(PI12*PI_12(delta, x, 2) + pow(PI_12(delta, x, 1), 2))*d2_psi_plus_dxidxj(delta, x, 2, i, j, xN_independent)
                    + 6.0*PI_12(delta, x, 1)*PI_12(delta, x, 2)*d2_psi_plus_dxidxj(delta, x, 1, i, j, xN_independent)
                    + 6.0*(PI12*d_PI_12_dxi(delta, x, 1, j, xN_independent) 
                            + PI_12(delta, x, 1)*d_PI_12_dxi(delta, x, 0, j, xN_independent)
                            )*d_psi_plus_dxi(delta, x, 3, i, xN_independent)
                    + 6.0*(PI12*d_PI_12_dxi(delta, x, 2, j, xN_independent) 
                            + 2.0*PI_12(delta, x, 1)*d_PI_12_dxi(delta, x, 1, j, xN_independent)
                            + d_PI_12_dxi(delta, x, 0, j, xN_independent)*PI_12(delta, x, 2)
                            )*d_psi_plus_dxi(delta, x, 2, i, xN_independent)
                    + 6.0*(PI_12(delta, x, 1)*d_PI_12_dxi(delta, x, 2, j, xN_independent)
                            + PI_12(delta, x, 2)*d_PI_12_dxi(delta, x, 1, j, xN_independent)
                            )*d_psi_plus_dxi(delta, x, 1, i, xN_independent)
                    + 6.0*PI12*PI_12(delta, x, 1)*d2_psi_plus_dxidxj(delta, x, 3, i, j, xN_independent) 
                    + 2.0*PI12*d_PI_12_dxi(delta, x, 0, j, xN_independent)*d_psi_plus_dxi(delta, x, 4, i, xN_independent)
                    ); 
            break;
        default:
            throw -1;
        }
        return -1.0/pow(PI12, 2)*bracket;
    }

    /**
     * \brief The third composition derivative of \f$ \psi^{(+)}\f$ and its \f$ \delta \f$ derivatives
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param idelta How many derivatives to take with respect to \f$\delta\f$
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d3_psi_plus_dxidxjdxk(TYPE delta, const std::vector<TYPE> &x, std::size_t idelta, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const
    {
        TYPE PI12 = PI_12(delta, x, 0);
        switch (idelta){
        case 0:
            return (A_term(delta, x)*d3_c_term_dxidxjdxk(x, i, j, k, xN_independent) 
                +c_term(x)*d3_A_term_dxidxjdxk(delta, x, i, j, k, xN_independent)
                +d_A_term_dxi(delta, x, i, xN_independent)*d2_c_term_dxidxj(x, j, k, xN_independent) 
                +d_A_term_dxi(delta, x, j, xN_independent)*d2_c_term_dxidxj(x, i, k, xN_independent) 
                +d_A_term_dxi(delta, x, k, xN_independent)*d2_c_term_dxidxj(x, i, j, xN_independent) 
                +d_c_term_dxi(x, i, xN_independent)*d2_A_term_dxidxj(delta, x, j, k, xN_independent)
                +d_c_term_dxi(x, j, xN_independent)*d2_A_term_dxidxj(delta, x, i, k, xN_independent)
                +d_c_term_dxi(x, k, xN_independent)*d2_A_term_dxidxj(delta, x, i, j, xN_independent)
                )/(Delta_1 - Delta_2);
        case 1:
            return -1.0/pow(PI12, 2)*(rho_r*d3_PI_12_dxidxjdxk(delta, x, 0, i, j, k, xN_independent)
                                    +2.0*(PI12*d2_PI_12_dxidxj(delta, x, 0, j, k, xN_independent) + d_PI_12_dxi(delta, x, 0, j, xN_independent)*d_PI_12_dxi(delta, x, 0, k, xN_independent))*d_psi_plus_dxi(delta, x, 1, i, xN_independent)
                                    +2.0*PI12*d_PI_12_dxi(delta, x, 0, j, xN_independent)*d2_psi_plus_dxidxj(delta, x, 1, i, k, xN_independent) + 2.0*PI12*d_PI_12_dxi(delta, x, 0, k, xN_independent)*d2_psi_plus_dxidxj(delta, x, 1, i, j, xN_independent)
                                    );
        default:
            throw -1;
        }
    }


    /** \brief The term \f$c\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * 
     * \f$c\f$ is given by
     * \f[
     * c = \frac{1}{b_m}
     * \f]
     * \param x The vector of mole fractions
     */
    TYPE c_term(const std::vector<TYPE> &x) const{
        return 1.0/bm_term(x);
    };
    /**
     * \brief The first composition derivative of the term \f$c\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d_c_term_dxi(const std::vector<TYPE> &x, std::size_t i, bool xN_independent) const {
        return -d_bm_term_dxi(x,i,xN_independent)/pow(bm_term(x), 2);
    };
    /**
     * \brief The second composition derivative of the term \f$c\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d2_c_term_dxidxj(const std::vector<TYPE> &x, std::size_t i, std::size_t j, bool xN_independent) const {
        TYPE b = bm_term(x);
        return (2.0*d_bm_term_dxi(x, i, xN_independent)*d_bm_term_dxi(x, j, xN_independent) - b*d2_bm_term_dxidxj(x, i,j,xN_independent))/pow(b, 3);
    };
    /**
     * \brief The third composition derivative of the term \f$c\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d3_c_term_dxidxjdxk(const std::vector<TYPE> &x, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const {
        TYPE b = bm_term(x);
        return 1.0/pow(b,4)*(2.0*b*(d_bm_term_dxi(x, i, xN_independent)*d2_bm_term_dxidxj(x, j, k, xN_independent)
                                +d_bm_term_dxi(x, j, xN_independent)*d2_bm_term_dxidxj(x, i, k, xN_independent)
                                +d_bm_term_dxi(x, k, xN_independent)*d2_bm_term_dxidxj(x, i, j, xN_independent)
                                )
                           - pow(b,2)*d3_bm_term_dxidxjdxk(x, i,j,k,xN_independent)
                           -6.0*d_bm_term_dxi(x, i, xN_independent)*d_bm_term_dxi(x, j, xN_independent)*d_bm_term_dxi(x, k, xN_independent)
                           );
    };

    /**
     * \brief The term \f$A\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     *
     * \f[
     * A = \log\left(\frac{\Delta_1\delta\rho_r b_m+1}{\Delta_2\delta\rho_r b+1}\right)
     * \f]
     * 
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     */
    TYPE A_term(TYPE delta, const std::vector<TYPE> &x) const{
        TYPE b = bm_term(x);
        return log((Delta_1*delta*rho_r*b+1.0)/(Delta_2*delta*rho_r*b+1.0));
    };
    /**
     * \brief The first composition derivative of the term \f$A\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d_A_term_dxi(TYPE delta, const std::vector<TYPE> &x, std::size_t i, bool xN_independent) const {
        std::size_t idelta = 0;
        return delta*rho_r*d_bm_term_dxi(x,i,xN_independent)*(Delta_1-Delta_2)/PI_12(delta, x, idelta);
    };
    /**
     * \brief The second composition derivative of the term \f$A\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param j The second index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d2_A_term_dxidxj(TYPE delta, const std::vector<TYPE> &x, std::size_t i, std::size_t j, bool xN_independent) const {
        std::size_t idelta = 0;
        TYPE PI12 = PI_12(delta, x, idelta);
        return delta*rho_r*(Delta_1-Delta_2)/pow(PI12, 2)*(PI12*d2_bm_term_dxidxj(x,i,j,xN_independent) 
                                                           - d_PI_12_dxi(delta, x, 0, j,xN_independent)* d_bm_term_dxi(x,i,xN_independent));
    };
    /**
     * \brief The third composition derivative of the term \f$A\f$ used in the pure composition partial derivatives of \f$\psi^{(+)}\f$
     * \param delta The reduced density \f$\delta = \frac{\rho}{\rho_c}\f$
     * \param x The vector of mole fractions
     * \param i The first index
     * \param j The second index
     * \param k The third index
     * \param xN_independent True if \f$x_N\f$ is an independent variable, false otherwise (dependent on other \f$N-1\f$ mole fractions)
     */
    TYPE d3_A_term_dxidxjdxk(TYPE delta, const std::vector<TYPE> &x, std::size_t i, std::size_t j, std::size_t k, bool xN_independent) const {
        std::size_t idelta = 0;
        TYPE PI12 = PI_12(delta, x, idelta);
        // The leading factor
        TYPE lead = delta*rho_r*(Delta_1-Delta_2)/pow(PI12, 3);
        return lead*(-PI12*(d_PI_12_dxi(delta, x, idelta, j, xN_independent)*d2_bm_term_dxidxj(x,i,k,xN_independent) 
                            +d_PI_12_dxi(delta, x, idelta, k, xN_independent)*d2_bm_term_dxidxj(x,i,j,xN_independent) 
                            +d_bm_term_dxi(x,i,xN_independent)*d2_PI_12_dxidxj(delta, x, idelta, j, k, xN_independent))
                    +pow(PI12, 2)*d3_bm_term_dxidxjdxk(x, i, j, k, xN_independent)
                    +2.0*d_PI_12_dxi(delta, x, idelta, j, xN_independent)*d_PI_12_dxi(delta, x, idelta, k, xN_independent)*d_bm_term_dxi(x,i, xN_independent)
                    );
    };
};

template<typename TYPE = double>
class PengRobinson : public AbstractCubic<TYPE>
{
public:
    PengRobinson(const std::vector<TYPE> &Tc, 
                 const std::vector<TYPE> &pc, 
                 const std::vector<TYPE> &acentric,
                 TYPE R_u,
                 const std::vector<TYPE> &C1 = std::vector<TYPE>(),
                 const std::vector<TYPE> &C2 = std::vector<TYPE>(),
                 const std::vector<TYPE> &C3 = std::vector<TYPE>(),
                 const std::vector<std::vector<TYPE>> &k = std::vector<std::vector<TYPE>>()
                 ) 
        : AbstractCubic<TYPE>(Tc, pc, acentric, R_u, 1+sqrt(2.0), 1-sqrt(2.0),C1,C2,C3,k) {};

    PengRobinson(const std::vector<TYPE> &Tc,
        const std::vector<TYPE> &pc,
        const std::vector<TYPE> &acentric,
        TYPE R_u,
        const std::vector<std::vector<TYPE>> &kij = std::vector<std::vector<TYPE>>()
    )
    : AbstractCubic<TYPE>(Tc, pc, acentric, R_u, 1 + sqrt(2.0), 1 - sqrt(2.0), std::vector<TYPE>(), std::vector<TYPE>(), std::vector<TYPE>(), kij) {};

    PengRobinson(TYPE Tc, 
        TYPE pc, 
        TYPE acentric,
        TYPE R_u) 
        : AbstractCubic<TYPE>(std::vector<TYPE>(1,Tc), std::vector<TYPE>(1,pc), std::vector<TYPE>(1,acentric), R_u, 1+sqrt(2.0), 1-sqrt(2.0)) {};

    TYPE a0_ii(std::size_t i) const override{ 
        TYPE a = 0.45724*this->R_u*this->R_u*this->Tc[i]*this->Tc[i]/this->pc[i]; 
        return a; 
    }
    TYPE b0_ii(std::size_t i) const override{ 
        TYPE b = 0.07780*this->R_u*this->Tc[i]/this->pc[i];
        return b;
    }
    TYPE m_ii(std::size_t i) const override{  
        TYPE omega = this->acentric[i];
        TYPE m = 0.37464 + 1.54226*omega - 0.26992*omega*omega;
        return m;
    }
};

template<typename TYPE = double>
class SRK : public AbstractCubic<TYPE>
{
public:
    SRK(const std::vector<TYPE> &Tc, 
        const std::vector<TYPE> &pc, 
        const std::vector<TYPE> &acentric,
        TYPE R_u,
        const std::vector<TYPE> &C1 = std::vector<TYPE>(),
        const std::vector<TYPE> &C2 = std::vector<TYPE>(),
        const std::vector<TYPE> &C3 = std::vector<TYPE>()
        )
        : AbstractCubic<TYPE>(Tc, pc, acentric, R_u, 1, 0, C1, C2, C3) {};
    SRK(TYPE Tc, 
        TYPE pc, 
        TYPE acentric,
        TYPE R_u) 
        : AbstractCubic<TYPE>(std::vector<TYPE>(1,Tc), std::vector<TYPE>(1,pc), std::vector<TYPE>(1,acentric), R_u, 1, 0) {};

    TYPE a0_ii(std::size_t i) const override{
        // Values from Soave, 1972 (Equilibium constants from a ..)
        TYPE a = 0.42747*this->R_u*this->R_u*this->Tc[i]*this->Tc[i]/this->pc[i];
        return a;
    }
    TYPE b0_ii(std::size_t i) const override{
        // Values from Soave, 1972 (Equilibium constants from a ..)
        TYPE b = 0.08664*this->R_u*this->Tc[i]/this->pc[i];
        return b;
    }
    TYPE m_ii(std::size_t i) const override{
        // Values from Soave, 1972 (Equilibium constants from a ..)
        TYPE omega = this->acentric[i];
        TYPE m = 0.480 + 1.574*omega - 0.176*omega*omega;
        return m;
    }
};




#endif
