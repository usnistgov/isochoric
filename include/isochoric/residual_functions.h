#ifndef RESIDUAL_FUNCTIONS
#define RESIDUAL_FUNCTIONS

#include "mixderiv.h"
#include <functional>
#include <Eigen/Dense>
#include <vector>
#include <numeric>

// From CoolProp
#include "Solvers.h"
// #include "MatrixMath.h" // (for debug only)

using namespace CoolProp;

template<typename TYPE = double>
class IsothermVLEResiduals : public FuncWrapperND {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    typedef std::function<std::unique_ptr<const MixDerivs<TYPE> >(const TYPE, const EigenArray&)> DerivFactory;
private:
    DerivFactory m_factory;
    TYPE m_T, rhoL0_target;
    std::size_t imposed_index;
    std::vector<std::vector<TYPE> > J;
    std::vector<TYPE> y;
    std::unique_ptr<const MixDerivs<TYPE> > m_derL, m_derV;
public:
    std::size_t icall = 0;
    IsothermVLEResiduals(DerivFactory factory, TYPE T, TYPE rhoL0_target, std::size_t imposed_index) : m_factory(factory), m_T(T), rhoL0_target(rhoL0_target), imposed_index(imposed_index) {};
    const std::vector<TYPE> &get_errors() { return y; };
    std::vector<TYPE> call(const std::vector<TYPE> &rhovec) {
        assert(rhovec.size() % 2 == 0); // Even length
        std::size_t N = rhovec.size() / 2;
        y.resize(rhovec.size());

        const Eigen::ArrayXd rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(rhovec[0]), N);
        const Eigen::ArrayXd rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(rhovec[0]) + N, N);
        const TYPE rhomolarL = rhovecL.sum(), rhomolarV = rhovecV.sum();
        // Allocate the Jacobian matrix
        J = std::vector<std::vector<TYPE> >(2 * N, std::vector<TYPE>(2 * N, 0));

        // Do all the liquid calls first (only needs one update), then do all the vapor calls
        m_derL = m_factory(m_T, rhovecL); 
        m_derV = m_factory(m_T, rhovecV);
        auto &derL = *m_derL, &derV = *m_derV;

        TYPE PSILr = derL.psir();
        TYPE R = derL.R;
        
        TYPE chempot0Lr = derL.dpsir_drhoi__constTrhoj(0),
             chempot1Lr = derL.dpsir_drhoi__constTrhoj(1);
        TYPE buffer[2][2];
        for (std::size_t i = 0; i <= 1; ++i) {
            for (std::size_t j = i; j <= 1; ++j) {
                buffer[i][j] = derL.d2psir_drhoidrhoj__constT(i, j);
                buffer[j][i] = buffer[i][j];
            }
        }
        for (std::size_t i = 0; i <= 1; ++i) {
            J[i][0] = -buffer[i][0] - ((i == 0) ? R*m_T / rhovecL[0] : 0);
            J[i][1] = -buffer[i][1] - ((i == 1) ? R*m_T / rhovecL[1] : 0);
        }
        J[2][0] = R*m_T + rhovecL[0] * buffer[0][0] + rhovecL[1] * buffer[1][0];
        J[2][1] = R*m_T + rhovecL[0] * buffer[0][1] + rhovecL[1] * buffer[1][1];

        // Then we do everything with the vapor phase
        TYPE PSIVr = derV.psir();
        TYPE chempot0Vr = derV.dpsir_drhoi__constTrhoj(0),
               chempot1Vr = derV.dpsir_drhoi__constTrhoj(1);
        for (std::size_t i = 0; i <= 1; ++i) {
            for (std::size_t j = i; j <= 1; ++j) {
                buffer[i][j] = derV.d2psir_drhoidrhoj__constT(i, j);
                buffer[j][i] = buffer[i][j];
            }
        }
        for (std::size_t i = 0; i <= 1; ++i) {
            J[i][2] = buffer[i][0] + ((i == 0) ? R*m_T / rhovecV[0] : 0);
            J[i][3] = buffer[i][1] + ((i == 1) ? R*m_T / rhovecV[1] : 0);
        }
        J[2][2] = -R*m_T - rhovecV[0] * buffer[0][0] - rhovecV[1] * buffer[1][0];
        J[2][3] = -R*m_T - rhovecV[0] * buffer[0][1] - rhovecV[1] * buffer[1][1];
        J[3][imposed_index] = 1;

        // Calculate the residual terms
        y[0] = chempot0Vr + R*m_T*log(rhovecV[0]) - (chempot0Lr + R*m_T*log(rhovecL[0]));
        y[1] = chempot1Vr + R*m_T*log(rhovecV[1]) - (chempot1Lr + R*m_T*log(rhovecL[1]));
        TYPE pL = R*m_T*rhomolarL - PSILr + chempot0Lr*rhovecL[0] + chempot1Lr*rhovecL[1];
        TYPE pV = R*m_T*rhomolarV - PSIVr + chempot0Vr*rhovecV[0] + chempot1Vr*rhovecV[1];
        y[2] = pL - pV;
        y[3] = rhovecL[imposed_index] - rhoL0_target;

        for (std::size_t i = 0; i < y.size(); ++i) {
            if (!ValidNumber(y[i])) {
                throw ValueError("Invalid value found");
            }
        }
        icall++;
        return y;
    }
    std::vector<std::vector<TYPE> > Jacobian(const std::vector<TYPE> &rhovec)
    {
        /*int rr = 0;*/
        //std::vector<std::vector<TYPE> > Jnum = FuncWrapperND::Jacobian(rhovec); 
        //std::cout << CoolProp::vec_to_eigen(J) << std::endl << std::endl << CoolProp::vec_to_eigen(Jnum) << std::endl;
        return J;
    }
};

template<typename TYPE = double>
class IsobarVLEResiduals : public FuncWrapperND {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    typedef std::function<std::unique_ptr<const MixDerivs<TYPE> >(const TYPE, const EigenArray&)> DerivFactory;
private:
    DerivFactory m_factory; 
    TYPE m_p, m_T;
    std::unique_ptr<const MixDerivs<TYPE> > m_derL, m_derV;
public:
    IsobarVLEResiduals(DerivFactory factory, TYPE p, TYPE T) : m_factory(factory), m_p(p), m_T(T) {};
    std::vector<TYPE> call(const std::vector<TYPE> &rhovec) {
        assert(rhovec.size() % 2 == 0); // Even length
        std::vector<TYPE> y(rhovec.size(), _HUGE);
        std::size_t N = rhovec.size() / 2;
        const EigenArray rhovecL = Eigen::Map<const EigenArray>(&(rhovec[0]), N);
        const EigenArray rhovecV = Eigen::Map<const EigenArray>(&(rhovec[0]) + N, N);

        m_derL = m_factory(m_T, rhovecL);
        m_derV = m_factory(m_T, rhovecV);
        auto &derL = *m_derL, &derV = *m_derV;
        TYPE R = derL.R;

        TYPE pL = derL.p(), chempotr0L = derL.dpsir_drhoi__constTrhoj(0), chempotr1L = derL.dpsir_drhoi__constTrhoj(1);
        TYPE pV = derV.p(), chempotr0V = derV.dpsir_drhoi__constTrhoj(0), chempotr1V = derV.dpsir_drhoi__constTrhoj(1);

        y[0] = chempotr0L - chempotr0V + R*m_T*log(rhovecL[0]/rhovecV[0]);
        y[1] = chempotr1L - chempotr1V + R*m_T*log(rhovecL[1]/rhovecV[1]);
        y[2] = pL - m_p;
        y[3] = pV - m_p;
        for (std::size_t i = 0; i < y.size(); ++i) {
            if (!ValidNumber(y[i])) {
                throw ValueError("Invalid value found");
            }
        }
        return y;
    }
    std::vector<std::vector<TYPE> > Jacobian(const std::vector<TYPE> &rhovec)
    {
        std::size_t N = rhovec.size() / 2;
        std::vector<TYPE> rhovecL(rhovec.begin(), rhovec.begin() + N),
            rhovecV(rhovec.begin() + N, rhovec.end());
        auto &derL = *m_derL, &derV = *m_derV;
        TYPE R = derL.R;
        std::vector<std::vector<TYPE> > J(2 * N, std::vector<TYPE>(2 * N, 0));
        for (std::size_t i = 0; i <= 1; ++i) {
            J[i][0] = derL.d2psir_drhoidrhoj__constT(i, 0) + ((i == 0) ? R*m_T/rhovecL[0] : 0);
            J[i][1] = derL.d2psir_drhoidrhoj__constT(i, 1) + ((i == 1) ? R*m_T/rhovecL[1] : 0);
            J[i][2] = -derV.d2psir_drhoidrhoj__constT(i, 0) - ((i == 0) ? R*m_T/rhovecV[0] : 0);
            J[i][3] = -derV.d2psir_drhoidrhoj__constT(i, 1) - ((i == 1) ? R*m_T/rhovecV[1] : 0);
        }
        J[2][0] = derL.dpdrhoi__constTrhoj(0);
        J[2][1] = derL.dpdrhoi__constTrhoj(1);
        J[2][2] = 0;
        J[2][3] = 0;
        J[3][0] = 0;
        J[3][1] = 0;
        J[3][2] = derV.dpdrhoi__constTrhoj(0);
        J[3][3] = derV.dpdrhoi__constTrhoj(1);

        //int rr = 0;
        //std::vector<std::vector<TYPE> > Jnum = FuncWrapperND::Jacobian(rhovec);

        return J;
    }
};

template<typename TYPE = double>
class IsobarImposedRho0VLEResiduals : public FuncWrapperND {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    typedef std::function<std::unique_ptr<const MixDerivs<TYPE> >(const TYPE, const EigenArray&)> DerivFactory;
private:
    DerivFactory m_factory;
    TYPE m_p, m_rho0;
    std::vector<TYPE> y;
    std::size_t m_imposed_index;
    std::unique_ptr<const MixDerivs<TYPE> > m_derL, m_derV;
public:
    IsobarImposedRho0VLEResiduals(DerivFactory factory, TYPE p, TYPE rho0, std::size_t imposed_index) : m_factory(factory), m_p(p), m_rho0(rho0), m_imposed_index(imposed_index) {};
    const std::vector<TYPE> & get_errors(){ return y; }
    void unpack(const std::vector<TYPE> &lnrhovec, TYPE &T, std::vector<TYPE> &rhovecL, std::vector<TYPE> &rhovecV) {
        assert((lnrhovec.size() - 1) % 2 == 0); // Even length of N-1, first element is T
        std::size_t N = (lnrhovec.size() - 1) / 2;
        T = lnrhovec[0];
        std::vector<TYPE> lnrhovecL(lnrhovec.begin() + 1, lnrhovec.begin() + N + 1),
            lnrhovecV(lnrhovec.begin() + N + 1, lnrhovec.end());
        rhovecL.resize(N); rhovecV.resize(N);
        for (std::size_t i = 0; i <= 1; ++i) {
            rhovecL[i] = exp(lnrhovecL[i]);
            rhovecV[i] = exp(lnrhovecV[i]);
        }
    }
    std::vector<TYPE> call(const std::vector<TYPE> &lnrhovec) {
        TYPE T; std::vector<TYPE> _rhovecL, _rhovecV;
        unpack(lnrhovec, T, _rhovecL, _rhovecV);

        auto N = _rhovecL.size();
        const Eigen::ArrayXd rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(_rhovecL[0]), N);
        const Eigen::ArrayXd rhovecV = Eigen::Map<const Eigen::ArrayXd>(&(_rhovecV[0]), N);

        m_derL = m_factory(T, rhovecL);
        m_derV = m_factory(T, rhovecV);
        auto &derL = *(m_derL.get()), &derV = *(m_derV.get()); TYPE R = derL.R;

        TYPE pL = derL.p(), chempotr0L = derL.dpsir_drhoi__constTrhoj(0), chempotr1L = derL.dpsir_drhoi__constTrhoj(1);
        TYPE pV = derV.p(), chempotr0V = derV.dpsir_drhoi__constTrhoj(0), chempotr1V = derV.dpsir_drhoi__constTrhoj(1);

        y.resize(5);
        y[0] = chempotr0L - chempotr0V + R*T*log(rhovecL[0] / rhovecV[0]);
        y[1] = chempotr1L - chempotr1V + R*T*log(rhovecL[1] / rhovecV[1]);
        y[2] = pL - m_p;
        y[3] = pV - m_p;
        y[4] = rhovecL[m_imposed_index] - m_rho0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            if (!ValidNumber(y[i])) {
                throw ValueError("Invalid value found");
            }
        }
        return y;
    }
    std::vector<std::vector<TYPE> > Jacobian(const std::vector<TYPE> &lnrhovec)
    {
        TYPE T; std::vector<TYPE> rhovecL, rhovecV;
        unpack(lnrhovec, T, rhovecL, rhovecV);
        auto N = rhovecL.size();
        auto &derL = *m_derL, &derV = *m_derV; TYPE R = derL.R;
        std::vector<std::vector<TYPE> > J(2 * N + 1, std::vector<TYPE>(2 * N + 1, 0));
        for (std::size_t i = 0; i <= 1; ++i) {
            J[i][0] = derL.d2psir_dTdrhoi__constrhoj(i) - derV.d2psir_dTdrhoi__constrhoj(i) + R*log(rhovecL[i]/rhovecV[i]);
            J[i][1] = (derL.d2psir_drhoidrhoj__constT(i, 0) + ((i == 0) ? R*T / rhovecL[0] : 0))*rhovecL[0];
            J[i][2] = (derL.d2psir_drhoidrhoj__constT(i, 1) + ((i == 1) ? R*T / rhovecL[1] : 0))*rhovecL[1];
            J[i][3] = (-derV.d2psir_drhoidrhoj__constT(i, 0) - ((i == 0) ? R*T / rhovecV[0] : 0))*rhovecV[0];
            J[i][4] = (-derV.d2psir_drhoidrhoj__constT(i, 1) - ((i == 1) ? R*T / rhovecV[1] : 0))*rhovecV[1];
        }
        J[2][0] = derL.dpdT__constrhovec();
        J[2][1] = derL.dpdrhoi__constTrhoj(0)*rhovecL[0];
        J[2][2] = derL.dpdrhoi__constTrhoj(1)*rhovecL[1];
        J[3][0] = derV.dpdT__constrhovec();
        J[3][3] = derV.dpdrhoi__constTrhoj(0)*rhovecV[0];
        J[3][4] = derV.dpdrhoi__constTrhoj(1)*rhovecV[1];
        J[4][1+m_imposed_index] = 1*rhovecL[m_imposed_index];

        //int rr = 0;
        //std::vector<std::vector<TYPE> > Jnum = FuncWrapperND::Jacobian(lnrhovec);
        //std::cout << CoolProp::vec_to_eigen(J) << std::endl << std::endl << CoolProp::vec_to_eigen(Jnum) << std::endl;
        //return Jnum;

        return J;
    }
};

#endif
