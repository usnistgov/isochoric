#ifndef ABSTRACT_TRACER_H
#define ABSTRACT_TRACER_H

#include <chrono>
#include "ODEIntegrators.h"


/// This class holds the data obtained by the tracer
template<typename TYPE = double>
class IsolineTracerData {
public:
    std::vector<TYPE> TL, TV, pL, pV, chempot0L, chempot0V, chempotr0L, chempotr0V, chempotr1L, chempotr1V, det_PSIV, det_PSIL;
    std::vector<std::vector<TYPE> > x, y, rhoL, rhoV;
    std::vector<TYPE> getvec(const std::string &name, std::size_t i) {
        std::vector<TYPE> o;
        for (std::size_t irow = 0; irow < x.size(); ++irow) {
            if (name == "rhoL") { o.push_back(rhoL[irow][i]); }
            else if (name == "rhoV") { o.push_back(rhoV[irow][i]); }
            else if (name == "x") { o.push_back(x[irow][i]); }
            else if (name == "y") { o.push_back(y[irow][i]); }
            else { throw - 1; }
        }
        return o;
    }
};

/// This class holds the data for the initial state defined by temperature and molar concentrations in the co-existing phases
template<typename TYPE = double>
class InitialState
{
public:
    TYPE T;
    Eigen::ArrayXd rhovecL, rhovecV;
    TYPE rhoL() const {
        return rhovecL.sum();
    }
    TYPE rhoV() const {
        return rhovecV.sum();
    }
    std::vector<TYPE> rhovec() const {
        std::vector<TYPE> rhovec(rhovecL.data(), rhovecL.data() + rhovecL.size());
        rhovec.insert(rhovec.end(), rhovecV.data(), rhovecV.data() + rhovecV.size());
        return rhovec;
    }
};

/**
* \brief This class actually implements the isochoric integration
*
* It is derived from the AbstractODEIntegrator of CoolProp which mandates that some functions related
* to ODE integration be implemented
*/
template<typename TYPE = double>
class AbstractIsolineTracer : public ODEIntegrators::AbstractODEIntegrator
{
protected:
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    typedef std::function<std::unique_ptr<const MixDerivs<TYPE> >(const TYPE, const EigenArray&)> DerivFactory;

    TYPE m_T, time_elapsed_sec, rhomolarmax, det_PSIL, det_PSIV, m_allowable_error;
    std::vector< std::vector<TYPE> > m_rhoL, m_rhoV, m_x, m_y;
    std::vector<TYPE> m_p, m_chempot0L, m_chempot0V, m_chempotr0L, m_chempotr0V, m_chempotr1L, m_chempotr1V, m_TL, m_TV, m_pL, m_pV, m_detPSIL, m_detPSIV;
    bool m_disable_polish, m_forwards_integration;
    std::vector<TYPE> Tc, pc;
    InitialState<TYPE> m_initial_state;
    std::string m_termination_reason;
    bool m_unstable_termination = true;
    bool m_debug_polishing = false;
    double m_maximum_pressure = 10e6;
    std::size_t m_max_size = 10000;
    TYPE m_timeout = 6e23; // A very, very large timeout
    std::chrono::high_resolution_clock::time_point startTime;
    int m_premature_termination_code = 0;
    bool m_termination_requested = false;
    int m_min_stepsize_counter = 0;
    double m_c_parametric = 1.0;

    double hmin = 1e-6, hmax = 10000000;

    /// Carry out some common initialization
    void common_init() {
        mode = STEP_INVALID;
        m_allowable_error = -1; // Not specified
        m_disable_polish = false;
        m_forwards_integration = true;
        m_termination_reason = "";
        m_premature_termination_code = 0;
        m_termination_requested = false;
        m_min_stepsize_counter = 0;
        m_c_parametric = 1.0;
    }
    EigenArray drhoLdt_old, drhoVdt_old,  drhovec_dtL_store, drhovec_dtV_store;
public:
    enum stepping_variable { STEP_INVALID = 0, STEP_IN_P, STEP_IN_RHO0, STEP_IN_RHO1, STEP_IN_T, STEP_PARAMETRIC };
    enum imposed_variable_types { IMPOSED_P, IMPOSED_T };
    stepping_variable mode;
    imposed_variable_types imposed_variable;
    TYPE imposed_value;
    TYPE sign_flag = 1.0;
    /// Initializer with a provided state class
    AbstractIsolineTracer(imposed_variable_types var, TYPE imposed_value) : imposed_variable(var), imposed_value(imposed_value) {
        common_init();
    };
    virtual ~AbstractIsolineTracer() {}

    virtual TYPE pure_sat_call(const std::string &out, const std::string &name1, const TYPE val1, const std::string&name2, const TYPE val2, std::size_t index) const = 0;
    virtual const std::vector<std::string> get_names() const = 0;
    virtual TYPE get_gas_constant() const { return 8.3144598; }
    virtual InitialState<TYPE> calc_initial_state() const = 0;
    virtual const std::vector<TYPE> get_Tcvec() const = 0;
    virtual const std::vector<TYPE> get_pcvec() const = 0;
    virtual std::unique_ptr<const MixDerivs<TYPE> > get_derivs(const TYPE T, const EigenArray &rhovec) = 0;
    virtual DerivFactory get_derivs_factory() const = 0;

    // *************************************************************
    // **************** Trivial getters/setters ********************
    // *************************************************************

    /// Get the elapsed time to trace the isoline
    TYPE get_tracing_time() const { return time_elapsed_sec; }

    /// Set the unstable termination flag; if true, if the state is unstable, stop
    void set_unstable_termination(bool unstable_termination) { m_unstable_termination = unstable_termination; }

    /// Get the unstable termination flag; if true, if the state is unstable, stop
    bool get_unstable_termination() const { return m_unstable_termination; }

    /// Get a string representation for why the tracer terminated
    std::string get_termination_reason() const { return m_termination_reason; }

    /// Set the allowable error for the integrator
    void set_allowable_error(TYPE error) { m_allowable_error = error; }

    /// Get the allowable error for the integrator
    TYPE get_allowable_error() const { return m_allowable_error; }

    /// Set the maximum number of steps allowed
    void set_max_size(std::size_t max_size) { m_max_size = max_size; }

    /// Get the maximum number of steps allowed
    std::size_t get_max_size() const { return m_max_size; }

    /// Set the maximum pressure when tracing an isotherm
    void set_maximum_pressure(TYPE maximum_pressure){ m_maximum_pressure = maximum_pressure; }

    /// Get the maximum pressure when tracing an isotherm
    TYPE get_maximum_pressure() { return m_maximum_pressure; }

    /// Get the debug polishing flag; if true, print out to stdout the error message when polishing fails
    bool get_debug_polishing() const { return m_debug_polishing; }

    /// Set the debug polishing flag; if true, print out to stdout the error message when polishing fails
    void set_debug_polishing(bool debug_polishing) { m_debug_polishing = debug_polishing; }

    /// Set the polishing flag; if true, try to polish after each step
    void polishing(bool polish) { m_disable_polish = !polish; }

    /// Get the integration direction flag; if true, integrate in the forwards direction
    bool get_forwards_integration() { return m_forwards_integration; }

    /// Set the integration direction flag; if true, integrate in the forwards direction
    void set_forwards_integration(bool forwards) { m_forwards_integration = forwards; }

    /// Override the default logic for deciding on the stepping variable
    void set_stepping_variable(stepping_variable the_mode){ mode = the_mode; }

    void set_timeout(TYPE timeout){ m_timeout = timeout; }

    TYPE get_timeout() const { return m_timeout; }

    int get_premature_termination_code() const { return m_premature_termination_code; }

    // *********************
    // Initial State Methods
    // *********************

    /// Get the initial array for the integrator
    virtual std::vector<TYPE> get_initial_array() const {
        std::vector<TYPE> rhovec = m_initial_state.rhovec();
        if (imposed_variable == IMPOSED_P && mode == STEP_IN_RHO0) {
            // Do some special stuff here because T is not fixed, rather it is 
            // a dependent variable
            auto x = m_initial_state.rhovecL / m_initial_state.rhovecL.sum(),
                y = m_initial_state.rhovecV / m_initial_state.rhovecV.sum();
            auto N = rhovec.size() / 2;
            rhovec[0] = m_T;
            rhovec[1] = m_initial_state.rhoL()*x[1];
            for (std::size_t i = 0; i < N; ++i) {
                rhovec[i + N] = m_initial_state.rhoV()*y[i]; // vapor
            }
        }
        return rhovec;
    }

    /// Get the initial state for the tracer; can be over-ridden by derived class
    virtual InitialState<TYPE> get_initial_state() const { return m_initial_state; }

    /// Set the initial state for the tracer
    virtual void set_initial_state(const InitialState<TYPE> &state) { m_initial_state = state; }

    /// If initial state is not already polished, do so in this function
    void polish_initial_state(bool force = false) {
        if (!m_disable_polish || force) {
            if (imposed_variable == AbstractIsolineTracer::IMPOSED_P) {
                if (mode == AbstractIsolineTracer::STEP_IN_T) {
                    auto N = m_initial_state.rhovecL.size();
                    Eigen::ArrayXd rhovec(2 * N);
                    rhovec.head(N) = m_initial_state.rhovecL.array();
                    rhovec.tail(N) = m_initial_state.rhovecV.array();
                    std::vector<TYPE> rhovec0(rhovec.data(), rhovec.data() + 2 * N);
                    IsobarVLEResiduals<TYPE> resid(get_derivs_factory(), imposed_value, m_initial_state.T);
                    std::vector<TYPE> rhovecnew = NDNewtonRaphson_Jacobian(&resid, rhovec0, 1e-7, 30);
                    // TODO : Does anything happen here? It seems rhovecnew is not used at all
                }
                else if (mode == AbstractIsolineTracer::STEP_IN_RHO0)
                {
                    std::vector<TYPE> initvals(4 + 1);
                    initvals[0] = m_initial_state.T;
                    initvals[1] = log(m_initial_state.rhovecL[0]);
                    initvals[2] = log(m_initial_state.rhovecL[1]);
                    initvals[3] = log(m_initial_state.rhovecV[0]);
                    initvals[4] = log(m_initial_state.rhovecV[1]);

                    IsobarImposedRho0VLEResiduals<TYPE> resid(get_derivs_factory(), imposed_value, m_initial_state.rhovecL[0], 0);
                    auto outvals = NDNewtonRaphson_Jacobian(&resid, initvals, 1e-7, 30);

                    std::vector<TYPE> rhovec(4);
                    rhovec[0] = m_initial_state.rhovecL[0];
                    rhovec[1] = exp(outvals[2]);
                    rhovec[2] = exp(outvals[3]);
                    rhovec[3] = exp(outvals[4]);

                    m_initial_state.T = outvals[0];
                    m_initial_state.rhovecL = Eigen::Map<Eigen::ArrayXd>(&(rhovec[0]), 2);
                    m_initial_state.rhovecV = Eigen::Map<Eigen::ArrayXd>(&(rhovec[0]) + 2, 2);
                }
            }
        }
    }

    virtual stepping_variable determine_integration_type() {

        // If already specified, override the default logic
        if (this->mode != STEP_INVALID) {
            return this->mode;
        }

        stepping_variable mode;
        if (imposed_variable == IMPOSED_T) {
            m_T = imposed_value;
            if (m_T < *std::min_element(Tc.cbegin(), Tc.cend())) {
                // This is a subcritical isotherm, T is less than both critical temperatures
                mode = AbstractIsolineTracer::STEP_IN_RHO0;
            }
            else {
                mode = AbstractIsolineTracer::STEP_IN_P;
            }
        }
        else if (imposed_variable == IMPOSED_P) {
            if (imposed_value < *std::max_element(pc.cbegin(), pc.cend())) {
                // This is a subcritical isobar, p is less than both critical pressures. Therefore
                // we will step in the concentration of one of the components of the binary. 
                mode = AbstractIsolineTracer::STEP_IN_RHO0;
            }
            else {
                mode = AbstractIsolineTracer::STEP_IN_T;
            }
        }
        else {
            throw ValueError("Invalid imposed variable");
        }
        return mode;
    }
    virtual std::pair<TYPE, TYPE> get_integration_limits() {
        TYPE xmin, xmax;
        auto flds = get_names();
        if (imposed_variable == IMPOSED_T) {
            if (mode == AbstractIsolineTracer::STEP_IN_RHO0) {
                if (m_forwards_integration) {
                    /// At the end of the isotherm, we have arrived at pure component #1 (with index 0)
                    if (m_T > get_Tcvec()[1]) {
                        throw ValueError("Cannot start isotherm because we are marching in RHO0 and temperature is above Tcrit of component");
                    }
                    rhomolarmax = pure_sat_call("Dmolar", "T", m_T, "Q", 0, 0);
                    if (!ValidNumber(rhomolarmax)) { throw ValueError("Cannot start isotherm; temperature out of range?"); }
                    xmin = m_initial_state.rhovecL[0]; xmax = 0.999*rhomolarmax;
                }
                else {
                    // We start off at pure first component, and therefore the concentration of first component
                    // is known. At the end of the tracing, we have arrived at pure second component, so concentration
                    // of first component is almost zero
                    xmax = m_initial_state.rhovecL[0]; xmin = 1e-6;
                    assert(xmax > xmin);
                }
            }
            else if (mode == AbstractIsolineTracer::STEP_IN_RHO1) {
                if (m_forwards_integration) {
                    /// At the end of the isotherm, we have arrived at pure component #2 (with index 1)
                    rhomolarmax = pure_sat_call("Dmolar", "T", m_T, "Q", 0, 1);
                    if (!ValidNumber(rhomolarmax)) { throw ValueError("Cannot start isotherm; temperature out of range?"); }
                    xmin = rhomolarmax; xmax = 1e-6;
                }
                else {
                    // We start off at pure first component, and therefore the concentration of second component
                    // is zero initially. At the end of the tracing, we have arrived at pure second component
                    
                    /// At the end of the isotherm, we have arrived at pure component #2 (with index 1)
                    if (m_T > get_Tcvec()[1]) {
                        throw ValueError("Cannot start isotherm because we are marching in RHO1 and temperature is above Tcrit of component");
                    }
                    rhomolarmax = pure_sat_call("Dmolar", "T", m_T, "Q", 0, 1);
                    if (!ValidNumber(rhomolarmax)) { throw ValueError("Cannot start isotherm; temperature out of range?"); }
                    xmin = m_initial_state.rhovecL[1]; xmax = 0.999*rhomolarmax;
                    assert(xmin < xmax);
                }
            }
            else if (mode == AbstractIsolineTracer::STEP_PARAMETRIC) {
                xmin = 0; xmax = 1e10;
            }
            else if (mode == AbstractIsolineTracer::STEP_IN_P) {
                xmin = 0; xmax = m_maximum_pressure;
            }
            else { throw ValueError("invalid input"); }
        }
        else if (imposed_variable == IMPOSED_P) {
            if (mode == AbstractIsolineTracer::STEP_IN_RHO0) {
                if (m_forwards_integration) {
                    rhomolarmax = pure_sat_call("Dmolar", "P", imposed_value, "Q", 0, 0);
                    if (!ValidNumber(rhomolarmax)) { throw ValueError("Cannot start isobar; pressure out of range?"); }
                    xmin = m_initial_state.rhovecL[0]; xmax = 0.999*rhomolarmax;
                }
                else {
                    // Initialization is already complete, and we have defined the density at the
                    // endpoint from the initialization.  Thus we can use that density, and
                    // then walk backwards to zero(ish).  In the initial state, we are
                    // close to the density of pure component #0
                    xmin = m_initial_state.rhovecL[0]*0.999; xmax = m_initial_state.rhovecL[1];
                    assert(xmin > xmax);
                }
            }
            else if (mode == AbstractIsolineTracer::STEP_IN_T) {
                TYPE Tmax = pure_sat_call("T", "P", imposed_value, "Q", 0, 0);
                if (!ValidNumber(rhomolarmax)) { throw ValueError("Cannot start isobar; pressure out of range?"); }
                xmin = m_T; xmax = Tmax*0.999;
            }
            else { throw ValueError("invalid input"); }
        }
        else { throw ValueError("invalid imposed variable"); }
        return std::make_pair(xmin, xmax);
    };

    // **********************
    // ODE Integrator methods
    // **********************

    /// Return the vector of dpdrho|T along the phase boundary
    std::tuple<EigenArray, EigenArray, EigenMatrix, EigenMatrix> get_drhovecdp_sat(const Eigen::ArrayXd &rhovecL, const Eigen::ArrayXd &rhovecV, const TYPE T)
    {
        auto derL = get_derivs(T, rhovecL), derV = get_derivs(T, rhovecV); 
        Eigen::MatrixXd PSIL = derL->get_Hessian(), PSIV = derV->get_Hessian();

        std::size_t N = rhovecL.size();
        Eigen::MatrixXd A(N, N), r(N, 1);
        TYPE R = derL->R;
        // The right-hand-side is just all ones
        r << 1, 1;

        if (rhovecL.minCoeff() == 0.0 || rhovecV.minCoeff() == 0.0){
            // Special treatment for infinite dilution

            // First, for the liquid part
            for (auto i = 0; i < 2; ++i){
                for (auto j = 0; j < 2; ++j) {
                    if (rhovecL[j] == 0){
                        // Analysis is special if j is the index that is a zero concentration. If you are multiplying by the vector
                        // of liquid concentrations, a different treatment than the case where you multiply by the vector
                        // of vapor concentrations is required
                        // ...
                        // Initial values
                        EigenArray Aij = PSIL.row(j).array().cwiseProduct(((i==0) ? rhovecV : rhovecL).array().transpose()); // coefficient-wise product;
                        if (Aij.size() != 2){ throw ValueError("Must be two"); }
                        // A throwaway boolean for clarity
                        bool is_liq = (i==1);
                        // Apply correction to the j term (RT if liquid, RT*phi for vapor)
                        Aij[j] = (is_liq) ? R*T : R*T*exp(-(derV->dpsir_drhoi__constTrhoj(j) - derL->dpsir_drhoi__constTrhoj(j)) / (R*T));
                        // Fill in entry
                        A(i, j) = Aij.sum();
                    }
                    else{
                        // Normal
                        A(i, j) = PSIL.row(j).dot(((i == 0) ? rhovecV : rhovecL).matrix().transpose());
                    }
                }
            }
            // Calculate the derivatives of the liquid phases
            Eigen::MatrixXd drhovec_dpL = A.fullPivHouseholderQr().solve(r);

            // Then, for the vapor part, also requiring special treatment
            // Left-multiplication of both sides of equation by diagonal matrix with liquid mole fractions along diagonal, all others zero
            Eigen::MatrixXd PSIVstar = rhovecL.matrix().asDiagonal()*PSIV.matrix();
            Eigen::MatrixXd PSILstar = rhovecL.matrix().asDiagonal()*PSIL.matrix();
            for (auto j = 0; j < 2; ++j) {
                if (rhovecL[j] == 0){
                    PSILstar(j, j) = R*T;
                    PSIVstar(j, j) = R*T / exp(-(derV->dpsir_drhoi__constTrhoj(j) - derL->dpsir_drhoi__constTrhoj(j)) / (R*T));
                }
            }

            Eigen::MatrixXd drhovec_dpV = PSIVstar.fullPivHouseholderQr().solve(PSILstar*drhovec_dpL);
            return std::make_tuple(drhovec_dpL, drhovec_dpV, PSILstar, PSIVstar);
        }
        else{
            // "Normal" evaluation, all concentrations are greater than zero
            A(0, 0) = PSIL.row(0).dot(rhovecV.matrix().transpose());
            A(0, 1) = PSIL.row(1).dot(rhovecV.matrix().transpose());
            A(1, 0) = PSIL.row(0).dot(rhovecL.matrix().transpose());
            A(1, 1) = PSIL.row(1).dot(rhovecL.matrix().transpose());
            // Calculate the derivatives of the liquid and vapor phases
            Eigen::MatrixXd drhovec_dpL = A.fullPivHouseholderQr().solve(r);
            Eigen::MatrixXd drhovec_dpV = PSIV.fullPivHouseholderQr().solve(PSIL*drhovec_dpL);
            return std::make_tuple(drhovec_dpL, drhovec_dpV, PSIL, PSIV);
        }
    }

    double determine_c_parametric() {
        if (mode == STEP_PARAMETRIC) {
            std::vector<TYPE> rhovec(m_initial_state.rhovecL.size()*2);
            rhovec[0] = m_initial_state.rhovecL[0];
            rhovec[1] = m_initial_state.rhovecL[1];
            rhovec[2] = m_initial_state.rhovecV[0];
            rhovec[3] = m_initial_state.rhovecV[1];
            std::vector<TYPE> f(rhovec.size());
            derivs(m_T, rhovec, f);
            double dt = 1e-6;
            Eigen::ArrayXd newrho(4);
            for (auto i = 0; i < 4; ++i) {
                newrho[i] = rhovec[i] + dt * f[i];
            }
            if ((newrho < 0).any()) {
                return -1;
            }
            else{
                return 1;
            }
        }
        else {
            return 1.0;
        }
    }

    /// Calculate the system of differential equations being integrated by this class
    void derivs(TYPE t, std::vector<TYPE> &rhovec, std::vector<TYPE> &f) {
        assert(rhovec.size() % 2 == 0);

        std::size_t N = rhovec.size() / 2;
        Eigen::Matrix<TYPE, Eigen::Dynamic, Eigen::Dynamic> PSIL, PSIV;

        std::vector<TYPE> rhoL, rhoV;
        if (imposed_variable == IMPOSED_P) {
            if (mode == STEP_IN_RHO0) {
                // "rhovec" has in it T, rho_1', rho_0'', rho_1''
                // the marching variable (t) is rho_0'
                rhoL.resize(2);
                rhoL[0] = t;
                m_T = rhovec[0];
                rhoL[1] = rhovec[1];
            }
            else if (mode == STEP_IN_T) {
                m_T = t; // The marching variable is the temperature
                rhoL = std::vector<TYPE>(rhovec.begin(), rhovec.begin() + N);
            }
            else {
                throw NotImplementedError();
            }
        }
        else {
            rhoL = std::vector<TYPE>(rhovec.begin(), rhovec.begin() + N);
        }
        rhoV = std::vector<TYPE>(rhovec.begin() + N, rhovec.end());

        Eigen::Map<const Eigen::VectorXd> rhoL_w(&(rhoL[0]), N),
            rhoV_w(&(rhoV[0]), N);

        if (imposed_variable == IMPOSED_T) {
            EigenArray drhovec_dpL, drhovec_dpV;
            
            // Calculate the derivatives of both phases w.r.t. p,
            // including the possibility of infinite dilution
            std::tie(drhovec_dpL, drhovec_dpV, PSIL, PSIV) = get_drhovecdp_sat(rhoL_w, rhoV_w, m_T);
            det_PSIL = PSIL.determinant();
            det_PSIV = PSIV.determinant();

            if (mode == STEP_IN_P) {
                // Our independent variable is p
                Eigen::Map<Eigen::VectorXd> drhovec_dpL_wrap(&(f[0]), N),
                    drhovec_dpV_wrap(&(f[0]) + N, N);
                drhovec_dpL_wrap = drhovec_dpL;
                drhovec_dpV_wrap = drhovec_dpV;
            }
            else if (mode == STEP_IN_RHO0) {
                // Our independent variable is the concentration of the first component in the liquid phase
                Eigen::Map<Eigen::VectorXd> drhovec_drhoL0_wrap(&(f[0]), N),
                    drhovec_drhoV0_wrap(&(f[0]) + N, N);
                drhovec_drhoL0_wrap = drhovec_dpL / drhovec_dpL(0);
                drhovec_drhoV0_wrap = drhovec_dpV / drhovec_dpL(0);
            }
            else if (mode == STEP_IN_RHO1) {
                // Our independent variable is the concentration of the second component in the liquid phase
                Eigen::Map<Eigen::VectorXd> drhovec_drhoL1_wrap(&(f[0]), N),
                    drhovec_drhoV1_wrap(&(f[0]) + N, N);
                drhovec_drhoL1_wrap = drhovec_dpL / drhovec_dpL(1);
                drhovec_drhoV1_wrap = drhovec_dpV / drhovec_dpL(1);
            }
            else if (mode == STEP_PARAMETRIC) {
                // Our independent variable is an analog to the arclength (or so)
                Eigen::Map<Eigen::VectorXd> drhovecL_dt_wrap(&(f[0]), N),
                    drhovecV_dt_wrap(&(f[0]) + N, N);
                auto norm = [](const auto& x) {return pow(x.square().sum(), 0.5); };
                auto dpdt = pow((norm(drhovec_dpL) + norm(drhovec_dpV)), -0.5);
                drhovecL_dt_wrap = m_c_parametric * drhovec_dpL * dpdt;
                drhovecV_dt_wrap = m_c_parametric * drhovec_dpV * dpdt;
            }
            else {
                throw NotImplementedError();
            }
        }
        else if (imposed_variable == IMPOSED_P) {
            
            Eigen::MatrixXd A(N, N);
            Eigen::MatrixXd r(N, 1);

            auto derL = get_derivs(m_T, rhoL_w), derV = get_derivs(m_T, rhoV_w);
            TYPE R = derL->R;
            PSIL = derL->get_Hessian();
            PSIV = derV->get_Hessian();
            det_PSIL = PSIL.determinant();
            det_PSIV = PSIV.determinant();

            A(0, 0) = PSIL.row(0).dot(rhoV_w);
            A(0, 1) = PSIL.row(1).dot(rhoV_w);
            A(1, 0) = PSIL.row(0).dot(rhoL_w);
            A(1, 1) = PSIL.row(1).dot(rhoL_w);

            // Calculate the d(mu)/dT|rho vectors for both phases
            Eigen::VectorXd DELTAdmu_dT(N); // Delta = vapor - liquid
            for (std::size_t i = 0; i < N; ++i) {
                DELTAdmu_dT(i) = derV->d2psir_dTdrhoi__constrhoj(i) - derL->d2psir_dTdrhoi__constrhoj(i) + R*log(rhoV[i] / rhoL[i]);
            }

            r(0) = (DELTAdmu_dT).dot(rhoV_w) - derV->dpdT__constrhovec();
            r(1) = -derL->dpdT__constrhovec();

            // Calculate the derivatives of the liquid phase
            Eigen::VectorXd drhovec_dTL = A.fullPivHouseholderQr().solve(r);

            // Calculate the derivatives of the vapor phase
            Eigen::VectorXd drhovec_dTV = PSIV.fullPivHouseholderQr().solve(PSIL*drhovec_dTL - (DELTAdmu_dT));

            if (mode == STEP_IN_T) {
                // Our independent variable is T
                Eigen::Map<Eigen::VectorXd> drhovec_dTL_wrap(&(f[0]), N),
                    drhovec_dTV_wrap(&(f[0]) + N, N);
                drhovec_dTL_wrap = drhovec_dTL;
                drhovec_dTV_wrap = drhovec_dTV;
            }
            else if (mode == STEP_IN_RHO0) {
                f[0] = 1 / drhovec_dTL(0); // dT/drho_0' @ constant pressure
                f[1] = drhovec_dTL(1) / drhovec_dTL(0);
                f[2] = drhovec_dTV(0) / drhovec_dTL(0);
                f[3] = drhovec_dTV(1) / drhovec_dTL(0);
            }
            else if (mode == STEP_IN_RHO1) {
                f[0] = drhovec_dTL(0) / drhovec_dTL(1); // dT/drho_0' @ constant pressure
                f[1] = 1 / drhovec_dTL(1);
                f[2] = drhovec_dTV(0) / drhovec_dTL(1);
                f[3] = drhovec_dTV(1) / drhovec_dTL(1);
            }
            else {
                throw NotImplementedError();
            }
        }
        else {
            throw NotImplementedError();
        }  
    };

    /// A required method from the ODE integrator class; not implemented
    virtual void pre_step_callback() {};

    /// A required method from the ODE integrator class
    virtual void post_deriv_callback() {
        // Capture the derivatives that were stored at the first call to the derivs function
        this->drhoLdt_old = drhovec_dtL_store;
        this->drhoVdt_old = drhovec_dtV_store;
    };

    /// After a step is taken, do polishing if required and cache values
    virtual void post_step_callback(TYPE t, TYPE h, std::vector<TYPE> &rhovec)
    {
        if (std::abs(h-hmin) < 1e-14*hmin) {
            m_min_stepsize_counter++;
        }
        else{
            m_min_stepsize_counter = 0;
        }
        // Do the polishing if requested
        if (!m_disable_polish) {
            if ((imposed_variable == IMPOSED_T || imposed_variable == IMPOSED_P) && (mode == STEP_IN_RHO0 || mode == STEP_IN_RHO1 || mode == STEP_IN_P || STEP_PARAMETRIC)) {
                // Store the old values
                std::vector<TYPE> rhovec_old = rhovec;
                try {
                    // Polish the solution if possible
                    if (imposed_variable == IMPOSED_T) {
                        std::size_t imposed_index = (mode==STEP_IN_RHO0) ? 0 : 1;
                        IsothermVLEResiduals<TYPE> resid(get_derivs_factory(), imposed_value, rhovec[imposed_index], imposed_index);
                        rhovec = NDNewtonRaphson_Jacobian(&resid, rhovec, 1e-7, 30);
                        const std::vector<TYPE> &errors = resid.get_errors();
                        auto ii = resid.icall;
                        for (auto i = 0; i < errors.size(); ++i) {
                            if (std::abs(errors[i]) > 1e-6) {
                                throw ValueError("polishing did not result in a good solution");
                            }
                            if (std::abs(rhovec[i] - rhovec_old[i]) > 0.05*rhovec_old[i]) {
                                throw ValueError("density changed too much (more than 5%)");
                            }
                            if (rhovec[i] < 0) {
                                throw ValueError("value is negative in polisher; impossible");
                            }
                        }
                        int ok = true;
                    }
                    else if (imposed_variable == IMPOSED_P && mode == STEP_IN_RHO0) {
                        // t is the molar concentration of component 1 in liquid phase
                        IsobarImposedRho0VLEResiduals<TYPE> resid(get_derivs_factory(), imposed_value, t, 0);
                        std::vector<TYPE> initvals(rhovec.size() + 1); initvals[0] = m_T;
                        initvals[1] = log(t);
                        for (std::size_t i = 1; i < rhovec.size(); ++i) { initvals[i + 1] = log(rhovec[i]); }
                        auto outvals = NDNewtonRaphson_Jacobian(&resid, initvals, 1e-7, 30);
                        const std::vector<TYPE> &errors = resid.get_errors();
                        for (auto i = 0; i < errors.size(); ++i) {
                            if (std::abs(errors[i]) > 1e-6) {
                                throw ValueError("polishing did not result in a good solution");
                            }
                            if (std::abs(outvals[i] - initvals[i]) > 0.03*initvals[i]) {
                                throw ValueError("polishing values changed too much (more than 3%)");
                            }
                        }
                        rhovec.resize(rhovec_old.size());
                        rhovec[0] = outvals[0];
                        for (std::size_t i = 1; i < rhovec.size(); ++i) { 
                            rhovec[i] = exp(outvals[i + 1]); 
                        }
                        int ok = true;
                    }
                    else {
                        throw CoolProp::ValueError("Unable to polish for this imposed variable");
                    }
                    int ok = true;
                    if (m_debug_polishing) {
                        std::cout << "Polishing success!" << std::endl;
                    }
                }
                catch (std::exception &e) {
                    // There was a failure, we reset the densities to the old values
                    if (m_debug_polishing) {
                        std::cout << "Polishing failed: " << e.what() << std::endl;
                    }
                    rhovec = rhovec_old;
                }
            }
        }

        // Store variables
        std::size_t N = rhovec.size() / 2;
        std::vector<TYPE> rhovecL, rhovecV;
        if (imposed_variable == IMPOSED_P) {
            if (mode == STEP_IN_RHO0) {
                // "rhovec" has in it T, rho_1', rho_0'', rho_1''
                // the marching variable (t) is rho_0'
                rhovecL.resize(2);
                rhovecL[0] = t;
                m_T = rhovec[0];
                rhovecL[1] = rhovec[1];
            }
            else if (mode == STEP_IN_T) {
                m_T = t; // The marching variable is the temperature
                rhovecL = std::vector<TYPE>(rhovec.begin(), rhovec.begin() + N);
            }
            else {
                throw NotImplementedError();
            }
        }
        else {
            rhovecL = std::vector<TYPE>(rhovec.begin(), rhovec.begin() + N);
        }
        rhovecV = std::vector<TYPE>(rhovec.begin() + N, rhovec.end());
        std::vector<TYPE> x = rhovecL, y = rhovecV;
        Eigen::Map<const Eigen::ArrayXd> rhovecL_w(&rhovecL[0], rhovecL.size()), rhovecV_w(&rhovecV[0], rhovecV.size());

        if (rhovecL_w.minCoeff() < 0) {
            TYPE min_rhoL = *std::min_element(rhovecL.cbegin(), rhovecL.cend()),
                 min_rhoV = *std::min_element(rhovecV.cbegin(), rhovecV.cend());
            // At the next call to premature_termination, the integration will stop, and the values will not get cached
            if (min_rhoL < 0 || min_rhoV < 0) {
                m_termination_requested = true;
                m_termination_reason = "One of min(rhoL)=" + std::to_string(min_rhoL) + " or min(rhoV)=" + std::to_string(min_rhoV) + " is less than 0";
                m_premature_termination_code = 104;
                return;
            }
        }

        auto derL = get_derivs(m_T, rhovecL_w), derV = get_derivs(m_T, rhovecV_w);
        m_p.push_back(derL->p());
        TYPE rhoL = std::accumulate(rhovecL.begin(), rhovecL.end(), 0.0);
        TYPE rhoV = std::accumulate(rhovecV.begin(), rhovecV.end(), 0.0);
        for (std::size_t i = 0; i < 2; ++i) {
            x[i] /= rhoL;
            y[i] /= rhoV;
        }
        m_x.push_back(x);
        m_y.push_back(y);
        m_rhoL.push_back(rhovecL);
        m_rhoV.push_back(rhovecV);
        m_TL.push_back(m_T);
        m_TV.push_back(m_T);
        m_pL.push_back(derL->p());
        m_pV.push_back(derV->p());
        m_chempot0L.push_back(derL->dpsi_drhoi__constTrhoj(0));
        m_chempot0V.push_back(derV->dpsi_drhoi__constTrhoj(0));
        m_chempotr0L.push_back(derL->dpsir_drhoi__constTrhoj(0));
        m_chempotr0V.push_back(derV->dpsir_drhoi__constTrhoj(0));
        m_chempotr1L.push_back(derL->dpsir_drhoi__constTrhoj(1));
        m_chempotr1V.push_back(derV->dpsir_drhoi__constTrhoj(1));
        m_detPSIL.push_back(det_PSIL);
        m_detPSIV.push_back(det_PSIV);
        //printf("%d %10.8f %10.8f %g %g %g %g %g\n", static_cast<int>(m_p.size()), rhovecL[0], rhovecV[0], x[0], y[0], derL->p(), det_PSIL, det_PSIV);
    };
    virtual bool premature_termination() {
        if (m_termination_requested) {
            return true;
        }
        if (get_unstable_termination()) {
            if (mode == STEP_IN_RHO0 && (det_PSIL < 1e-12 || det_PSIV < 1e-12)) {
                m_termination_reason = "One of det(PSIL)=" + std::to_string(det_PSIL) + " or det(PSIV)=" + std::to_string(det_PSIV) + " is less than 1e-12";
                m_premature_termination_code = 101;
                return true;
            }
            if (mode == STEP_IN_P && (det_PSIL < 1e-12 || det_PSIV < 1e-12)) {
                m_termination_reason = "One of det(PSIL)=" + std::to_string(det_PSIL) + " or det(PSIV)=" + std::to_string(det_PSIV) + " is less than 1e-12";
                m_premature_termination_code = 101;
                return true;
            }
        }
        
        const std::size_t max_size = get_max_size();
        if (m_x.size() > max_size) {
            m_termination_reason = std::to_string(max_size) + " steps have been taken";
            m_premature_termination_code = 102;
            return true;
        }
        // Check the timeout
        auto toc = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(toc - startTime).count();
        if (elapsed > get_timeout()) {
            m_termination_reason = "Max allowed time ["  + std::to_string(get_timeout()) + " s] exceeded";
            m_premature_termination_code = 103;
            return true;
        }
        // Check whether the phases are very similar
        if (!m_rhoL.empty() && (std::abs(m_rhoL.back()[0]- m_rhoV.back()[0]) < 1e-3 && std::abs(m_rhoL.back()[1] - m_rhoV.back()[1]) < 1e-3)) {
            m_termination_reason = "Molar concentrations of phases have converged; likely approaching critical point";
            m_premature_termination_code = 104;
            return true;
        }
        // Check if the pressure has stopped changing
        if (m_pL.size() > 10 && imposed_variable == IMPOSED_T) {
            Eigen::Map<const Eigen::ArrayXd> last5p(&(m_pL[m_pL.size()-1])-6, 5);
            // If pressure is very near zero, better to divide by 1 than the value itself
            TYPE denom = std::min(1.0, last5p.cwiseAbs().maxCoeff());
            if (std::abs(last5p.minCoeff() - last5p.maxCoeff()) / denom < 1e-6) {
                m_termination_reason = "Pressure has converged; likely approaching critical point";
                m_premature_termination_code = 105;
                return true;
            }
        }
        // Check if the stepsize is stuck at the minimum value
        if (m_min_stepsize_counter > 10) {
            m_termination_reason = std::to_string(m_min_stepsize_counter) + " steps in a row of minimum size";
            m_premature_termination_code = 106;
            return true;
        }
        // Check if the mole fraction of the first component in the gas phase has stopped changing
        if (m_x.size() > 10) {
            int Nlast = 10;
            Eigen::ArrayXd lasty(Nlast);
            for (auto i = 0; i < Nlast; ++i) {
                lasty(i) = m_y[m_y.size()-1-i][0];
            }
            if (std::abs(lasty.minCoeff() - lasty.maxCoeff()) < 1e-7) {
                m_termination_reason = "Mole fraction has converged within one part in 10^7; likely approaching critical point";
                m_premature_termination_code = 107;
                return true;
            }
        }
        return false;
    };

    // **************************
    // Outputs of tracing results
    // **************************

    // Store a csv with the tracing information (no longer used, but should still work)
    void dump_csv() {
        FILE *fp = fopen("data.csv", "w");
        fprintf(fp, "T,rhoL0,rhoL1,rhoV0,rhoV1,x0,y0,p\n");
        for (std::size_t i = 0; i < m_p.size(); ++i) {
            fprintf(fp, "%g,%g,%g,%g,%g,%g,%g,%g\n", m_T, m_rhoL[i][0], m_rhoL[i][1], m_rhoV[i][0], m_rhoV[i][1], m_x[i][0], m_y[i][0], m_p[i]);
        }
        fclose(fp);
    }

    // Store tracing data in output struct
    IsolineTracerData<TYPE> get_tracer_data() {
        IsolineTracerData<TYPE> out;
        out.pL = m_pL;
        out.pV = m_pV;
        out.TL = m_TL;
        out.TV = m_TV;
        out.x = m_x;
        out.y = m_y;
        out.rhoL = m_rhoL;
        out.rhoV = m_rhoV;
        out.chempot0L = m_chempot0L;
        out.chempot0V = m_chempot0V;
        out.chempotr0L = m_chempotr0L;
        out.chempotr0V = m_chempotr0V;
        out.chempotr1L = m_chempotr1L;
        out.chempotr1V = m_chempotr1V;
        out.det_PSIL = m_detPSIL;
        out.det_PSIV = m_detPSIV;
        return out;
    }

    /// Actually carry out the integration!
    void trace() {
        det_PSIL = det_PSIV = 999;
        startTime = std::chrono::high_resolution_clock::now();

        // Store the initial state; This is implemented via the set/calc functions
        // so that they can be overloaded via pybind11 and defined in python
        try {
            set_initial_state(calc_initial_state());
            // Polish the initial state provided
            polish_initial_state();
        }
        catch (std::exception &e) {
            std::string message;
            if (imposed_variable == IMPOSED_T) {
                message = "isotherm @ T = " +std::to_string(imposed_value)+ " K";
            }
            else if (imposed_variable == IMPOSED_P){
                message = "isobar @ p = " + std::to_string(imposed_value) + " Pa";
            }
            else {
                throw ValueError("Invalid imposed variable");
            }
            throw ValueError("Cannot start " + message + "; error was: " + std::string(e.what()));
        }

        // Get the temperature from the initial state
        m_T = m_initial_state.T;

        // Figure out what kind of integration we are going to carry out
        mode = determine_integration_type();

        // Determine the tracing direction if parametric tracing
        m_c_parametric = determine_c_parametric();

        // Get the specification of the ODE integration
        auto limits = get_integration_limits();
        TYPE xstart = limits.first, xend = limits.second;
        // If allowable_error has not been already specified, the allowable error will be -1,
        // and a default value is used.
        TYPE step_epsilon = (get_allowable_error() > 0) ? get_allowable_error() : 1e-8;

        // Carry out the integration
        
        bool aborted = ODEIntegrators::AdaptiveRK54(*this, xstart, xend, hmin, hmax, step_epsilon, 1.0);

        // If it did not prematurely terminate, store a success message
        if (!aborted){
            m_termination_reason = "Successfully integrated from " + std::to_string(xstart) + " to " + std::to_string(xend);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        time_elapsed_sec = std::chrono::duration<TYPE>(endTime - startTime).count();
    }
};

#endif