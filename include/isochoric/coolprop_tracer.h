#ifndef COOLPROP_TRACER_H
#define COOLPROP_TRACER_H

#include "isochoric/residual_functions.h"
#include "isochoric/mixderiv.h"
#include "isochoric/providers.h"
#include "isochoric/abstract_tracer.h"

#include "Backends/Helmholtz/MixtureParameters.h"

#include "ODEIntegrators.h"

//#include <atomic>
//static std::atomic_size_t derivs_calls{0};

template<typename TYPE = double>
class VLEIsolineTracer : public AbstractIsolineTracer<> {
private:
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef std::function<std::unique_ptr<const MixDerivs<TYPE> >(const TYPE, const EigenArray&)> DerivFactory;
    std::unique_ptr<CoolProp::HelmholtzEOSMixtureBackend> HEOS;
public:
    /// Initializer with a provided backend/fluid pair
    VLEIsolineTracer(imposed_variable_types var, double imposed_value, const std::string &backend, const std::vector<std::string> &fluids) : AbstractIsolineTracer<TYPE>(var, imposed_value) {
        HEOS.reset(static_cast<HelmholtzEOSMixtureBackend*>(CoolProp::AbstractState::factory(backend, fluids)));
        Tc = get_Tcvec(); pc = get_pcvec();
    };

    /// Get the idempotent derivative provider structure
    std::unique_ptr<const MixDerivs<TYPE> > get_derivs(const double T, const EigenArray &rhovec) override {
        //derivs_calls++;
        std::unique_ptr<const AbstractNativeDerivProvider<TYPE> > HD(new CoolPropNativeDerivProvider(*HEOS, T, rhovec));
        std::unique_ptr<const MixDerivs<TYPE> > ptr(new MixDerivs<TYPE>(std::move(HD)));
        return ptr;
    }

    /// Get the idempotent derivative provider structure
    DerivFactory get_derivs_factory() const override {
        return [this](const double T, const Eigen::ArrayXd &rhovec){
            return const_cast<VLEIsolineTracer*>(this)->get_derivs(T,rhovec); 
        };
    }

    double pure_sat_call(const std::string &out, const std::string &name1, const double val1, const std::string&name2, const double val2, std::size_t index) const override{
        return PropsSI(out, name1, val1, name2, val2, get_names()[index]);
    }
    
    /// Get a pointer to the AbstractState being managed by this class, for instance in order to change binary interaction parameters
    AbstractState * get_AbstractState_pointer() { return static_cast<CoolProp::AbstractState*>(HEOS.get()); }

    /// Set the binary interaction parameters of a managed state instance
    void set_binary_interaction_double(int i, int j, const std::string &parameter, double value) {
        HEOS->set_binary_interaction_double(i, j, parameter, value);
    }

    /// Get the names of the fluids
    const std::vector<std::string> get_names() const override {
        return HEOS->fluid_names();
    }
    const std::vector<double> get_Tcvec() const override {
        auto names = get_names();
        std::vector<double> Tcvec; 
        for (auto name : names) {
            Tcvec.push_back(Props1SI("Tcrit", name));
        }
        return Tcvec;
    }
    const std::vector<double> get_pcvec() const override {
        auto names = get_names();
        std::vector<double> pcvec;
        for (auto name : names) {
            pcvec.push_back(Props1SI("pcrit", name));
        }
        return pcvec;
    }

    /// Actually calculate the initial state; can be over-ridden by derived class
    InitialState<TYPE> calc_initial_state() const override{

        InitialState<TYPE> state;
        auto names = get_names();
        auto fld0 = names[0], fld1 = names[1];

        std::size_t N = HEOS->fluid_names().size();
        std::vector<double> rhovec(2 * N);

        // Use guesses from ancillaries (because it's fast and accurate enough)
        GuessesStructure g;
        // For forward integration, we start at almost pure component #1
        // For backwards integration, we start at almost pure component #0
        auto fld = (m_forwards_integration) ? fld1 : fld0; // fluid name for the pure fluid we are starting near
        std::vector<double> rho(4, 0); 
        std::size_t imposed_index = (m_forwards_integration) ? 0 : 1;
        if (imposed_variable == IMPOSED_T) {
            state.T = imposed_value;
            std::string backend_string = (HEOS->backend_name() == "PengRobinsonBackend") ? "PR" : "HEOS";
            g.rhomolar_liq = PropsSI("Dmolar", "Q", 0, "T", imposed_value, backend_string + "::" + fld);
            if (!std::isfinite(g.rhomolar_liq)) {
                throw ValueError(get_global_param_string("errstring"));
            }
            g.rhomolar_vap = PropsSI("Dmolar", "Q", 1, "T", imposed_value, backend_string + "::" + fld);
            if (!std::isfinite(g.rhomolar_vap)) {
                throw ValueError(get_global_param_string("errstring"));
            }
            
            rho[0+imposed_index] = 0;
            rho[1-imposed_index] = g.rhomolar_liq;
            rho[2+imposed_index] = 0;
            rho[3-imposed_index] = g.rhomolar_vap;

        }
        else if (imposed_variable == IMPOSED_P) {
            // Invert ancillary to obtain saturation temperature
            g.T = saturation_ancillary(fld, "T", 0, "P", imposed_value);
            g.rhomolar_liq = saturation_ancillary(fld, "Dmolar", 0, "T", g.T);
            g.rhomolar_vap = saturation_ancillary(fld, "Dmolar", 1, "T", g.T);
            state.T = g.T;
            // Start off at basically pure of component #1, with just a little bit of component #0
            rho[imposed_index] = 1e-6; // x[0]*rho' = rho[0]
            rho[1 - imposed_index] = (g.rhomolar_liq - rho[0]);
            rho[2] = rho[0] / g.rhomolar_liq*g.rhomolar_vap;
            rho[3] = rho[1] / g.rhomolar_liq*g.rhomolar_vap;
        }

        if (imposed_variable == IMPOSED_T) {
            rhovec = rho;
        }
        else if (imposed_variable == IMPOSED_P) {
            IsobarImposedRho0VLEResiduals<TYPE> resid(get_derivs_factory(), imposed_value, rho[imposed_index], imposed_index);
            std::vector<double> initvals(rho.size() + 1); initvals[0] = g.T;
            for (std::size_t i = 0; i < rho.size(); ++i) { initvals[i + 1] = log(rho[i]); }
            auto outvals = NDNewtonRaphson_Jacobian(&resid, initvals, 1e-7, 30);
            state.T = outvals[0];
            std::vector<double> lnrhovec = std::vector<double>(outvals.begin() + 1, outvals.end());
            rhovec.resize(lnrhovec.size());
            for (std::size_t i = 0; i < rhovec.size(); ++i) { rhovec[i] = exp(lnrhovec[i]); }
        }
        std::vector<double> rhovecL(rhovec.begin(), rhovec.begin() + N), rhovecV(rhovec.begin() + N, rhovec.end());
        state.rhovecL = Eigen::Map<Eigen::ArrayXd>(&rhovecL[0], rhovecL.size());
        state.rhovecV = Eigen::Map<Eigen::ArrayXd>(&rhovecV[0], rhovecV.size());
        return state;
    };  
};

/*
\brief This class implements the algebraic tracer for an isoline

It marches through the given values of molar concentration, and tries to carry out a phase equilibrium calculation at each molar concentration
*/
template<typename TYPE>
class AlgebraicIsolineTracer {
    typedef Eigen::Array<TYPE, Eigen::Dynamic, 1> EigenArray;
    typedef Eigen::Array<TYPE, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    typedef std::function<std::unique_ptr<const MixDerivs<TYPE> >(const TYPE, const EigenArray&)> DerivFactory;

public:
    typename VLEIsolineTracer<TYPE>::imposed_variable_types m_imposed_variable;
    TYPE imposed_value;
    std::vector<TYPE> m_rhovec;
    shared_ptr<HelmholtzEOSMixtureBackend> HEOS;
    bool logging = false;
    std::vector<TYPE> logger_p, logger_x, logger_y, logger_rhovecL, logger_rhovecV;

    AlgebraicIsolineTracer(typename VLEIsolineTracer<TYPE>::imposed_variable_types imposed_variable, TYPE val, const std::string &backend, const std::vector<std::string> &fluids)
    {
        imposed_value = val;
        HEOS = shared_ptr<HelmholtzEOSMixtureBackend>(dynamic_cast<HelmholtzEOSMixtureBackend*>(CoolProp::AbstractState::factory(backend, fluids)));
    }

    /// Get the idempotent derivative provider structure
    std::unique_ptr<const MixDerivs<TYPE> > get_derivs(const TYPE T, const Eigen::ArrayXd &rhovec) {
        std::unique_ptr<const AbstractNativeDerivProvider<TYPE> > HD(new CoolPropNativeDerivProvider(*HEOS, T, rhovec));
        return std::make_unique<const MixDerivs<TYPE> >(std::move(HD));
    }

    void trace_rho0(const std::vector<TYPE> &rhovec, const std::vector<TYPE> & rho0values) {
        m_rhovec = rhovec;

        for (auto &&rho0 : rho0values) {
            m_rhovec[0] = rho0; 
            DerivFactory factory = [this](const TYPE T, const Eigen::ArrayXd &rhovec) {return this->get_derivs(T, rhovec); };
            IsothermVLEResiduals<TYPE> resid(factory, imposed_value, m_rhovec[0], 0);
            m_rhovec = NDNewtonRaphson_Jacobian(&resid, m_rhovec, 1e-7, 30);

            // Store values
            if (logging){
                std::size_t N = m_rhovec.size()/2;
                logger_rhovecL = std::vector<TYPE>(m_rhovec.begin(), m_rhovec.begin() + N);
                logger_rhovecV = std::vector<TYPE>(m_rhovec.begin() + N, m_rhovec.end());
                std::vector<TYPE> x = logger_rhovecL, y = logger_rhovecV;
                Eigen::ArrayXd rhovecL = Eigen::Map<const Eigen::ArrayXd>(&(logger_rhovecL[0]), logger_rhovecL.size());
                auto der = factory(imposed_value, rhovecL);
                logger_p.push_back(der->p());
                TYPE rhoL = std::accumulate(logger_rhovecL.begin(), logger_rhovecL.end(), 0.0);
                TYPE rhoV = std::accumulate(logger_rhovecV.begin(), logger_rhovecV.end(), 0.0);
                logger_x.push_back(logger_rhovecL[0]/rhoL);
                logger_y.push_back(logger_rhovecV[0]/rhoV);
            }
        }
    }
};

#if defined(PYBIND11)

#include <pybind11/pybind11.h>
namespace py = pybind11;

/* Trampoline class; passes calls up into the derived class in python
 * One overload is needed for each virtual function you want to be able to overload
 */
template<typename TYPE = double>
class PyVLEIsolineTracer : public VLEIsolineTracer<> {
public:
    /* Inherit the constructors */
    using VLEIsolineTracer<>::VLEIsolineTracer;

    std::pair<TYPE, TYPE> get_integration_limits() override {
        // Release the GIL
        py::gil_scoped_release release;
        {
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;

            // Parameters to macro are: Return type, parent class, name of function in C++, argument(s)
            typedef std::pair<TYPE, TYPE> pairdoubledouble;
            PYBIND11_OVERLOAD(pairdoubledouble, 
                              VLEIsolineTracer, 
                              get_integration_limits, 
                              // No arguments
                              );
        }
    }
    InitialState<TYPE> calc_initial_state() const override {
        // Release the GIL
        py::gil_scoped_release release;
        {
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;

            // Parameters to macro are: Return type, parent class, name of function in C++, argument(s)
            PYBIND11_OVERLOAD(InitialState<TYPE>,
                VLEIsolineTracer,
                calc_initial_state,
                // No arguments
                );
        }
    }
    stepping_variable determine_integration_type() override{
        // Release the GIL
        py::gil_scoped_release release;
        {
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;

            // Parameters to macro are: Return type, parent class, name of function in C++, argument(s)
            PYBIND11_OVERLOAD(stepping_variable,
                VLEIsolineTracer,
                determine_integration_type,
                // No arguments
                );
        }
    }
};

#endif

#endif