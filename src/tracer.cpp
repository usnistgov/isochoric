#include "isochoric/coolprop_tracer.h"

#if defined(PYBIND11)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Prototype for initialization of module with CoolProp objects
void init_CoolProp(py::module &m);

// Prototype for initialization of module with REFPROP objects
void init_REFPROPisochoricthermo(py::module &m);

PYBIND11_MODULE(VLEIsoTracer, m) {
    m.doc() = "Module for tracing vapor-liquid equilibrium isotherms, isobars, and isopleths";

    // Populate module with the CoolProp objects
    init_CoolProp(m);

    py::class_<MixDerivs<double> >(m, "MixDerivs")
        .def("p", &MixDerivs<double>::p)
        .def("get_Hessian", &MixDerivs<double>::get_Hessian)
        .def("dpsir_drhoi__constTrhoj", &MixDerivs<double>::dpsir_drhoi__constTrhoj)
        .def("dpsi_drhoi__constTrhoj", &MixDerivs<double>::dpsi_drhoi__constTrhoj)
        .def("dpdT__constrhovec", &MixDerivs<double>::dpdT__constrhovec)
        .def("d2psi_dTdrhoi__constrhoj", &MixDerivs<double>::d2psi_dTdrhoi__constrhoj)
        .def("d2psir_dTdrhoi__constrhoj", &MixDerivs<double>::d2psir_dTdrhoi__constrhoj)
        .def("dtau_drhoi__constTrhoj", &MixDerivs<double>::dtau_drhoi__constTrhoj)
        .def("ddelta_drhoi__constTrhoj", &MixDerivs<double>::ddelta_drhoi__constTrhoj)
        .def("d2tau_drhoidrhoj__constT", &MixDerivs<double>::d2tau_drhoidrhoj__constT)
        .def("d2delta_drhoidrhoj__constT", &MixDerivs<double>::d2delta_drhoidrhoj__constT)
        .def("dTr_drhoi__constrhoj", &MixDerivs<double>::dTr_drhoi__constrhoj)
        .def("dTr_dxi__constxj", &MixDerivs<double>::dTr_dxi__constxj)
        .def("drhor_drhoi__constrhoj", &MixDerivs<double>::drhor_drhoi__constrhoj)
        .def("drhorTr_dxi__xj", &MixDerivs<double>::drhorTr_dxi__xj)
        .def("drhor_dxi__constxj", &MixDerivs<double>::drhor_dxi__constxj)
        .def("d2Tr_drhoidrhoj", &MixDerivs<double>::d2Tr_drhoidrhoj)
        .def("d2rhor_drhoidrhoj", &MixDerivs<double>::d2rhor_drhoidrhoj)
        .def("d_dpsir_ddelta_drhoi__constrhoj", &MixDerivs<double>::d_dpsir_ddelta_drhoi__constrhoj)
        .def("d_dpsir_dtau_drhoi__constrhoj", &MixDerivs<double>::d_dpsir_dtau_drhoi__constrhoj)
        .def("d_dpsir_dxm_drhoi__constTrhoi", &MixDerivs<double>::d_dpsir_dxm_drhoi__constTrhoi)
        .def_readonly("T", &MixDerivs<double>::T)
        .def_readonly("R", &MixDerivs<double>::R)
        .def_readonly("Tr", &MixDerivs<double>::Tr)
        .def_readonly("rhor", &MixDerivs<double>::rhor)
        ;
    
    py::class_<AbstractIsolineTracer<double> > tracer(m, "AbstractIsolineTracer");
        tracer.def("trace", &AbstractIsolineTracer<double>::trace)
        .def("get_tracer_data", &AbstractIsolineTracer<double>::get_tracer_data)
        .def("set_allowable_error", &AbstractIsolineTracer<double>::set_allowable_error)
        .def("get_allowable_error", &AbstractIsolineTracer<double>::get_allowable_error)
        .def("get_maximum_pressure", &AbstractIsolineTracer<double>::get_maximum_pressure)
        .def("set_maximum_pressure", &AbstractIsolineTracer<double>::set_maximum_pressure)
        .def("get_termination_reason", &AbstractIsolineTracer<double>::get_termination_reason)
        .def("get_unstable_termination", &AbstractIsolineTracer<double>::get_unstable_termination)
        .def("get_integration_limits", &AbstractIsolineTracer<double>::get_integration_limits)
        .def("set_unstable_termination", &AbstractIsolineTracer<double>::set_unstable_termination)
        .def("set_stepping_variable", &AbstractIsolineTracer<double>::set_stepping_variable)
        .def("get_debug_polishing", &AbstractIsolineTracer<double>::get_debug_polishing)
        .def("set_debug_polishing", &AbstractIsolineTracer<double>::set_debug_polishing)
        .def("get_c_parametric", &AbstractIsolineTracer<double>::get_c_parametric)
        .def("set_c_parametric", &AbstractIsolineTracer<double>::set_c_parametric)
        .def("get_tracing_time", &AbstractIsolineTracer<double>::get_tracing_time)
        .def("get_max_size", &AbstractIsolineTracer<double>::get_max_size)
        .def("set_max_size", &AbstractIsolineTracer<double>::set_max_size)
        .def("get_drhovecdp_sat", &AbstractIsolineTracer<double>::get_drhovecdp_sat)
        .def("get_derivs", &AbstractIsolineTracer<double>::get_derivs)
        .def("polishing", &AbstractIsolineTracer<double>::polishing)
        .def("calc_initial_state", &AbstractIsolineTracer<double>::calc_initial_state)
        .def("polish_initial_state", &AbstractIsolineTracer<double>::polish_initial_state)
        .def("get_premature_termination_code", &AbstractIsolineTracer<double>::get_premature_termination_code)
        .def("set_forwards_integration", &AbstractIsolineTracer<double>::set_forwards_integration);

    py::class_<VLEIsolineTracer<double>, AbstractIsolineTracer<double>, PyVLEIsolineTracer<double> >(m, "VLEIsolineTracer")
        .def(py::init<AbstractIsolineTracer<double>::imposed_variable_types, double, const std::string&, const std::vector<std::string> &>())
        .def("set_binary_interaction_double", &VLEIsolineTracer<double>::set_binary_interaction_double)
        .def("get_AbstractState_pointer", &VLEIsolineTracer<double>::get_AbstractState_pointer, py::return_value_policy::reference)
        ;

    py::class_<AlgebraicIsolineTracer<double>>(m, "AlgebraicIsolineTracer")
        .def(py::init<AbstractIsolineTracer<double>::imposed_variable_types, double, const std::string&, const std::vector<std::string> &>())
        .def_readwrite("p", &AlgebraicIsolineTracer<double>::logger_p)
        .def_readwrite("x", &AlgebraicIsolineTracer<double>::logger_x)
        .def_readwrite("y", &AlgebraicIsolineTracer<double>::logger_y)
        .def_readwrite("logging", &AlgebraicIsolineTracer<double>::logging)
        .def("trace_rho0", &AlgebraicIsolineTracer<double>::trace_rho0)
        ;

    py::enum_<AbstractIsolineTracer<double>::imposed_variable_types>(tracer, "imposed_variable")
        .value("IMPOSED_T", AbstractIsolineTracer<double>::imposed_variable_types::IMPOSED_T)
        .value("IMPOSED_P", AbstractIsolineTracer<double>::imposed_variable_types::IMPOSED_P)
        ;

    py::enum_<AbstractIsolineTracer<double>::stepping_variable >(tracer, "stepping_variable")
        .value("STEP_IN_RHO0", AbstractIsolineTracer<double>::stepping_variable::STEP_IN_RHO0)
        .value("STEP_IN_RHO1", AbstractIsolineTracer<double>::stepping_variable::STEP_IN_RHO1)
        .value("STEP_IN_P", AbstractIsolineTracer<double>::stepping_variable::STEP_IN_P)
        .value("STEP_IN_T", AbstractIsolineTracer<double>::stepping_variable::STEP_IN_T)
        .value("STEP_PARAMETRIC", AbstractIsolineTracer<double>::stepping_variable::STEP_PARAMETRIC)
        ;

    py::class_<IsolineTracerData<double>>(m, "IsolineTracerData")
        .def_readonly("pL", &IsolineTracerData<double>::pL)
        .def_readonly("pV", &IsolineTracerData<double>::pV)
        .def_readonly("TL", &IsolineTracerData<double>::TL)
        .def_readonly("TV", &IsolineTracerData<double>::TV)
        .def_readonly("x", &IsolineTracerData<double>::x)
        .def_readonly("y", &IsolineTracerData<double>::y)
        .def_readonly("rhoL", &IsolineTracerData<double>::rhoL)
        .def_readonly("rhoV", &IsolineTracerData<double>::rhoV)
        .def_readonly("chempot0L", &IsolineTracerData<double>::chempot0L)
        .def_readonly("chempot0V", &IsolineTracerData<double>::chempot0V)
        .def_readonly("chempotr0L", &IsolineTracerData<double>::chempotr0L)
        .def_readonly("chempotr0V", &IsolineTracerData<double>::chempotr0V)
        .def_readonly("chempotr1L", &IsolineTracerData<double>::chempotr1L)
        .def_readonly("chempotr1V", &IsolineTracerData<double>::chempotr1V)
        .def_readonly("det_PSIL", &IsolineTracerData<double>::det_PSIL)
        .def_readonly("det_PSIV", &IsolineTracerData<double>::det_PSIV)
        ;

    py::class_<InitialState<double> >(m, "InitialState")
        .def(py::init<>())
        .def_readwrite("T", &InitialState<double>::T)
        .def_readwrite("rhovecL", &InitialState<double>::rhovecL)
        .def_readwrite("rhovecV", &InitialState<double>::rhovecV)
        ;

    #ifdef REFPROP_SUPPORT
    init_REFPROPisochoricthermo(m);
    #endif
}

#else

template<class TRACER>
void do_trace()
{
    TRACER IT0(TRACER::IMPOSED_T, 260, "HEOS", strsplit("CO2&ethane", '&'));
    IT0.polishing(true);
    //IT0.set_allowable_error(1e-5);
    IT0.set_forwards_integration(true);
    IT0.trace();
    IsolineTracerData<> data0 = IT0.get_tracer_data();
    std::cout << IT0.get_tracing_time() << " s; N: " << data0.pL.size() << " p[-1]: " << data0.pL.back() << std::endl;
}

void just_do_one(){
    VLEIsolineTracer<> IT0(VLEIsolineTracer<>::IMPOSED_T, 230, "PR", strsplit("CO2&ethane", '&'));
    IT0.polishing(true);
    //IT0.set_allowable_error(1e-5);
    IT0.set_forwards_integration(false);
    IT0.trace();
    IsolineTracerData<> data0 = IT0.get_tracer_data();
    std::cout << IT0.get_tracing_time() << " s; N: " << data0.pL.size() << " p[-1]: " << data0.pL.back() << std::endl;
}

int main() {
    apply_simple_mixing_rule("SO2", "N2", "linear"); // placeholder only
    apply_simple_mixing_rule("SO2", "Water", "linear"); // placeholder only
    //std::vector<std::string> keys = { "betaT","gammaT","betaV","gammaV" };
    //std::vector<double> vals = { 1.019562, 0.916311, 1.094032, 0.962547 };
    VLEIsolineTracer<> tracer(VLEIsolineTracer<>::IMPOSED_P, 5455594.781168515, "HEOS", strsplit("SO2&N2", '&'));
    tracer.polishing(true);
    tracer.set_forwards_integration(false);
    tracer.trace();
    //tracer.set_stepping_variable(VLEIsolineTracer<>::stepping_variable::STEP_IN_RHO1);
    auto data = tracer.get_tracer_data();
    std::string reason = tracer.get_termination_reason();

    double T = 300, dT = 1e-3;
    Eigen::ArrayXd rhovec(2); rhovec.fill(300);
    double base = tracer.get_derivs(T, rhovec)->dpdT__constrhovec();
    double deriv = (tracer.get_derivs(T+dT, rhovec)->p() - tracer.get_derivs(T-dT, rhovec)->p())/(2*dT);

    just_do_one();
    do_trace<VLEIsolineTracer<>>();
    
    return EXIT_SUCCESS;

    /*std::vector<double> rhoL = data0.rhoL[0], rhoV = data0.rhoV[0];
    double rhostart = rhoL[0], rhoend = data0.rhoL.back()[0]; for (auto &&el : rhoV){ rhoL.push_back(el); }
    AlgebraicIsolineTracer IT3(VLEIsolineTracer::IMPOSED_T, 280, "HEOS", strsplit("CO2&Ethane", '&'));
    IT3.logging = true;
    IT3.trace_rho0(rhoL, linspace(rhostart, rhoend, 100));

    return EXIT_SUCCESS;*/
}
#endif
