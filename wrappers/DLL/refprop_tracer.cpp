#include "nlohmann/json.hpp"


#define REFPROP_LIB_NAMESPACE REFPROP_lib
#include "REFPROP_lib.h"
#undef REFPROP_LIB_NAMESPACE

#include "isochoric/residual_functions.h"
#include "isochoric/mixderiv.h"
#include "isochoric/abstract_tracer.h"
#include "isochoric/coolprop_tracer.h"

#include "ISOCHOR.h"
#include "refprop_provider.h"

template<class T>
void HandleException(T &errcode, std::string &message_buffer)
{
    try {
        throw; // Rethrow the error, and here we handle the error
    }
    catch (std::exception &e) {
        message_buffer = std::string("Exception: ") + e.what();
        errcode = 2;
    }
    catch (...) {
        errcode = 3;
    }
}

// In Microsoft Excel, they seem to check the FPU exception bits and error out because of it.  
// By calling the _clearfp(), we can reset these bits, and not get the error
// See also http://stackoverflow.com/questions/11685441/floating-point-error-when-calling-dll-function-from-vba/27336496#27336496
// See also http://stackoverflow.com/questions/16849009/in-linux-do-there-exist-functions-similar-to-clearfp-and-statusfp for linux and OSX
struct fpu_reset_guard {
    ~fpu_reset_guard() {
#if defined(_MSC_VER)
        _clearfp(); // For MSVC, clear the floating point error flags
#elif defined(FE_ALL_EXCEPT)
        feclearexcept(FE_ALL_EXCEPT);
#endif
    }
};

extern bool my_load_REFPROP(std::string &err, const std::string &shared_library_path = "", const std::string &shared_library_name = "");
extern bool my_unload_REFPROP(std::string &err);
extern bool is_hooked();

class REFPROPIsolineTracer : public AbstractIsolineTracer<double> {
public:

    /// Initializer with a provided backend/fluid pair
    REFPROPIsolineTracer(imposed_variable_types var, double imposed_value, const std::string &backend, const std::vector<std::string> &fluids, const std::string &HMX = "HMX.BNC") 
        : AbstractIsolineTracer<>(var, imposed_value) 
    {
        using namespace REFPROP_lib;
        assert(REFPROP_lib::SETUPdll != nullptr);
        int nc = 2, ierr = 0; char hrf[10000], hmx[255], hdef[4] = "DEF", herr[255];
        std::string fldsjoined = strjoin(fluids, "|");
        strcpy(hrf, fldsjoined.c_str());
        strcpy(hmx, HMX.c_str());
        SETUPdll(nc, hrf, hmx, hdef, ierr, herr, 10000, 255, 3, 255);
        if (ierr > 100) {
            throw ValueError(("Could not load fluids with error: "+ std::string(herr)));
        }
        if (PHI0dll == nullptr) {
            throw ValueError("You must use a newer version of REFPROP that includes the PHI0dll function (version 9.1.1+)");
        }
        Tc = get_Tcvec(); 
        pc = get_pcvec();
    };

    std::tuple<double,double> REDX(std::vector<double> &z){
        using namespace REFPROP_lib;
        double Tr=-1,rhor=-1;
        REDXdll(&(z[0]),Tr,rhor);
        return std::make_tuple(Tr,rhor);
    }

    /// Get the names of the fluids
    const std::vector<std::string> get_names() const override {
        using namespace REFPROP_lib;
        std::vector<std::string> names(2);
        for (int icomp = 1; icomp <= 2; ++icomp){
            char hnam[12], hn80[80], hcasn[12];
            NAMEdll(icomp, hnam, hn80, hcasn, 12, 80, 12);
            names[icomp-1] = hnam;
        }
        return names;
    }
    const std::vector<double> get_Tcvec() const override {
        using namespace REFPROP_lib;
        std::vector<double> Tcvec(2);
        for (int icomp = 1; icomp <= 2; ++icomp) {
            double wmm = -1, Ttrp = -1, Tnbpt = -1, Tc = -1, Pc = -1, Dc = -1, Zc = -1, acf = -1, dip = -1, Rgas = -1;
            INFOdll(icomp, wmm, Ttrp, Tnbpt, Tc, Pc, Dc, Zc, acf, dip, Rgas);
            Tcvec[icomp-1]=Tc;
        }
        return Tcvec;
    }
    const std::vector<double> get_pcvec() const override {
        using namespace REFPROP_lib;
        std::vector<double> pcvec(2);
        for (int icomp = 1; icomp <= 2; ++icomp) {
            double wmm = -1, Ttrp = -1, Tnbpt = -1, Tc = -1, Pc = -1, Dc = -1, Zc = -1, acf = -1, dip = -1, Rgas = -1;
            INFOdll(icomp, wmm, Ttrp, Tnbpt, Tc, Pc, Dc, Zc, acf, dip, Rgas);
            pcvec[icomp - 1] = Pc*1000;
        }
        return pcvec;
    }

    /// Get the idempotent derivative provider structure
    std::unique_ptr<const MixDerivs<>> get_derivs(const double T, const Eigen::ArrayXd &rhovec) override {
        //derivs_calls++;
        auto HD = std::make_unique<REFPROPNativeDerivProvider>(T, rhovec);
        auto ptr = std::make_unique<const MixDerivs<>>(std::move(HD));
        return ptr;
    }

    /// Get the idempotent derivative provider structure
    DerivFactory get_derivs_factory() const override {
        return [this](const double T, const Eigen::ArrayXd &rhovec) {
            return const_cast<REFPROPIsolineTracer*>(this)->get_derivs(T, rhovec);
        };
    }

    double pure_sat_call(const std::string &out, const std::string &name1, const double val1, const std::string&name2, const double val2, std::size_t index) const override {
        using namespace REFPROP_lib; 
        if (name2 != "Q"){ throw ValueError("name2 must be Q"); }
        std::vector<double> z(20, 0.0);
        z[index] = 1; z[1-index] = 0;
        int ncomp = index +1;
        PUREFLDdll(ncomp);
        double p_kPa, T, rho_mol_L, rhoLmol_L, rhoVmol_L, emol, hmol, smol, cvmol, cpmol, w;
        std::vector<double> x_liq(2), x_vap(2);  // Saturation terms
        if (name1 == "P") {
            // From REFPROP:
            //additional input--only for TQFLSH and PQFLSH
            //     kq--flag specifying units for input quality
            //         kq = 1 quality on MOLAR basis [moles vapor/total moles]
            //         kq = 2 quality on MASS basis [mass vapor/total mass]
            int kq = 1;
            int ierr = 0; char herr[255];
            p_kPa = val1 / 1000.0; double q = val2;
            // Use flash routine to find properties
            PQFLSHdll(p_kPa, q, &(z[0]), kq, T, rho_mol_L,
                rhoLmol_L, rhoVmol_L, x_liq.data(), x_vap.data(), // Saturation terms
                emol, hmol, smol, cvmol, cpmol, w, // Other thermodynamic terms
                ierr, herr, errormessagelength); // Error terms
            if (ierr > 100) { ncomp = 0; PUREFLDdll(ncomp); throw ValueError(herr); }
        }
        else if (name1 == "T") {
            // From REFPROP:
            //additional input--only for TQFLSH and PQFLSH
            //     kq--flag specifying units for input quality
            //         kq = 1 quality on MOLAR basis [moles vapor/total moles]
            //         kq = 2 quality on MASS basis [mass vapor/total mass]
            int kq = 1;
            int ierr = 0; char herr[255];
            T = val1; double q = val2;
            // Use flash routine to find properties
            TQFLSHdll(T, q, &(z[0]), kq, p_kPa, rho_mol_L,
                rhoLmol_L, rhoVmol_L, x_liq.data(), x_vap.data(), // Saturation terms
                emol, hmol, smol, cvmol, cpmol, w, // Other thermodynamic terms
                ierr, herr, errormessagelength); // Error terms
            if (ierr > 100){ ncomp = 0; PUREFLDdll(ncomp);  throw ValueError(herr); }
        }
        else { ncomp = 0; PUREFLDdll(ncomp); throw ValueError("name1 is invalid:" + name1); }

        ncomp = 0; PUREFLDdll(ncomp);
        if (out == "Dmolar") { return rho_mol_L * 1000; }
        else if (out == "P") { return p_kPa * 1000; }
        else if (out == "T") { return T; }
        else { throw ValueError("this is invalid:" + out); }
    }

    /// Actually calculate the initial state; can be over-ridden by derived class
    InitialState<> calc_initial_state() const override {

        InitialState<> state;
        auto names = get_names();
        auto fld0 = names[0], fld1 = names[1];

        std::size_t N = names.size();
        std::vector<double> rhovec(2 * N);

        GuessesStructure g;
        // For forward integration, we start at almost pure component #1
        // For backwards integration, we start at almost pure component #0
        auto fld = (m_forwards_integration) ? 1 : 0; // fluid index (0-based) for the pure fluid we are starting near
        if (imposed_variable == IMPOSED_T) {
            state.T = imposed_value;
            g.rhomolar_liq = pure_sat_call("Dmolar","T",imposed_value,"Q",0,fld);
            g.rhomolar_vap = pure_sat_call("Dmolar","T",imposed_value,"Q",1,fld);
        }
        else if (imposed_variable == IMPOSED_P) {
            // Obtain saturation temperature
            g.T = pure_sat_call("T","P",imposed_value,"Q",0, fld);
            g.rhomolar_liq = pure_sat_call("Dmolar","T",g.T,"Q",0,fld);
            g.rhomolar_vap = pure_sat_call("Dmolar", "T", g.T, "Q", 1, fld);
            state.T = g.T;
        }

        std::vector<double> rho(4, 0);
        std::size_t pure_index = (m_forwards_integration) ? 0 : 1;
        // Start off at pure
        rho[1-pure_index] = g.rhomolar_liq; // x[0]*rho' = rho[0]
        rho[pure_index] = 0;
        rho[2+1-pure_index] = g.rhomolar_vap;
        rho[2+pure_index] = 0;

        /*if (imposed_variable == IMPOSED_T) {
            bool ok = false;
            IsothermVLEResiduals<> resid(get_derivs_factory(), imposed_value, rho[imposed_index], imposed_index);
            for(double w = 1; w > 0.2; w -= 0.2){
                try{
                    rhovec = NDNewtonRaphson_Jacobian(&resid, rho, 1e-7, 30, w); 
                    for (auto i = 0; i < rhovec.size(); ++i) {
                        if (!ValidNumber(rhovec[i])) {
                            throw ValueError("Invalid value detected");
                        }
                    }
                    ok = true; break;
                }
                catch(...){
                    continue;
                }
            }
            if (!ok) {
                throw ValueError("Even with small steps, not able to initialize the isotherm");
            }
        }
        else if (imposed_variable == IMPOSED_P) {
            IsobarImposedRho0VLEResiduals<> resid(get_derivs_factory(), imposed_value, rho[imposed_index], imposed_index);
            std::vector<double> initvals(rho.size() + 1); initvals[0] = g.T;
            for (std::size_t i = 0; i < rho.size(); ++i) { initvals[i + 1] = log(rho[i]); }
            auto outvals = NDNewtonRaphson_Jacobian(&resid, initvals, 1e-7, 30);
            state.T = outvals[0];
            std::vector<double> lnrhovec = std::vector<double>(outvals.begin() + 1, outvals.end());
            rhovec.resize(lnrhovec.size());
            for (std::size_t i = 0; i < rhovec.size(); ++i) { rhovec[i] = exp(lnrhovec[i]); }
        }*/
        std::vector<double> rhovecL(rho.begin(), rho.begin() + N), rhovecV(rho.begin() + N, rho.end());
        state.rhovecL = Eigen::Map<Eigen::ArrayXd>(&rhovecL[0], rhovecL.size());
        state.rhovecV = Eigen::Map<Eigen::ArrayXd>(&rhovecV[0], rhovecV.size());
        return state;
    };

    double get_binary_interaction_double(int i, int j, const std::string &parameter){
        using namespace REFPROP_lib; 
        int icomp = static_cast<int>(i)+1, jcomp = static_cast<int>(j)+1;
        char hmodij[4], hfmix[255], hbinp[255], hfij[255], hmxrul[255];
        double fij[6];

        // Get the current state
        GETKTVdll(icomp, jcomp, hmodij, fij, hfmix, hfij, hbinp, hmxrul, 3, 255, 255, 255, 255);

        double val;
        if (parameter == "betaT"){ val = fij[0];}
        else if (parameter == "gammaT"){ val = fij[1]; }
        else if (parameter == "betaV"){ val = fij[2]; }
        else if (parameter == "gammaV"){ val = fij[3]; }
        else if (parameter == "Fij"){ val = fij[4]; }
        else{
            throw ValueError(format(" I don't know what to do with your parameter [%s]", parameter.c_str()));
            return _HUGE;
        }
        return val;
    }
};

std::vector<std::string> split_string(const std::string &s, const std::string &delim) {
    std::vector<std::string> out;
    auto start = 0U;
    auto end = s.find(delim);
    out.push_back(s.substr(start, end - start));
    while (end != std::string::npos)
    {
        start = end + delim.length();
        end = s.find(delim, start);
        out.push_back(s.substr(start, end - start));
    }
    return out;
}

#if !defined(PYBIND11)
EXPORT_CODE void CONVENTION trace(
    const char *JSON_in, 
    double *T, double *p, double *rhoL, double *rhoV, double *x0, double *y0, 
    double *errcode, char *JSON_out, const double JSON_out_size
)
{
    fpu_reset_guard guard;
    std::string message_buffer;
    try {
        nlohmann::json doc = nlohmann::json::parse(JSON_in);
        // Get variables from the JSON data structure
        std::string path = doc["path"];
        std::string imposed_variable = doc["imposed_variable"];
        std::string stepping_variable = (doc.find("stepping_variable") != doc.end()) ? doc["stepping_variable"] : ""; 
        double timeout = (doc.find("timeout") != doc.end()) ? doc["timeout"] : 6e23; // max time is very, very big
        double imposed_value = doc["imposed_value"];
        bool polishing = doc["polishing"];
        bool forwards = doc["forwards"];
        std::size_t Nallocated = doc["Nallocated"];
        double allowable_error = doc["allowable_error"];
        double maximum_pressure = doc["maximum_pressure"];
        bool unstable_termination = doc["unstable_termination"];
        std::string fluids = doc["fluids"];
        std::string HMX_BNC = (doc.find("hmx_path") != doc.end()) ? doc["hmx_path"] : "HMX.BNC";

        std::string shared_library_filename = "";
        std::string err = "";
        bool did_load = my_load_REFPROP(err, path, shared_library_filename);
        if (!did_load) { throw CoolProp::ValueError("Unable to load REFPROP DLL with error: " + err); }
        
        char RPpath[255];
        strcpy(RPpath, path.c_str());
        REFPROP_lib::SETPATHdll(RPpath, 255);

        auto var = (imposed_variable == "T") ? REFPROPIsolineTracer::IMPOSED_T : REFPROPIsolineTracer::IMPOSED_P;
        REFPROPIsolineTracer tracer(var, imposed_value, strsplit(fluids, '&'), HMX_BNC);
        tracer.set_forwards_integration(static_cast<bool>(forwards));
        tracer.polishing(polishing);
        tracer.set_max_size(Nallocated);
        tracer.set_allowable_error(allowable_error);
        tracer.set_maximum_pressure(maximum_pressure);
        tracer.set_unstable_termination(static_cast<bool>(unstable_termination));
        tracer.set_timeout(timeout);
        if (!stepping_variable.empty()){
            if (stepping_variable == "STEP_IN_RHO0"){
                tracer.set_stepping_variable(REFPROPIsolineTracer::STEP_IN_RHO0);
            }
            else if (stepping_variable == "STEP_IN_RHO1") {
                tracer.set_stepping_variable(REFPROPIsolineTracer::STEP_IN_RHO1);
            }
            else if (stepping_variable == "STEP_IN_P") {
                tracer.set_stepping_variable(REFPROPIsolineTracer::STEP_IN_P);
            }
            else if (stepping_variable == "STEP_IN_T") {
                tracer.set_stepping_variable(REFPROPIsolineTracer::STEP_IN_T);
            }
            else{
                throw CoolProp::ValueError(format("Invalid stepping variable: [%s]", stepping_variable.c_str()));
            }
        }

        tracer.trace();
        IsolineTracerData<> data0 = tracer.get_tracer_data();
        std::cout << tracer.get_tracing_time() << " s; N: " << data0.pL.size() << " p[-1]: " << (!data0.pL.empty() ? std::to_string(data0.pL.back()) : "empty") << std::endl;

        if (data0.pL.size() > static_cast<std::size_t>(Nallocated)) {
            throw CoolProp::ValueError(format("Length of data vector [%d] is greater than allocated buffer length [%d]", static_cast<int>(data0.pL.size()), static_cast<int>(Nallocated)));
        }
        int N = static_cast<int>(data0.pL.size());
        for (std::size_t i = 0; i < data0.pL.size(); ++i) {
            *(T + i) = data0.TL[i];
            *(p + i) = data0.pL[i];
            *(rhoL + i) = std::accumulate(data0.rhoL[i].begin(), data0.rhoL[i].end(), 0.0);
            *(rhoV + i) = std::accumulate(data0.rhoV[i].begin(), data0.rhoV[i].end(), 0.0);
            *(x0 + i) = data0.x[i][0];
            *(y0 + i) = data0.y[i][0];
        }
        nlohmann::json j = {{"N", N},
                            {"message_buffer",tracer.get_termination_reason()},
                            {"elapsed [s]", tracer.get_tracing_time()}
                            };
        std::string out = j.dump();
        if (out.size() < JSON_out_size){
            strcpy(JSON_out, out.c_str());
            *errcode = tracer.get_premature_termination_code();
        }
        else {
            *errcode = 1.0;
        }
    }
    catch (...) {
        HandleException(*errcode, message_buffer);
        nlohmann::json j = {
            { "errcode", *errcode},
            { "message_buffer", message_buffer}
        };
        std::string out = j.dump();
        if (out.size() < JSON_out_size) {
            strcpy(JSON_out, out.c_str());
        }
    }
}
#else

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

/* Trampoline class; passes calls up into the derived class in python
* One overload is needed for each virtual function you want to be able to overload
*/
class PyREFPROPIsolineTracer : public REFPROPIsolineTracer {
public:
    /* Inherit the constructors */
    using REFPROPIsolineTracer::REFPROPIsolineTracer;

    std::pair<double, double> get_integration_limits() override {
        // Release the GIL
        py::gil_scoped_release release;
        {
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;

            // Parameters to macro are: Return type, parent class, name of function in C++, argument(s)
            typedef std::pair<double, double> pairdoubledouble;
            PYBIND11_OVERLOAD(pairdoubledouble,
                REFPROPIsolineTracer,
                get_integration_limits,
                // No arguments
                );
        }
    }
    InitialState<double> calc_initial_state() const override {
        // Release the GIL
        py::gil_scoped_release release;
        {
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;

            // Parameters to macro are: Return type, parent class, name of function in C++, argument(s)
            PYBIND11_OVERLOAD(InitialState<double>,
                REFPROPIsolineTracer,
                calc_initial_state,
                // No arguments
                );
        }
    }
    stepping_variable determine_integration_type() override {
        // Release the GIL
        py::gil_scoped_release release;
        {
            // Acquire GIL before calling Python code
            py::gil_scoped_acquire acquire;

            // Parameters to macro are: Return type, parent class, name of function in C++, argument(s)
            PYBIND11_OVERLOAD(stepping_variable,
                REFPROPIsolineTracer,
                determine_integration_type,
                // No arguments
                );
        }
    }
};

void init_REFPROPisochoricthermo(py::module &m) {

    py::class_<REFPROPIsolineTracer, AbstractIsolineTracer<double>, PyREFPROPIsolineTracer >(m, "REFPROPIsolineTracer")
        .def(py::init<AbstractIsolineTracer<double>::imposed_variable_types, double, const std::string &, const std::vector<std::string> &, const std::string&>())
        .def(py::init<AbstractIsolineTracer<double>::imposed_variable_types, double, const std::string &, const std::vector<std::string> &>())
        .def("get_binary_interaction_double", &REFPROPIsolineTracer::get_binary_interaction_double)
        ;

    m.def("load_REFPROP", [](const std::string &path) {
        std::string shared_library_filename = "";
        std::string err = "";
        bool did_load = my_load_REFPROP(err, path, shared_library_filename);
        if (!did_load) { throw CoolProp::ValueError("Unable to load REFPROP DLL with error: " + err); }

        char RPpath[255];
        strcpy(RPpath, path.c_str());
        REFPROP_lib::SETPATHdll(RPpath, 255);

    }
    );

    m.def("unload_REFPROP", []() {
        std::string err = "";
        bool did_unload = my_unload_REFPROP(err);
        if (!did_unload) { throw CoolProp::ValueError("Unable to unload REFPROP DLL with error: " + err); }
    }
    );
}
#endif