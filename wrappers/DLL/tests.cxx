#include "providers.h"
#include "mixderiv.h"

#include "funcs.h"

#include "AbstractState.h"
#include "Backends/Helmholtz/HelmholtzEOSMixtureBackend.h"

#define REFPROP_LIB_NAMESPACE REFPROP_lib
#include "REFPROP_lib.h"
#undef REFPROP_LIB_NAMESPACE

#include "Backends/REFPROP/REFPROPMixtureBackend.h"

using namespace CoolProp;

int main(){

    //PengRobinsonBackend HEOS(strsplit("Methane&Ethane",'&'));
    HelmholtzEOSMixtureBackend HEOS(strsplit("Methane&Ethane", '&'));
    const std::vector<double> rhovec = {0.1, 5};
    double rho = std::accumulate(rhovec.begin(), rhovec.end(), 0.0), drho = rho*1e-6, T = 300;
    std::size_t i = 1;
    {
        using namespace REFPROP_lib;
        std::string RPPREFIX(getenv("RPPREFIX"));
        set_config_string(ALTERNATIVE_REFPROP_PATH, RPPREFIX);

        bool is_loaded = CoolProp::REFPROPMixtureBackend::REFPROP_supported();
        long ierr = 0, nc = 2;
        char herr[255], hfld[10000] = "METHANE.FLD|ETHANE.FLD", hhmx[] = "HMX.BNC", href[] = "DEF";
        
        SETPATHdll(const_cast<char*>(RPPREFIX.c_str()), static_cast<long>(RPPREFIX.size()));
        SETUPdll(nc, hfld, hhmx, href, ierr, herr, 10000, 255, 3, 255);
        ierr = 1;
    }
    
    Eigen::Map<const Eigen::ArrayXd> _mapped_rhovec(&(rhovec[0]), rhovec.size());
    Eigen::ArrayXd mapped_rhovec = _mapped_rhovec;
    auto tic = std::chrono::high_resolution_clock::now();
    std::unique_ptr<const AbstractNativeDerivProvider> RP(new REFPROPNativeDerivProvider(T, mapped_rhovec));
    std::unique_ptr<const MixDerivs>_mix( new MixDerivs(std::move(RP)));
    const MixDerivs &mix = *(_mix.get());
    auto toc = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(toc-tic).count()*1e6 << " us" << std::endl;

    {
        auto tic = std::chrono::high_resolution_clock::now();
        std::unique_ptr<const AbstractNativeDerivProvider> HD(new CoolPropNativeDerivProvider(HEOS, T, mapped_rhovec));
        std::unique_ptr<const MixDerivs> (new MixDerivs(std::move(HD)));
        auto toc = std::chrono::high_resolution_clock::now();
        std::cout << std::chrono::duration<double>(toc - tic).count()*1e6 << " us" << std::endl;
    }

    std::unique_ptr<const AbstractNativeDerivProvider> HD(new CoolPropNativeDerivProvider(HEOS, T, mapped_rhovec));
    std::unique_ptr<const MixDerivs> _cp(new MixDerivs(std::move(HD)));
    const MixDerivs &cp = *(_cp.get());

    Eigen::ArrayXd z = mapped_rhovec/mapped_rhovec.sum();
    std::vector<double> zz(z.data(), z.data()+mapped_rhovec.size());
    HEOS.set_mole_fractions(rhovec);
    double ana0a = HEOS.T_reducing();
    //double REF0a = mix.get_native().Tr();
    auto GERG = static_cast<CoolProp::GERG2008ReducingFunction*>(HEOS.Reducing.get());
    
    auto N = z.size();
    std::cout << "z:\n" << mapped_rhovec << std::endl;
    std::cout << "i,dTr_dxi,drhordxi\n";
    for (auto i = 0; i < N; ++i) {
        std::cout << i << " " << GERG->dTrdxi__constxj(rhovec, i, XN_INDEPENDENT) << " " << GERG->drhormolardxi__constxj(rhovec, i, XN_INDEPENDENT) << std::endl;
    }
    std::cout << "i,j,d2Trdxidxj,d2rhordxidxj\n";
    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < N; ++j){
            std::cout << i << " " << j << " " << GERG->d2Trdxidxj(rhovec, i, j, XN_INDEPENDENT) << " " << GERG->d2rhormolardxidxj(rhovec, i, j, XN_INDEPENDENT) << std::endl;
        }
    }
    
    for (auto i = 0; i < 2; ++i) {

        double num1a = dTr_drhoi_numeric(HEOS, rhovec, i, drho);
        double ana1a = dTr_drhoi_analytic(HEOS, rhovec, i);
        double REF1a = mix.get_native().dTr_drhoi(i);
        double num1b = drhor_drhoi_numeric(HEOS, rhovec, i, drho);
        double ana1b = drhor_drhoi_analytic(HEOS, rhovec, i);
        double REF1b = mix.get_native().drhor_drhoi(i);

        double REF13a = mix.get_native().dTr_dxi__constxj(i);
        double COO13a = cp.get_native().dTr_dxi__constxj(i);
        double REF13b = mix.get_native().drhor_dxi__constxj(i);
        double COO13b = cp.get_native().drhor_dxi__constxj(i);

        double num2a = dtau_drhoi_numeric(HEOS, rhovec, T, i, drho);
        double ana2a = dtau_drhoi_analytic(HEOS, rhovec, T, i);
        double REF2a = mix.dtau_drhoi__constTrhoj(i);
        double num2b = ddelta_drhoi_numeric(HEOS, rhovec, i, drho);
        double ana2b = ddelta_drhoi_analytic(HEOS, rhovec, i);
        double REF2b = mix.ddelta_drhoi__constTrhoj(i);

        double COO11 = cp.drhorTr_dxi__xj(i);
        double REF11 = mix.drhorTr_dxi__xj(i);

        double ana4 = get_dpsi_ddelta(HEOS, rhovec, T);
        double num4 = dpsi_ddelta_numeric(HEOS, rhovec, T, drho);
        double REF4 = mix.dpsi_ddelta();

        double ana5 = get_dpsi_dtau(HEOS, rhovec, T);
        double num5 = dpsi_dtau_numeric(HEOS, rhovec, T, drho);
        double REF5 = mix.dpsi_dtau();

        double ana10 = dpsir_dxi__consttaudeltaxj(HEOS, rhovec, T, i);
        double COO10 = cp.dpsir_dxi__consttaudeltaxj(i);
        double REF10 = mix.dpsir_dxi__consttaudeltaxj(i);

        double REF7 = mix.get_native().d2alphar_dxi_dtau__constdeltaxj(i);
        double COO7 = cp.get_native().d2alphar_dxi_dtau__constdeltaxj(i);

        double REF9 = mix.get_native().d2alphar_dxi_ddelta__consttauxj(i);
        double COO9 = cp.get_native().d2alphar_dxi_ddelta__consttauxj(i);

        double REFaa9 = mix.d_dpsir_ddelta_drhoi__constrhoj(i);
        double COOaa9 = cp.d_dpsir_ddelta_drhoi__constrhoj(i);

        double REF8 = mix.get_native().dalphar_dxi__taudeltaxj(i);
        double COO8 = cp.get_native().dalphar_dxi__taudeltaxj(i);

        double ana3 = dpsir_drhoi(HEOS, rhovec, T, i);
        double REF3 = mix.dpsir_drhoi__constTrhoj(i);
        double COO3 = cp.dpsir_drhoi__constTrhoj(i);

        for (auto j = 0; j < 2; ++j) {

            double REF343a = mix.get_native().d2Tr_dxidxj (i, j);
            double COO343a = cp.get_native().d2Tr_dxidxj(i, j);

            double REF343b = mix.get_native().d2rhor_dxidxj(i, j);
            double COO343b = cp.get_native().d2rhor_dxidxj(i, j);
            
            double REF34a = mix.d2rhorTr_dxidxj__consttaudelta(i, j);
            double COO34a = cp.d2rhorTr_dxidxj__consttaudelta(i, j);

            double REF34 = mix.get_native().d2Tr_dxidxj(i, j);
            double COO34 = cp.get_native().d2Tr_dxidxj(i, j);
            
            double ANA343 = d2Tr_drhoidrhoj_analytic(HEOS, rhovec, i, j);
            double REF343 = mix.get_native().d2Tr_drhoidrhoj(i, j);
            double COO343 = cp.get_native().d2Tr_drhoidrhoj(i, j);

            double ANA3431 = d2rhor_drhoidrhoj_analytic(HEOS, rhovec, i, j);
            double REF3431 = mix.get_native().d2rhor_drhoidrhoj(i, j);
            double COO3431 = cp.get_native().d2rhor_drhoidrhoj(i, j);

            double REF3430 = mix.d_dpsir_dxm_drhoi__constTrhoi(i, j);
            double COO3430 = cp.d_dpsir_dxm_drhoi__constTrhoi(i, j);

            double REF343v = mix.d2psir_dxidxj__consttaudelta(i, j);
            double COO343v = cp.d2psir_dxidxj__consttaudelta(i, j);

            double REFr33 = mix.get_native().d2alphar_dxidxj__consttaudelta(i, j);
            double COOr33 = cp.get_native().d2alphar_dxidxj__consttaudelta(i, j);

            double ana33 = d2psir_drhoidrhoj(HEOS, rhovec, T, i, j);
            double REF33 = mix.d2psir_drhoidrhoj__constT(i, j);
            double COO33 = cp.d2psir_drhoidrhoj__constT(i, j);
            
            int ok = 3;
        }
    }
    
    return EXIT_SUCCESS;
}