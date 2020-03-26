#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "isochoric/coolprop_tracer.h"

#include "isochoric/complex_derivatives.h"

TEST_CASE("Check derivatives of exp(x)","")
{
    double x0 = 0.7;
    double f1 = exp(x0);
    auto r = 0.5;
    int N = 61;
    std::complex<double> z0 = x0;
    std::function<std::complex<double>(std::complex<double>)> f = [](std::complex<double> z) { return exp(z); };
    Eigen::ArrayXcd derivs = complex_derivative(f, 15, z0, r, N);
    Eigen::ArrayXd real_devs = derivs.cwiseAbs() - f1;
    auto error = real_devs.cwiseAbs().segment(1,6).sum();
    CHECK(error < 1e-10);
}

TEST_CASE("CO2 + ethane @ 290 K"){
    VLEIsolineTracer<> IT(VLEIsolineTracer<>::IMPOSED_T, 290, "HEOS", strsplit("CO2&ethane", '&'));
    IT.polishing(true);
    IT.set_forwards_integration(true);
    IT.trace();
    IsolineTracerData<> data = IT.get_tracer_data();
    SECTION("Difference of cached chemical potentials") {
        double full_chempot_err = 0.0;
        for (auto k = 0; k < data.pL.size(); ++k) {
            full_chempot_err += std::abs(data.chempot0L[k] - data.chempot0V[k]);
        }
        CHECK(full_chempot_err < 1);
    }
    SECTION("Difference of cached chemical potentials") {
        double recalc_chempot_err = 0.0;
        for (auto k = 0; k < data.pL.size(); ++k) {
            auto N = 2;
            Eigen::Map<const Eigen::ArrayXd> rhovecL(&(data.rhoL[k][0]), N), rhovecV(&(data.rhoV[k][0]), N);
            auto chempot0L = IT.get_derivs(data.TL[k], rhovecL)->dpsi_drhoi__constTrhoj(0);
            auto chempot0V = IT.get_derivs(data.TL[k], rhovecV)->dpsi_drhoi__constTrhoj(0);
            recalc_chempot_err += std::abs(chempot0L - chempot0L);
        }
        CHECK(recalc_chempot_err < 1);
    }
    SECTION("Difference of residual(ish) chemical potentials") {
        double recalc_chempotrish_err = 0.0;
        for (auto k = 0; k < data.pL.size(); ++k) {
            auto N = 2;
            Eigen::Map<const Eigen::ArrayXd> rhovecL(&(data.rhoL[k][0]), N), rhovecV(&(data.rhoV[k][0]), N);
            const auto& derL = IT.get_derivs(data.TL[k], rhovecL), &derV = IT.get_derivs(data.TV[k], rhovecV);
            auto chempotr0L = derL->dpsir_drhoi__constTrhoj(0) + derL->R*derL->T*log(data.rhoL[k][0]);
            auto chempotr0V = derV->dpsir_drhoi__constTrhoj(0) + derV->R*derV->T*log(data.rhoV[k][0]);
            recalc_chempotrish_err += std::abs(chempotr0L - chempotr0V);
        }
        CHECK(recalc_chempotrish_err < 1e-3);
    }
    SECTION("Check calculation of chemical potential") {
        double recalc_chempotrish_err = 0.0;
        for (auto k = 0; k < data.pL.size(); ++k) {
            auto N = 2;
            Eigen::Map<const Eigen::ArrayXd> rhovecL(&(data.rhoL[k][0]), N), rhovecV(&(data.rhoV[k][0]), N);
            auto valfromtracer = data.chempot0L[k];
            auto AS = IT.get_AbstractState_pointer();
            AS->specify_phase(iphase_gas);
            
            auto& rhoLk = data.rhoL[k];
            double rhomolarL = std::accumulate(rhoLk.begin(), rhoLk.end(), 0.0);
            AS->update(DmolarT_INPUTS, rhomolarL, data.TL[k]);
            auto valfromCP0L = AS->chemical_potential(0);
            auto valfromCP1L = AS->chemical_potential(1);

            auto& rhoVk = data.rhoV[k];
            double rhomolarV = std::accumulate(rhoVk.begin(), rhoVk.end(), 0.0);
            AS->update(DmolarT_INPUTS, rhomolarV, data.TL[k]);
            auto valfromCP0V = AS->chemical_potential(0);
            auto valfromCP1V = AS->chemical_potential(1);

            AS->unspecify_phase();
            
            recalc_chempotrish_err += std::abs(valfromCP0L - valfromCP0V);
        }
        CHECK(recalc_chempotrish_err < 1);
    }
}