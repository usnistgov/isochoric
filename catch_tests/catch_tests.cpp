#define CATCH_CONFIG_MAIN
#include "catch.hpp""

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