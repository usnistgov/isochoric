#ifndef COMPLEX_DERIVATIVES_H
#define COMPLEX_DERIVATIVES_H

#include <complex>

// This should be enough digits of pi...
constexpr double PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058;

#include <Eigen/Dense>

template<typename T>
T kahanSum(const Eigen::Array<T, Eigen::Dynamic, 1> &x)
{
    T sum = x[0], y, t;
    T c = 0.0;          //A running compensation for lost low-order bits.
    for (auto i = 1; i < x.size(); ++i)
    {
        y = x[i] - c;    //So far, so good: c is zero.
        t = sum + y;         //Alas, sum is big, y small, so low-order digits of y are lost.
        c = (t - sum) - y;   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
        sum = t;             //Algebraically, c should always be zero. Beware eagerly optimising compilers!
    }
    return sum;
}

/**
* Trapezoidal integration
**/
static std::complex<double> trapz(const Eigen::ArrayXd &x, const Eigen::ArrayXcd &y)
{
    auto N = x.size();
    Eigen::ArrayXcd els = (y.tail(N - 1) + y.head(N - 1))*0.5*(x.tail(N - 1) - x.head(N - 1)).cast<std::complex<double> >();
    auto I0 = els.sum();
    //auto I1 = kahanSum(els);
    return I0;
}

static Eigen::ArrayXcd complex_derivative(std::function<std::complex<double>(std::complex<double>)> &f, std::size_t Nderivs, std::complex<double> z0, double r, Eigen::Index N) {
    Eigen::ArrayXcd z, y(N), out, z_C(N);
    out.resize(Nderivs + 1);
    Eigen::ArrayXd phi = Eigen::ArrayXd::LinSpaced(N, 0, 2 * PI);
    z_C.real() = r*cos(phi); z_C.imag() = r*sin(phi);
    z = z0 + z_C;

    // Evaluate points on the circle around the point
    for (auto i = 0; i < N; ++i) {
        y(i) = f(z(i));
    }
    //Eigen::ArrayXcd integrand = y / z_C.pow(6);
    //for (auto i = 0; i < integrand.size(); ++i) {
    //    printf("%d %20.18f %20.18f\n", i, integrand[i].real(),integrand[i].imag());
    //}
    // Evaluate all the derivatives we want
    out[0] = 0;
    for (auto ideriv = 1; ideriv <= Nderivs; ++ideriv) {
        Eigen::ArrayXcd integrand = y / z_C.pow(ideriv);
        out[ideriv] = std::tgamma(ideriv + 1) / (2 * PI)*trapz(phi, integrand);
    }
    return out;
}
#endif
