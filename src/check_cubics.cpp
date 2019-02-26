#include "isochoric/cubic_provider.h"
#include "isochoric/mixderiv.h"
#include "isochoric/complex_derivatives.h"

#include <iostream>
#include <complex>
#include <chrono>

#include "ODEIntegrators.h"
#include "AbstractState.h"

class DerivCheckerHolder {
public:
    typedef std::unique_ptr<const AbstractCubic<std::complex<double> > > EOSrf;
    typedef std::complex<double> TYPE;
protected:
    const EOSrf &m_cubic;
    TYPE m_T;
    Eigen::Array<TYPE, Eigen::Dynamic, 1> m_rhovec;
    Eigen::Index m_i;
public:
    DerivCheckerHolder(const EOSrf &cubic, TYPE T, const Eigen::Array<TYPE, Eigen::Dynamic, 1> &rhovec, Eigen::Index i)
        : m_cubic(cubic), m_T(T), m_rhovec(rhovec), m_i(i) {
    };
    std::complex<double> Psir(const std::complex<double> &rhoi) {
        Eigen::Array<TYPE, Eigen::Dynamic, 1> rhovec = m_rhovec;
        rhovec[m_i] = rhoi;
        std::unique_ptr<AbstractNativeDerivProvider<TYPE> > HD(new CubicNativeDerivProvider<TYPE>(m_cubic, m_T, rhovec));
        std::unique_ptr<const MixDerivs<TYPE> > _cp(new MixDerivs<TYPE>(std::move(HD)));
        return _cp->psir();
    }
};

template<typename TYPE>
void check_cubic() {
    std::vector<TYPE> rhovec = { 2045.86591188, 8183.46364751};
    Eigen::Map<Eigen::Array<TYPE, Eigen::Dynamic, 1> > mapped_rhovec(&(rhovec[0]), rhovec.size());
    // Nominally nitrogen + methane
    std::vector<TYPE> Tc = {126.192,190.564 };
    std::vector<TYPE> pc = {3395800.0,4599200.0 };
    std::vector<TYPE> acentric = {0.0372,0.01142};
    TYPE R = 8.3144598;
    TYPE T = 180.6381701029613;

    std::vector<std::vector<TYPE>> k(2, std::vector<TYPE>(2, 0.0));
    //k[0][1] = 4.28227e-2; k[1][0] = k[0][1];
    std::unique_ptr<const AbstractCubic<TYPE> > cubic(new PengRobinson<TYPE>(Tc,pc,acentric,R,k));

    Eigen::Array<TYPE,Eigen::Dynamic,1> eig_rhovec = mapped_rhovec;
    std::unique_ptr<AbstractNativeDerivProvider<TYPE> > HD(new CubicNativeDerivProvider<TYPE>(cubic, T, eig_rhovec));
    std::unique_ptr<const MixDerivs<TYPE> > _cp(new MixDerivs<TYPE>(std::move(HD)));
    const MixDerivs<TYPE> &cp = *(_cp.get());
    std::size_t ii = 0;
    auto dpsir_drho10 = _cp->dpsir_drhoi__constTrhoj(ii);
    auto d2psir_drho10 = _cp->d2psir_drhoidrhoj__constT(ii, ii);

    DerivCheckerHolder dh(cubic, T, mapped_rhovec, 0);
    auto psir1 = dh.Psir(rhovec[0]);

    std::function<std::complex<double>(std::complex<double>)> f = std::bind(&DerivCheckerHolder::Psir, &dh, std::placeholders::_1);
    auto val = complex_derivative(f, 3, rhovec[0], 0.5*std::real(rhovec[0]), 211);
    auto diff = val[1]- dpsir_drho10;
}

template<typename TYPE>
std::unique_ptr<const AbstractCubic<TYPE> > get_cubic(const std::string & fluid1, const std::string &fluid2)
{
    if (fluid1 == "Nitrogen" && fluid2 == "Methane"){
        // Nominally nitrogen + methane
        std::vector<TYPE> Tc = { 126.192,190.564 };
        std::vector<TYPE> pc = { 3395800.0,4599200.0 };
        std::vector<TYPE> acentric = { 0.0372,0.01142 };
        TYPE R = 8.3144598;
        std::vector<std::vector<TYPE>> k(2, std::vector<TYPE>(2, 0.0));
        return std::make_unique<const PengRobinson<TYPE>>(Tc, pc, acentric, R, k);
    }
    else if (fluid1 == "Methane" && fluid2 == "Propane"){
        std::vector<TYPE> Tc = {190.555, 369.825};
        std::vector<TYPE> pc = {4.595e6, 4.248e6};
        std::vector<TYPE> acentric = { 0.0, 1.53080000e-01 };
        TYPE R = 8.3144598;
        std::vector<std::vector<TYPE>> k(2, std::vector<TYPE>(2, 0.0));
        k[0][1] = 4.28227e-2; k[1][0] = k[0][1];
        return std::make_unique<const PengRobinson<TYPE>>(Tc, pc, acentric, R, k);
    }
    else{
        throw std::invalid_argument("Fluids are not valid");
    }
}

Eigen::ArrayXd IsothermalPolish(const typename MixDerivs<std::complex<double>>::ProviderFactory &provider_factory,
                                const typename MixDerivs<std::complex<double>>::ProviderFactory &Cauchy_provider_factory,
                                const double T,
                                const Eigen::ArrayXd &rhovec0)
{

    // Convenience typedefs
    typedef std::complex<double> TYPE;
    
    // Copy of the initial vector of molar concentrations
    Eigen::ArrayXd rhovec = rhovec0;
    
    // A convenience lambda function here to generate the needed derivatives
    // In this case a two-element array of the second and third derivatives with respect to sigma_1
    // Both of those values should be zero at the critical point
    auto derivs_func = [&provider_factory, &Cauchy_provider_factory](const double T, const Eigen::ArrayXd& rhovec) {
        auto derivs = std::make_unique<const MixDerivs<TYPE>>(provider_factory(T, rhovec));
        return std::get<0>(derivs->get_dnPsi_dsigma1n(Cauchy_provider_factory)).real().segment(2,2);
    };
    
    Eigen::ArrayXd drhoi = 1e-6*rhovec;
    Eigen::MatrixXd J(2,2), r(2,1);
    auto drho = [drhoi](Eigen::Index N, Eigen::Index i) {
        Eigen::ArrayXd _drho = Eigen::ArrayXd::Zero(N);
        _drho(i) = drhoi(i);
        return _drho;
    };
    auto Nsteps_max = 10;
    double r0 = -1;
    for (auto counter = 0; counter < Nsteps_max; ++counter){
        // Construct Jacobian by column
        J.col(0) = (derivs_func(T, rhovec+drho(2,0)) - derivs_func(T, rhovec-drho(2,0)))/(2.0*drhoi(0));
        J.col(1) = (derivs_func(T, rhovec+drho(2,1)) - derivs_func(T, rhovec-drho(2,1)))/(2.0*drhoi(1));
        // Residue
        r = derivs_func(T, rhovec);
        if (counter == 0){
            r0 = r.matrix().norm();
        }
        // Step from Newton-Raphson
        Eigen::MatrixXd deltarho = J.colPivHouseholderQr().solve(-r);
        rhovec += deltarho.array();
        auto stepnorm = deltarho.norm(), rhonorm = rhovec.matrix().norm();
        if (stepnorm < 1e-8*rhonorm){
            break;
        }
        auto rnorm = r.matrix().norm();
        if (rnorm < 1e-16){
            break;
        }
        if (counter == Nsteps_max-1){
            break;
        }
    }
    std::cout << r.matrix().norm() << " " << r0 << std::endl;
    return rhovec;
}

void test_sigma1_deriv() {
    typedef std::complex<double> TYPE;
    
    // Nominally nitrogen + methane
    TYPE T = 180.638170102961311159;
    const std::vector<TYPE> rhovec = { 2045.865911877545386233, 8183.463647510181544931 };
    const std::unique_ptr<const AbstractCubic<TYPE> > cubic = get_cubic<TYPE>("Nitrogen", "Methane");
    
    Eigen::Map<const MixDerivs<TYPE>::EigenArray > mapped_rhovec(&(rhovec[0]), rhovec.size());
    
    const MixDerivs<TYPE>::EigenArray eig_rhovec = mapped_rhovec;
    MixDerivs<TYPE>::ProviderFactory cubic_provider_factory = [&cubic](const TYPE T, const MixDerivs<TYPE>::EigenArray& rhovec) {
        return std::make_unique<CubicNativeDerivProvider<TYPE>>(cubic, T, rhovec);
    };
    
    auto md = std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec));
    auto base_derivs = md->get_dnPsi_dsigma1n(cubic_provider_factory, 4);
    MixDerivs<TYPE>::EigenArray derivstot, derivs0,derivsr;
    std::tie(derivstot, derivsr, derivs0) = base_derivs;
    MixDerivs<TYPE>::EigenMatrix H, U_T;
    MixDerivs<TYPE>::EigenArray eigs;
    md->get_Hessian_and_eigs(H,eigs,U_T);
//    for (auto i = 2; i <= 4; ++i){
//        std::cout << i << ": " << derivs0(i) << std::endl;
//    }
    for (auto i = 2; i <= 4; ++i){
        std::cout << i << ": " << derivsr(i) << std::endl;
    }
    
    TYPE dsigmai = 1e-3*rhovec[0];
    for (auto j = 0; j <= 1; ++j){
        MixDerivs<TYPE>::EigenArray v0 = U_T.col(0), v = U_T.col(j);
        MixDerivs<TYPE>::EigenArray derivsrplus = std::get<1>(std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec + (v*dsigmai).array()))->get_dnPsi_dsigma1n(cubic_provider_factory,4,0,v0));
        MixDerivs<TYPE>::EigenArray derivsrminus = std::get<1>(std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec - (v*dsigmai).array()))->get_dnPsi_dsigma1n(cubic_provider_factory,4,0,v0));
        MixDerivs<TYPE>::EigenArray deriv3 = (derivsrplus-derivsrminus)/(2.0*dsigmai);
        std::cout << " ---------- for j="+std::to_string(j)+" ---------\n";
        for (auto i = 1; i < 4; ++i){
            std::cout << i+1 << ": " << deriv3(i) << std::endl;
        }
    }
    // This works too - you need to use the original eigenvector you obtained
    
    TYPE derivs2plus = std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec + 2.0*(U_T.col(0)*dsigmai).array()))->psir();
    TYPE derivsplus = std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec + (U_T.col(0)*dsigmai).array()))->psir();
    TYPE derivsbase = std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec))->psir();
    TYPE derivsminus = std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec - (U_T.col(0)*dsigmai).array()))->psir();
    TYPE derivs2minus = std::make_unique<const MixDerivs<TYPE>>(cubic_provider_factory(T, eig_rhovec - 2.0*(U_T.col(0)*dsigmai).array()))->psir();
    TYPE base_diff2r = ((derivsplus -2.0*derivsbase + derivsminus)/(dsigmai*dsigmai));
    TYPE base_diff3r = ((derivs2plus -2.0*derivsplus +2.0*derivsminus-derivs2minus)/(2.0*dsigmai*dsigmai*dsigmai));
    TYPE base_diff4r = ((derivs2plus -4.0*derivsplus +6.0*derivsbase -4.0*derivsminus+derivs2minus)/(dsigmai*dsigmai*dsigmai*dsigmai));
    std::cout << "---------\n2: " << base_diff2r << "\n3: "  << base_diff3r << "\n4: "  << base_diff4r << std::endl;
    // This checks out - the derivatives with Cauchy agree with the finite differences in the eigenvector direction associated with the minimum eigenvalue
    int rr =0;
}


/// The abstract class defining the interface for the integrator routines
class CriticalIntegrator : public ODEIntegrators::AbstractODEIntegrator{
private:
    typedef typename MixDerivs<std::complex<double> >::ProviderFactory ProviderFactory;
    
    const ProviderFactory m_provider_factory;
    const ProviderFactory m_Cauchy_provider_factory;
    double m_T;
    Eigen::ArrayXd m_rhovec;
public:
    CriticalIntegrator(const ProviderFactory &provider_factory,
                       const ProviderFactory &Cauchy_provider_factory,
                       const double T,
                       const Eigen::ArrayXd &rhovec)
        : m_provider_factory(provider_factory), m_Cauchy_provider_factory(Cauchy_provider_factory), m_T(T), m_rhovec(rhovec)
    {
    }
    
    virtual std::vector<double> get_initial_array() const override{
        return std::vector<double>(m_rhovec.data(), m_rhovec.data() + m_rhovec.size());
    };
    
    // These methods must be implemented, but they don't really do anything...
    virtual void pre_step_callback() override { return; }
    virtual bool premature_termination() override { return false; };
    virtual void post_deriv_callback() override { return; };
    
    virtual void post_step_callback(double t, double h, std::vector<double> &x) override{
        Eigen::ArrayXd rhovec = Eigen::Map<const Eigen::ArrayXd>(&(x[0]), x.size());
        rhovec = IsothermalPolish(m_provider_factory, m_Cauchy_provider_factory, t, rhovec);
        x = std::vector<double>(rhovec.data(), rhovec.data() + rhovec.size());
        std::cout << t << " " << h << " " << x[0]/(x[0]+x[1]) << "," << x[1]/(x[0]+x[1]) << std::endl;
    };
    
    virtual void derivs(double t, std::vector<double> &x, std::vector<double> &f) override{
        Eigen::ArrayXcd rhovec = Eigen::Map<const Eigen::ArrayXd>(&(x[0]), x.size());
        Eigen::ArrayXd drhovec_dT = calc_drhovec_dT_crit<std::complex<double>>(m_provider_factory, m_Cauchy_provider_factory, t, rhovec).real();
        f = std::vector<double>(drhovec_dT.data(), drhovec_dT.data() + drhovec_dT.size());
    };
};

void trace()
{
    typedef std::complex<double> TYPE;
    // Nominally nitrogen + methane
    MixDerivs<TYPE>::EigenArray rhovec(2); rhovec << 2045.865911877545386233, 8183.463647510181544931;
    TYPE T = 180.638170102961311159;
    const std::unique_ptr<const AbstractCubic<TYPE> > cubic = get_cubic<TYPE>("Nitrogen", "Methane");
    
    MixDerivs<TYPE>::EigenArray rhovec0 = rhovec;
    TYPE T0 = T;
    
    const MixDerivs<TYPE>::ProviderFactory cubic_provider_factory = [&cubic](const TYPE T, const MixDerivs<TYPE>::EigenArray& rhovec) {
        bool only_psir = false; // Everything
        return std::make_unique<CubicNativeDerivProvider<TYPE>>(cubic, T, rhovec, only_psir);
    };
    const MixDerivs<TYPE>::ProviderFactory cubic_Cauchy_provider_factory = [&cubic](const TYPE T, const MixDerivs<TYPE>::EigenArray& rhovec) {
        bool only_psir = true; // Not the rest of the stuff, just psir
        return std::make_unique<CubicNativeDerivProvider<TYPE>>(cubic, T, rhovec, only_psir);
    };
    
//    TYPE dT = +1e-3 ;
//    for (auto ii = 0; ii < 10000; ++ii){
//        MixDerivs<TYPE>::EigenArray drhovec_dT = calc_drhovec_dT_crit<TYPE>(cubic_provider_factory, cubic_Cauchy_provider_factory, T, rhovec);
//        rhovec += drhovec_dT*dT;
//        T += dT;
////        rhovec = IsothermalPolish(cubic_provider_factory, cubic_Cauchy_provider_factory, std::real(T), rhovec.real()); // polish
//        MixDerivs<TYPE>::EigenArray z =  rhovec.cwiseAbs()/rhovec.cwiseAbs().sum();
//        if (ii%100 == 0){
//            std::cout << std::real(T) << " -1 " << z.real()(0) << "," << z.real()(1) << std::endl;
//        }
//    }
//
    std::cout << "------------\n";
    CriticalIntegrator integrator(cubic_provider_factory,cubic_Cauchy_provider_factory,std::real(T0),rhovec0.real());
    ODEIntegrators::AdaptiveRK54(integrator, std::real(T0), 190.6, 1e-6, 1e6, 1e-2, 0.9);
}

void check_slope(){
    std::unique_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory("PR","Methane-ULRICH&N-Propane-ULRICH"));
    AS->set_mole_fractions({0.8,0.2});
    AS->set_binary_interaction_double(0, 1, "kij", 4.28227e-2);
    
    double z0 = 0.8, dz = 1e-6;
    double zm = z0-dz, zp = z0+dz;
    
    AS->set_mole_fractions({zm, 1-zm});
    AS->all_critical_points();
    auto pt0m = AS->all_critical_points()[0];
    
    AS->set_mole_fractions({zp, 1-zp});
    auto pt0p = AS->all_critical_points()[0];
    
    double drho1_dT = (pt0p.rhomolar*zp - pt0m.rhomolar*zm)/(pt0p.T-pt0m.T);
    double drho2_dT = (pt0p.rhomolar*(1-zp) - pt0m.rhomolar*(1-zm))/(pt0p.T-pt0m.T);
    int ttt = 9;
    
}
void check_protocol(){
    
    std::unique_ptr<CoolProp::AbstractState> AS(CoolProp::AbstractState::factory("PR","Methane-ULRICH&N-Propane-ULRICH"));
    AS->set_mole_fractions({0.8,0.2});
    AS->set_binary_interaction_double(0, 1, "kij", 4.28227e-2);
    auto pts = AS->all_critical_points();
    for (auto &pt : pts){
        std::cout << pt.T << std::endl;
    }
    AS->specify_phase(CoolProp::iphase_gas);
    AS->update(CoolProp::DmolarT_INPUTS, pts[0].rhomolar, pts[0].T);
    double alpharCP = AS->alphar();
    double PsirCP = AS->alphar()*AS->rhomolar()*AS->gas_constant()*AS->T();
    
    typedef std::complex<double> TYPE;
    // Nominally methane + propane
//    MixDerivs<TYPE>::EigenArray rhovec(2); rhovec << pts[0].rhomolar*0.8, pts[0].rhomolar*0.2;
//    TYPE T = pts[0].T;
    MixDerivs<TYPE>::EigenArray rhovec(2); rhovec << 9.344975e3, 2.336249e3;
    TYPE T = 250.65594;
    std::cout << "rhovec" << rhovec << std::endl;
    const std::unique_ptr<const AbstractCubic<TYPE> > cubic = get_cubic<TYPE>("Methane", "Propane");
    
    
    const MixDerivs<TYPE>::ProviderFactory cubic_provider_factory = [&cubic](const TYPE T, const MixDerivs<TYPE>::EigenArray& rhovec) {
        bool only_psir = false; // Everything
        return std::make_unique<CubicNativeDerivProvider<TYPE>>(cubic, T, rhovec, only_psir);
    };
    const MixDerivs<TYPE>::ProviderFactory cubic_Cauchy_provider_factory = [&cubic](const TYPE T, const MixDerivs<TYPE>::EigenArray& rhovec) {
        bool only_psir = false; // Not the rest of the stuff, just psir
        return std::make_unique<CubicNativeDerivProvider<TYPE>>(cubic, T, rhovec, only_psir);
    };
    
    MixDerivs<TYPE>::EigenArray drhovec_dT = calc_drhovec_dT_crit<TYPE>(cubic_provider_factory, cubic_Cauchy_provider_factory, T, rhovec);
    std::cout << "drhovec_dT" << drhovec_dT.real() << std::endl;
    int rr =0  ;
}

int main() {
//    test_sigma1_deriv();
//    check_slope();
//    check_protocol();
    //check_cubic<double>();
    //check_cubic<std::complex<double>>();
    
    trace();
    //check_crit_deriv();
    return EXIT_SUCCESS;
}
