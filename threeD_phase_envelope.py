"""
Python 3+ only
"""
import os, json, time, sys

import pandas, scipy.interpolate
import matplotlib.pyplot as plt, numpy as np

import VLEIsoTracer as vle

CC = vle.AbstractState('HEOS','Ethane&n-Octane')
CC = vle.AbstractState('HEOS','n-Propane&n-Octane')

header = """module line(p1,p2,r) {
    /*translate(p1){
        dx = p2[0]-p1[0];
        dy = p2[1]-p1[1];
        dz = p2[2]-p1[2];
        length = norm([dx,dy,dz]);  // radial distance
        b = acos(dz/length); // inclination angle
        c = atan2(dy,dx);     // azimuthal angle

        rotate([0, b, c]) 
            cylinder(h=length, r=r, $fn=30);
    }
    translate(p1)
        sphere(r,$fn=10);
    translate(p2)
        sphere(r,$fn=10);*/

    hull(){
        translate(p1)
            sphere(r,$fn=10);
        translate(p2)
            sphere(r,$fn=10);
    }
 }

line_radius = 0.13;
"""

def calc_crits(backend, fluids, force = False):
    fname = backend + '-'.join(fluids)
    if not os.path.exists(fname) or force:
        print('rebuilding critical line')
        x, pc, Tc = [0], [vle.PropsSI('pcrit',fluids[1])], [vle.PropsSI('Tcrit',fluids[1])]
        for x0 in np.linspace(1e-5, 1-1e-5, 40):
            AS = vle.AbstractState(backend, '&'.join(fluids))
            AS.set_mole_fractions([x0,1-x0])
            try:
                pts = AS.all_critical_points()
                if len(pts) > 0 and pts[0].T < 1000:
                    Tc.append(pts[0].T)
                    pc.append(pts[0].p)
                    x.append(x0)
            except BaseException as BE:
                print(BE)
        x.append(1); Tc.append(vle.PropsSI('Tcrit',fluids[0])); pc.append(vle.PropsSI('pcrit',fluids[0]))
        with open(fname,'w') as fp:
            fp.write(json.dumps(dict(x = x, Tc = Tc, pc = pc, fluids = fluids, backend = backend)))
    with open(fname,'r') as fp:
        return json.load(fp)

class TraceLibrary(object):
    def __init__(self):
        self.traces = []

    def add_trace(self, *, imposed, T, x, p, data):
        self.traces.append(dict(imposed=imposed,T=T,x=x,log10p = np.log10(p),data=data))

    def get_minmax(self):
        mins,maxs = {},{}
        for k in ['T', 'x', 'log10p']:
            minvals = []
            maxvals = []
            for trace in self.traces:
                minvals.append(np.min(trace[k]))
                maxvals.append(np.max(trace[k]))
            mins[k] = np.min(minvals)
            maxs[k] = np.max(maxvals)
        return mins, maxs

    def get_lines(self):
        mins, maxs = self.get_minmax()
        lines = []
        for i, trace in enumerate(self.traces):

            xnorm = (trace['T']-mins['T'])/(maxs['T']-mins['T'])
            ynorm = (trace['x']-mins['x'])/(maxs['x']-mins['x'])
            znorm = (trace['log10p']-mins['log10p'])/(maxs['log10p']-mins['log10p'])

            if len(xnorm) < 2:
                continue

            # Arclength interpolation factor
            dt = [0] + np.sqrt(np.diff(xnorm)**2 + np.diff(ynorm)**2 + np.diff(znorm)**2).tolist() # Lengths of each contribution
            t = np.cumsum(dt)

            tt = np.linspace(0, np.max(t), 20)
            xnorm = scipy.interpolate.interp1d(t,xnorm,'linear')(tt)
            ynorm = scipy.interpolate.interp1d(t,ynorm,'linear')(tt)
            znorm = scipy.interpolate.interp1d(t,znorm,'linear')(tt)

            # plt.plot(t, xnorm)
            # plt.plot(t, ynorm)
            # plt.plot(t, znorm)
            # plt.show()

            scale_x = 10
            scale_y = 10
            scale_z = 10

            def build_line(i):
                p1 = [xnorm[i]*scale_x, ynorm[i]*scale_y, znorm[i]*scale_z]
                p2 = [xnorm[i+1]*scale_x, ynorm[i+1]*scale_y, znorm[i+1]*scale_z]
                return 'line([{0:g},{1:g},{2:g}],[{3:g},{4:g},{5:g}],line_radius);'.format(*(p1+p2))

            for j in range(len(xnorm)-1):
                lines.append(build_line(j))

        return lines

    def get_lowest_isotherm(self):
        Tmin = 1e100
        iTmin = -1
        for i, trace in enumerate(self.traces):
            if trace['imposed'] == 'T':
                T = trace['T'][0]
                if T < Tmin:
                    Tmin = T
                    iTmin = i
        return self.traces[iTmin]

    def get_highest_isobar(self):
        pmax = 1e100
        iTmax = -1
        for i, trace in enumerate(self.traces):
            if trace['imposed'] == 'p':
                p = trace['p'][0]
                if p > pmax:
                    pmax = p
                    iTmax = i
        return self.traces[iTmax]

    def to_json(self, ofname):

        trace_dump = []
        for trace in self.traces:
            T,x,log10p = [np.array(_).tolist() for _ in (trace['T'],trace['x'],trace['log10p'])]
            trace_dump.append(dict(imposed=trace['imposed'],T=T,x=x,log10p=log10p))
        with open(ofname, 'w') as fp:
            fp.write(json.dumps(trace_dump, indent=2))

    def to_SCAD(self, ofname):
        with open(ofname, 'w') as fp:
            fp.write(header)
            fp.write('\n'.join(self.get_lines()))

def SCAD_3D(*, TracerClass, fluids, Tvec, pvec=[], pvec_lowestT=[], Tvec_highestp=[], model_setter=None, HMX = [], pmax=100e6):

    library = TraceLibrary()

    class SuperTTracer(TracerClass):
        def __init__(self, T, backend, fluids, p0, rhovec0, lims, *args):
            print(rhovec0, lims)
            TracerClass.__init__(self, TracerClass.imposed_variable.IMPOSED_T, T, backend, fluids, *args)
            self.p0 = p0
            self.T0 = T
            self.rhovec0 = rhovec0
            self.lims = lims
            print('T',rhovec0, lims)

        def get_integration_limits(self):
            return self.lims

        def calc_initial_state(self):
            init = vle.InitialState()
            init.T = self.T0
            init.rhovecL = self.rhovec0[0:2]
            init.rhovecV = self.rhovec0[2::]
            return init

        def determine_integration_type(self):
            return TracerClass.stepping_variable.STEP_IN_RHO0

    class SuperPTracer(TracerClass):
        def __init__(self, P, backend, fluids, T0, rhovec0, lims, *args):
            TracerClass.__init__(self, TracerClass.imposed_variable.IMPOSED_P, P, backend, fluids, *args)
            self.T0 = T0
            self.rhovec0 = rhovec0
            self.lims = lims[::-1]
            print('P',rhovec0, lims)

        def get_integration_limits(self):
            return self.lims

        def calc_initial_state(self):
            init = vle.InitialState()
            init.T = self.T0
            init.rhovecV = self.rhovec0[0:2]
            init.rhovecL = self.rhovec0[2::]
            return init

        def determine_integration_type(self):
            return TracerClass.stepping_variable.STEP_IN_RHO0

    backend = 'HEOS' # This is ignored for REFPROP wrapper class
    tracer = TracerClass(TracerClass.imposed_variable.IMPOSED_T, Tvec[0], backend, fluids, *HMX)
    print(tracer.get_binary_interaction_double(0,1,'betaT'))

    # Build the isotherms
    for T in Tvec:
        print('Tracing T=',T,'K')
        for forwards in False, True:
            try:
                tracer = TracerClass(TracerClass.imposed_variable.IMPOSED_T, T, backend, fluids, *HMX)
                tracer.set_forwards_integration(forwards)
                tracer.set_maximum_pressure(pmax)
                tracer.polishing(True)
                tracer.trace()
                if model_setter is not None:
                    model_setter(tracer.get_AbstractState_pointer())

                _data = tracer.get_tracer_data()

                for Q, comp in [(0,np.array(_data.x).T[0].tolist()),
                                (1,np.array(_data.y).T[0].tolist())
                                ]:
                    library.add_trace(imposed='T',T=_data.TL, x=comp, p=_data.pL, data=_data)
                break

            except BaseException as BE:
                print(T, forwards, BE)

    # Build the isobars
    for p in pvec:
        # print(p)
        for forwards in False, True:
            try:
                tracer = TracerClass(TracerClass.imposed_variable.IMPOSED_P, p, backend, fluids, *HMX)
                tracer.set_forwards_integration(forwards)
                tracer.set_stepping_variable(TracerClass.stepping_variable.STEP_IN_RHO0)
                tracer.polishing(True)
                tracer.trace()
                if model_setter is not None:
                    model_setter(tracer.get_AbstractState_pointer())

                _data = tracer.get_tracer_data()
                # print(_data.TL, _data.rhoL)

                for Q, comp in [(0,np.array(_data.x).T[0].tolist()),
                                (1,np.array(_data.y).T[0].tolist())
                                ]:
                    library.add_trace(imposed='P',T=_data.TL, x=comp, p=_data.pL, data=_data)
                    print(p)
                break

            except BaseException as BE:
                print(p, forwards, BE)

    # Saturation curves
    for i in [0, 1]:
        fld = fluids[1-i]
        Tt = vle.Props1SI('Ttriple', fld)
        Tc = vle.Props1SI('Tcrit', fld)
        if not Tvec.tolist() or Tc < np.min(Tvec): continue
        x,T,p =[],[],[]
        AS = vle.AbstractState(backend, fld)
        Tmin = max(np.min(Tvec), Tt)
        for _T in np.linspace(Tmin, Tc):
            AS.update(vle.input_pairs.QT_INPUTS, float(i), _T)
            x.append(i)
            T.append(_T)
            p.append(AS.p())
        library.add_trace(imposed='SAT', T=T, x=x, p=p,data=None)

    # Fill in the isobars along the lowest temperature isotherm
    lowest_isotherm = library.get_lowest_isotherm()
    data = lowest_isotherm['data']
    for p in pvec_lowestT:
        # The starting T is specified since we are on the lowest T isotherm
        T = lowest_isotherm['T'][0]
        # Interpolate to find the molar concentrations of interest at the specified pressure
        rhoLmat = np.array(data.rhoL)
        rhoVmat = np.array(data.rhoV)
        rhovecL = [0, 0]
        rhovecV = [0, 0]

        try:
            for i in range(2):
                rhovecL[i] = float(scipy.interpolate.interp1d(data.pL, rhoLmat[:,i])(p))
                rhovecV[i] = float(scipy.interpolate.interp1d(data.pV, rhoVmat[:,i])(p))
            rhovec0 = rhovecL + rhovecV # starting values
            limits = [rhovecL[0], rhovecV[0]] # limits of integration
            ptrace = SuperPTracer(p, backend, fluids, T, rhovec0, limits, *HMX)
            # ptrace.set_debug_polishing(True)
            # if model_setter is not None:
            #     model_setter(ptrace.get_AbstractState_pointer())
            # ptrace.polishing(False)
            ptrace.trace()
            _data = ptrace.get_tracer_data()
            for Q, comp in [(0,np.array(_data.x).T[0].tolist()),
                            (1,np.array(_data.y).T[0].tolist())
                            ]:
                library.add_trace(imposed='P',T=_data.TL, x=comp, p=_data.pL, data=_data)
        except BaseException as BE:
            print(BE)

    # Now at the highest pressure, we fill in the upper branches of some isotherms
    highest_isobar = library.get_highest_isobar()
    data = highest_isobar['data']
    for T in Tvec_highestp:
        # Interpolate to find the molar concentrations of interest at the specified temperature
        rhoLmat = np.array(data.rhoL)
        rhoVmat = np.array(data.rhoV)
        rhovecL = [0, 0]
        rhovecV = [0, 0]
        try:
            for i in range(2):
                rhovecL[i] = float(scipy.interpolate.interp1d(data.TL, rhoLmat[:,i])(T))
                rhovecV[i] = float(scipy.interpolate.interp1d(data.TV, rhoVmat[:,i])(T))
            rhovec0 = rhovecL + rhovecV # starting values
            limits = [rhovecL[0], rhovecV[0]] # limits of integration
            Ttrace = SuperTTracer(T, backend, fluids, T, rhovec0, limits, *HMX)
            # ptrace.set_debug_polishing(True)
            # if model_setter is not None:
            #     model_setter(ptrace.get_AbstractState_pointer())
            # ptrace.polishing(False)
            Ttrace.trace()
            _data = Ttrace.get_tracer_data()
            for Q, comp in [(0,np.array(_data.x).T[0].tolist()),
                            (1,np.array(_data.y).T[0].tolist())
                            ]:
                library.add_trace(imposed='T',T=_data.TL, x=comp, p=_data.pL, data=_data)
        except BaseException as BE:
            print(BE)

    library.to_json('trace_dump.json')
    library.to_SCAD('_'.join(fluids)+'.scad')    

def threeDplot():
    from mpl_toolkits.mplot3d import Axes3D 
    import matplotlib as mpl

    with mpl.rc_context({"font.family":"Times New Roman", "font.size": 10, "mathtext.fontset": "dejavuserif"}):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(6,4))
        for trace in json.load(open('trace_dump.json')):
            x,y,z = trace['T'], trace['x'], np.array(trace['log10p'])
            # for v in [x,y,z]:
            #     print(np.min(v), np.max(v))
            ax.plot(x,y,z,color='k' if trace['imposed'] == 'T' else 'r')
        ax.set_xlabel(r'T / K')
        ax.set_ylabel(r'$x_{\rm SO_{2}}$')
        ax.set_zlabel(r'$\log_{10}(p/Pa)$')
        # ax.set_zticks(np.arange(6,8.1,0.5))
        # fig.tight_layout(pad=0.2)
        plt.savefig('TPX_SO2_N2.pdf')
        plt.show()

def plot_propane_octane():
    SCAD_3D(
        TracerClass=vle.VLEIsolineTracer,
        fluids=['n-Propane', 'n-Octane'], 
        Tvec=np.arange(250, vle.Props1SI('Tcrit','n-Octane'), 16),
        pvec=np.logspace(np.log10(1.3e5), np.log10(3e6), 20)
    )

def plot_SO2_N2():
    CAS_SO2, CAS_N2 = [vle.get_fluid_param_string(f, 'CAS') for f in ['SO2','N2']]
    vle.apply_simple_mixing_rule(CAS_SO2, CAS_N2, 'linear') # placeholder only
    for k, v in zip(['betaT','gammaT','betaV','gammaV'],[1.045874,1.1946588,0.9036245,1.2155808]):
        vle.set_mixture_binary_pair_data(CAS_SO2, CAS_N2, k, v)

    # ####### See also Tsiklis, 1947
    Tvec = np.arange(315, vle.Props1SI('Tcrit','SO2'), 20)
    pvec = np.logspace(np.log10(1e6), np.log10(100e6), 20)
    SCAD_3D(
        # TracerClass=vle.VLEIsolineTracer,
        TracerClass=vle.REFPROPIsolineTracer,
        fluids=['SO2','Nitrogen'], 
        Tvec=Tvec,
        pvec_lowestT=pvec,
        Tvec_highestp=Tvec
    )

def plot_N2_NH3():
    Tvec = np.arange(315, 500, 10)
    pvec = np.logspace(np.log10(1e6), np.log10(100e6), 40)
    SCAD_3D(
        TracerClass=vle.REFPROPIsolineTracer,
        fluids=['N2', 'NH3'], 
        Tvec=Tvec,
        pvec_lowestT=pvec,
        Tvec_highestp=Tvec,
        pmax=100e7,
        HMX = [os.path.join(os.path.abspath('..'), 'HMX.BNC')]
    )

if __name__=='__main__':
    vle.load_REFPROP("D:/Program Files (x86)/REFPROP")
    # plot_propane_octane()
    plot_SO2_N2()
    # plot_N2_NH3()
    threeDplot()