from __future__ import print_function

# Python imports
import os, json, timeit, math

# Packages that can be installed via pip or conda
import pandas
import matplotlib.pyplot as plt, numpy as np
import matplotlib
mpl = matplotlib
# import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go

# The pybind11 wrapper module 
import VLEIsoTracer as vle
print('VLEIsoTracer is located at:', vle.__file__)

plt.style.use('classic')
if os.path.exists('Elsevier_journals.mplstyle'):
    plt.style.use('Elsevier_journals.mplstyle')

vle.AbstractState('HEOS', '&'.join(['CO2','Ethane']))
    
# vle.apply_simple_mixing_rule('SO2','Water','linear')
# vle.apply_simple_mixing_rule('Methane','n-Hexane','linear')

def lowTerror(polishing):
    """
    Plot the error in chemical potential and pressure as we trace 
    along an isotherm of the phase envelope
    """

    R = 8.3144598
    T = 230
    fluids = ['n-Hexane','n-Octane']
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(3.5,4),sharex= True)
    backend = 'PR'

    for err, ms in [[1e-3,'^-'],[1e-9,'o-']]:
        data = None
        try:
            # Build the integrator class
            tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
            # Set flags
            tracer.set_allowable_error(err)
            tracer.polishing(polishing)
            tracer.set_debug_polishing(True)
            # Do it!
            tic = timeit.default_timer()
            tracer.trace()
            toc = timeit.default_timer()
            data = tracer.get_tracer_data()

            lbl = r'$\varepsilon_{\rm allowed}$: ' + '$10^{{{0:0.0f}}}$'.format(np.log10(err))
            # Plot the data
            x = np.array(data.x).T[0]
            perr = np.abs((np.array(data.pL) - np.array(data.pV))/np.array(data.pL))*100
            rhoL0 = np.array(data.rhoL).T[0]; rhoV0 = np.array(data.rhoV).T[0]
            chempoterr = np.abs(np.array(data.chempotr0L) - np.array(data.chempotr0V) + R*T*np.log(rhoL0/rhoV0))
            chempoterr2 = np.abs(np.array(data.chempot0L) - np.array(data.chempot0V))
            ax1.plot(x[0:-1], perr[0:-1], ms, label = lbl)
            ax2.plot(x[0:-1], chempoterr[0:-1], ms, label = lbl)
            ax2.plot(x[0:-1], chempoterr2[0:-1], ms, label = lbl)

        except BaseException as BE:
            print(BE)
        
        # print(np.array(data.x).T[0])
        # print(np.abs((np.array(data.pL) - np.array(data.pV))/np.array(data.pL))*100)
        # print(np.abs((np.array(data.chempot0L) - np.array(data.chempot0V))/np.array(data.chempot0V))*100)

        print(err, len(data.pL), toc-tic)

    ax1.legend(loc='best')
    ax1.set_xlim(0, 1)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$x_{\rm '+fluids[0]+'}$ (-)')
    ax1.set_ylabel(r'$|\Delta p/p|\times 100$ (%)')
    ax2.set_ylabel(r'$|\Delta \mu_1|$ (J/mol)')

    fig.tight_layout(pad=0.2)
    fig.savefig('_'.join(fluids)+'-polishing'+str(polishing)+'.pdf')
    plt.show()

def tocritline():
    """
    Plot values (determinant of Hessian and others) along a high-temperature isotherm
    """
    T = 300
    fluids = ['CO2','Ethane']

    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(3.5,3.5),sharex = True)
    fig3, ax3 = plt.subplots(1,1,figsize=(3.5,2.5),sharex = True)
    fig4, ax4 = plt.subplots(1,1,figsize=(3.5,2.5),sharex = True)
    backend = 'HEOS'
    try:
        tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
        tracer.trace()
        data = tracer.get_tracer_data()
        xval = np.array(data.rhoL).T[0]
        line, = ax1.plot(xval, np.abs(data.det_PSIL), label = 'liquid')
        ax1.plot(xval, np.abs(data.det_PSIV), dashes=[2,2], label='vapor')

        xval = np.array(data.rhoL).T[0][0:-1]
        yval = np.diff(np.array(data.rhoL).T[0])
        ax2.plot(xval, yval, 'r')

        rho1 = np.array(data.rhoL).T[0]
        ax3.plot(list(range(len(rho1))), rho1/np.max(rho1))

        ax4.plot(np.array(data.rhoL).T[0], np.array(data.rhoL).T[1])

    except BaseException as BE:
        print(BE)
        pass

    for ax in ax1,ax2:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1e-6,1e-4,1e-2,1e0,1e2,1e4])
    ax1.legend(loc='best')

    ax1.set_ylabel(r'$\det(\mathbf{H}_{\Psi})$')
    ax2.set_ylabel(r'$h$ (mol/m$^3$)')
    ax2.set_xlabel(r"$\rho_1'$ (mol/m$^3$)")
    ax1.set_yticks([1e-2,1e0,1e2,1e4,1e6,1e8])
    ax2.set_yticks([1e-8, 1e-6,1e-4,1e-2,1e0,1e2])

    ax3.set_xlabel(r'Step index')
    ax3.set_ylabel(r"$\rho_1'/{\rm max}(\rho_1')$ (-)")
    ax3.set_ylim(0, 1.01)

    fig.tight_layout(pad=0.2)
    fig.savefig('_'.join(fluids)+'-'+str(T)+'Ktocritline.pdf')

    fig3.tight_layout(pad=0.2)
    fig3.savefig('_'.join(fluids)+'-'+str(T)+'Ktocritline-progress.pdf')

    fig4.tight_layout(pad=0.2)
    fig4.savefig('_'.join(fluids)+'-'+str(T)+'Ktocritline-debug.pdf')

    plt.close('all')

def CO2ethaneconcentrations():
    T = 285
    fluids = ['CO2','Ethane']
    fig4, ax4 = plt.subplots(1,1,figsize=(3.5,2.5),sharex = True)
    backend = 'HEOS'
    try:
        tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
        # Set flags
        tracer.set_allowable_error(1e-6)
        # tracer.polishing(polishing)
        tracer.set_debug_polishing(True)
        tracer.set_forwards_integration(True)
        tracer.set_unstable_termination(False)
        # tracer.set_stepping_variable(vle.VLEIsolineTracer.stepping_variable.STEP_IN_RHO1)
        tracer.trace()
        data = tracer.get_tracer_data()
        x = np.array(data.x).T[0]
        rhoLmat = np.array(data.rhoL)
        rhoVmat = np.array(data.rhoV)
        ax4.plot(x, rhoLmat[:,0],color='b')
        ax4.plot(x, rhoLmat[:,1],color='r')
        ax4.plot(x, rhoVmat[:,0],dashes = [2,2], color='b')
        ax4.plot(x, rhoVmat[:,1],dashes = [2,2], color='r')
        print(tracer.get_termination_reason())

    except BaseException as BE:
        print(BE)
        pass

    ax4.set_xlabel(r'x / molar')
    ax4.set_ylabel(r"$\rho'$,$\rho''$  / mol/m$^3$")
    # ax4.set_ylim(0, 1.01)

    fig4.tight_layout(pad=0.2)
    fig4.savefig('_'.join(fluids)+'-'+str(T)+'conc.pdf')

    plt.close('all')

def MethaneEthane():
    fluids = ['Methane','Ethane']
    T = vle.Props1SI('Tcrit','Methane')-1
    fig1, ax1 = plt.subplots(1,1,figsize=(3.5,2.5),sharex = True)
    fig4, ax4 = plt.subplots(1,1,figsize=(3.5,2.5),sharex = True)
    backend = 'HEOS'
    try:
        tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
        # Set flags
        tracer.set_allowable_error(1e-6)
        # tracer.polishing(polishing)
        tracer.set_debug_polishing(True)
        tracer.set_forwards_integration(True)
        tracer.set_unstable_termination(False)
        tracer.set_stepping_variable(vle.VLEIsolineTracer.stepping_variable.STEP_IN_RHO1)
        tracer.trace()
        data = tracer.get_tracer_data()
        x = np.array(data.x).T[0]
        y = np.array(data.y).T[0]
        pL = np.array(data.pL)/1e6
        rhoLmat = np.array(data.rhoL)
        rhoVmat = np.array(data.rhoV)
        ax4.plot(x, rhoLmat[:,0],color='b')
        ax4.plot(x, rhoLmat[:,1],color='r')
        ax4.plot(x, rhoVmat[:,0],dashes = [2,2], color='b')
        ax4.plot(x, rhoVmat[:,1],dashes = [2,2], color='r')
        print(tracer.get_termination_reason())
        ax1.plot(x, pL)
        ax1.plot(y, pL)

    except BaseException as BE:
        print(BE)
        pass

    ax1.set_xlabel(r'x / molar')
    ax1.set_ylabel(r"$p$  / MPa")
    ax4.set_xlabel(r'x / molar')
    ax4.set_ylabel(r"$\rho'$,$\rho''$  / mol/m$^3$")

    fig1.tight_layout(pad=0.2)
    fig1.savefig('_'.join(fluids)+'-'+str(T)+'px.pdf')
    fig4.tight_layout(pad=0.2)
    fig4.savefig('_'.join(fluids)+'-'+str(T)+'conc.pdf')

    plt.close('all')

def speedtest():
    """
    Time some results from the isoline tracing, and compare with algebraic solver
    """
    fluids = ['CO2','Ethane']    
    fig, (ax1) = plt.subplots(1,1,figsize=(3.5,2.5),sharex = True)

    lib = []
    for T in np.arange(260, 304):
        time_trace = np.nan; alg_fail = True; time_alg = np.nan
        print('--')
        for backend in ['HEOS']:
            for polishing in [False, True]:
                tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
                tracer.set_allowable_error(1e-4)
                tracer.polishing(polishing)
                try:
                    tracer.trace()
                    data = tracer.get_tracer_data()

                    print(backend, T, polishing, len(data.pL), tracer.get_tracing_time(), tracer.get_termination_reason())

                    if backend == 'HEOS' and polishing:
                        time_trace = tracer.get_tracing_time()

                        xx = np.array(data.x).T
                        yy = np.array(data.y).T

                        AS = vle.AbstractState('HEOS', 'CO2&Ethane')
                        tic = timeit.default_timer()
                        bad = 0
                        for i in range(xx.shape[1]):
                            AS.set_mole_fractions(xx[:,1])
                            try:
                                AS.update(vle.QT_INPUTS, 0, T)
                            except:
                                bad +=1
                        toc = timeit.default_timer()
                        print(T, 'FLSH-HEOS', toc-tic, len(data.pL), bad)

                        # "Trace" with the algebraic solver
                        rhoL = data.rhoL[0]; rhoV = data.rhoV[0]
                        rhostart = data.rhoL[0][0]; rhoend = data.rhoL[-1][0]
                        for el in rhoV: rhoL.append(el)
                        alg_tracer = vle.AlgebraicIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
                        alg_tracer.logging = True
                        try:
                            tic = timeit.default_timer()
                            alg_tracer.trace_rho0(rhoL, np.linspace(rhostart, rhoend, 100))
                            toc = timeit.default_timer()
                            alg_fail = False
                            time_alg = toc-tic
                        except BaseException as BE:
                            pass
                        print(T, 'ALG', toc-tic, len(alg_tracer.p))

                except BaseException as BE:
                    print(BE)
                    pass
        lib.append(dict(Temp=T, time_alg=time_alg, time_trace = time_trace, alg_fail = alg_fail))
    df = pandas.DataFrame(lib)
    plt.plot(df.Temp, df.time_trace, 'ko', ms = 5, label = 'Isochoric tracer')
    plt.plot(df.Temp, df.time_alg, 'b^', ms = 5, label = 'Algebraic tracer')
    for index, row in df.iterrows():
        if row.alg_fail:
            plt.plot(row.Temp, row.time_trace, 'rx', ms = 12, lw = 3)
        else:
            # plt.plot(row.T, row.time_trace, '')
            pass
    plt.yscale('log')
    plt.legend(loc='best', numpoints = 1)
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlabel('$T$ (K)')
    plt.ylabel('$t$ (s)')
    fig.tight_layout(pad=0.2)
    plt.savefig('speedtest.pdf')
    plt.close()

def lowTSO2Water():
    T = 400
    fluids = ['SO2','Water']
    fig, ax = plt.subplots(1,1,figsize=(3.5,2.2))
    lbls = {'PR':'Peng-Robinson','HEOS':'Multi-fluid'}
    lw = {'PR':0,'HEOS':0}

    for backend in ['HEOS']:
        try:
            tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
            tracer.set_forwards_integration(True)
            tracer.set_unstable_termination(False)
            tracer.set_stepping_variable(vle.VLEIsolineTracer.stepping_variable.STEP_IN_RHO1)
            if backend == 'HEOS':
                AS = tracer.get_AbstractState_pointer()                
                for k,v in zip(['betaT','gammaT','betaV','gammaV'],[1.019562, 0.916311, 1.094032, 0.962547]):
                    AS.set_binary_interaction_double(0,1,k,v)
            tracer.trace()
            reason = tracer.get_termination_reason()
            if reason: 
                print(reason)
            data = tracer.get_tracer_data()

            # AS = tracer.get_AbstractState_pointer()
            # for x0 in np.arange(0.3, 0.6, 0.025):
               #  AS.set_mole_fractions([x0,1-x0])
               #  pts = AS.all_critical_points()
               #  for pt in pts:
               #    print(x0, pt.T, pt.p/1e6, pt.rhomolar)
               #    plt.plot(x0, pt.p/1e6,'o')
            col = mpl.cm.jet
            cpmax = np.max(np.array(data.chempot0L))
            cpmin = np.min(np.array(data.chempot0L))

            # p-x plot
            stable_mask = np.array(data.det_PSIL).T > 0
            for mask, marker in zip([stable_mask, ~stable_mask],['o', '*']):
                chempot = np.array(data.chempot0L)[mask]
                color = (chempot-cpmin)/(cpmax-cpmin)
                # print(color)
                x = 1-np.array(data.x).T[0][mask]
                y = (np.array(data.pL)/1e6)[mask]
                c = chempot
                line = ax.scatter(x, y, label = lbls[backend], c = color, cmap = plt.cm.jet, edgecolor = 'w')
                ax.scatter(1-np.array(data.y).T[0][mask], (np.array(data.pL)/1e6)[mask], lw = lw[backend], c = color, cmap = plt.cm.jet, edgecolor = 'w')

        except BaseException as BE:
            print(BE)
            raise

    ax.set_xlim(0, 1)
    # ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlabel(r'$x_{\rm CO_{2}}$ (-)')
    ax.set_ylabel('$p$ (MPa)')

    fig.tight_layout(pad=0.2)
    fig.savefig('_'.join(fluids)+'-'+str(T)+'K.pdf')
    plt.show()

def calc_crits(backend, fluids, force = False):
    fname = backend + '-'.join(fluids)
    if not os.path.exists(fname) or force:
        print('rebuilding critical line')
        x, pc, Tc = [], [], []
        for x0 in np.linspace(1e-5, 1-1e-5, 101):
            AS = vle.AbstractState(backend, '&'.join(fluids))
            AS.set_mole_fractions([x0, 1-x0])
            try:
                pts = AS.all_critical_points()
                if len(pts) > 0:
                    Tc.append(pts[0].T)
                    pc.append(pts[0].p)
                    x.append(x0)
            except BaseException as BE:
                print(x0, BE)
        with open(fname,'w') as fp:
            fp.write(json.dumps(dict(x = x, Tc = Tc, pc = pc, fluids = fluids, backend = backend)))
    with open(fname,'r') as fp:
        return json.load(fp)

def plot_isolines(fluids, Tvec, pvec, backend = 'HEOS', only_px = False):

    print(fluids)
    if only_px:
        fig, ax1 = plt.subplots(1,1,figsize=(3.5,2.5))
    else:
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(6.5,2.5))
    fig2, ax3 = plt.subplots(1,1)

    for T in Tvec:
        try:
            tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
            tracer.set_allowable_error(1e-8)
            tracer.polishing(True)
            tracer.trace()
            data = tracer.get_tracer_data()

            # p-x plot
            line, = ax1.plot(np.array(data.x).T[0], data.pL, label = str(T), lw = 0.5)
            ax1.plot(np.array(data.y).T[0], data.pL, color = line.get_color(), lw = 0.5)
            ax3.plot(np.array(data.x).T[0], np.array(data.rhoL), label = str(T))
            print('ISOT:', T, 'K;', len(data.pL), 'points')
            reason = tracer.get_termination_reason()
            if reason: 
                print(reason)

            # If we didn't get to the end, see if we can do the backwards integration
            if np.max(np.array(data.x).T[0]) < 0.99:
                # Stop if the temperature is above the critical point of the pure fluid
                # In that case you cannot do the backwards calculation
                if T > vle.Props1SI('Tcrit',fluids[0]): 
                    continue
                tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
                tracer.set_forwards_integration(False)
                tracer.trace()
                data = tracer.get_tracer_data()
                ax1.plot(np.array(data.x).T[0], data.pL, color = line.get_color(), lw = 0.5)
                ax1.plot(np.array(data.y).T[0], data.pV, color = line.get_color(), lw = 0.5)
                print('ISOT backwards',T, len(data.pL))
        except BaseException as BE:
            print(T, BE)

    # You could also modify this block to do a two-part integration with forwards and backwards parts
    # For my purposes this wasn't necessary
    if not only_px:
        for p in pvec:
            if p > vle.Props1SI('Pcrit', backend+'::'+fluids[1]):
                forwards = False
            else:
                forwards = True
            try:
                tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_P, p, backend, fluids)
                tracer.set_forwards_integration(forwards)
                tracer.trace()
                data = tracer.get_tracer_data()

                # T-x plot
                line, = ax2.plot(np.array(data.x).T[0], data.TL, label = '', lw = 0.5)
                ax2.plot(np.array(data.y).T[0], data.TV, color = line.get_color(), lw = 0.5)

                print('ISOP:', p, 'Pa;', len(data.pL),'points')
            except BaseException as BE:
                print(p, BE)

    # Critical line of the mixture
    crit = calc_crits(backend, fluids)
    x, Tc, pc = crit['x'],crit['Tc'],crit['pc']
    x += [1,0]
    Tc += [vle.Props1SI('Tcrit',f) for f in fluids]
    pc += [vle.Props1SI('pcrit',f) for f in fluids]
    x, Tc, pc = zip(*(sorted(zip(x, Tc, pc))))
    for lw, sty in [[1.0,'k-']]:
        ax1.plot(x, pc, sty, lw = lw)
        if not only_px: ax2.plot(x, Tc, sty, lw = lw)

    # plt.legend(loc='best')
    if only_px: 
        axes = [ax1]
    else:   
        axes = [ax1,ax2]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xlabel(r'$x_{{\rm {fld:s} }}, y_{{\rm {fld:s} }}$ (-)'.format(fld=fluids[0]).replace(r'\rm CO2',r'{\rm CO_2}').replace(r'\rm n-Hexane',r'{n-{\rm Hexane}}') )

    ax1.set_yscale('log')
    ax1.set_ylabel('$p$ (Pa)')
    if not only_px: ax2.set_ylabel('$T$ (K)')

    if backend == 'PR' and fluids == ['Methane','n-Propane']:
        ax.set_ylim(ymin=1e6, ymax=1.5e7)

    if 'CO2' in fluids[0]:
        print('setting labels')
        ticks = [1e6,2e6,4e6,6e6,8e6,1e7]
        
        ax1.set_yticks(ticks)
        lbls = []
        for t in ticks:
            e = int(math.floor(math.log10(t)))
            c = int(t/10**e)
            print(c, e)
            l = r'${c:d}\times 10^{{{e:d}}}$'.format(c=c,e=e)
            lbls.append(l)
        ax1.set_yticklabels(lbls)

    fig.tight_layout(pad=0.2)
    fig.savefig(backend+'_'.join(fluids)+'.pdf')
    plt.show()

def plotly_surface(fluids, Tvec, pvec):
    """
    Construct a 3D set of lines forming the phase envelope 
    with the tracer we developed and the use of the plotly 
    plotting library
    """

    backend = 'HEOS'
    
    data = []
    def add_trace(T,comp,p,linecolor):
        trace = go.Scatter3d(
            mode='lines',
            name=None,
            showlegend=False,
            x=T,
            y=comp,
            z=p, 
            marker=dict(
                size=0,
                #color=z,
                colorscale='Viridis',
            ),
            line=dict(
                color=linecolor,
                width=5
            )
        )
        data.append(trace)

    for T in Tvec:
        try:
            tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
            tracer.trace()
            _data = tracer.get_tracer_data()

            for comp in [np.array(_data.x).T[0].tolist(),
                         np.array(_data.y).T[0].tolist()]:
                add_trace(_data.TL, comp, _data.pL, linecolor = '#000000')

            if np.max(np.array(_data.x).T[0]) < 0.99:
                tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_T, T, backend, fluids)
                tracer.set_forwards_integration(False)
                tracer.trace()
                _data = tracer.get_tracer_data()
                for comp in [np.array(_data.x).T[0].tolist(),
                             np.array(_data.y).T[0].tolist()]:
                    add_trace(_data.TL, comp, _data.pL, linecolor = '#000000')

        except BaseException as BE:
            print(BE)

    for p in pvec:
        try:
            tracer = vle.VLEIsolineTracer(vle.VLEIsolineTracer.imposed_variable.IMPOSED_P, p, backend, fluids)
            tracer.trace()
            _data = tracer.get_tracer_data()
            for comp in [np.array(_data.x).T[0].tolist(),
                         np.array(_data.y).T[0].tolist()]:
                add_trace(_data.TL, comp, _data.pL, linecolor = '#ff0000')
        except BaseException as BE:
            print(BE)

    # Critical line of the mixture
    crit = calc_crits(backend, fluids)
    x, Tc, pc = crit['x'],crit['Tc'],crit['pc']
    x += [1,0]
    Tc += [vle.Props1SI('Tcrit',f) for f in fluids]
    pc += [vle.Props1SI('pcrit',f) for f in fluids]
    x, Tc, pc = zip(*(sorted(zip(x, Tc, pc))))
    add_trace(Tc, x, pc, linecolor = '#ff00ff')

    layout = dict(
        width=1200,
        height=1000,
        autosize=False,
        font= {
            'family': 'Times New Roman',
            'size': 20,
            'color': '#7f7f7f'
        },
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=4
        ),
        # title='Phase envelope',
        scene=dict(
            xaxis=dict(
                title='T (K)',
                type='linear',
                gridwidth=4,
            ),
            yaxis=dict(
                title='x, y',
                type='linear',
                gridwidth=4,
            ),
            zaxis=dict(
                title='p (Pa)',
                type='log',
                gridwidth=4,
            ),
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=-1.7428,
                    y=1.0707,
                    z=0.7100,
                )
            ),
            aspectratio = dict( x=1, y=1, z=0.7 ),
            aspectmode = 'manual'
        ),
    )
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename='phase_envelope-'+'-'.join(fluids)+'.html', validate=True)

if __name__=='__main__':
    vle.PropsSI('Dmolar','T',230,'Q',0,'n-Hexane')
    lowTerror(polishing = True)
    lowTerror(polishing = False)
    # lowTSO2Water()
    # speedtest()
    # plotly_surface(['CO2','Ethane'], np.arange(250, 310, 5), np.logspace(np.log10(2e6),np.log10(1e7),20))
    # plotly_surface(['n-Hexane','n-Octane'], np.arange(300, 600, 20), np.logspace(np.log10(1e4),np.log10(1e7),20))
    # CO2ethaneconcentrations()
    # MethaneEthane()
    # tocritline()
    # for fluids, Tvec, pvec, backend, only_px in [
    #     # (['CO2','Ethane'], np.arange(250, 310, 5), np.logspace(np.log10(1e5),np.log10(1e8),30),'HEOS', False),
    #     (['Methane','Ethane'], np.arange(150, 325, 5), np.logspace(np.log10(1e5),np.log10(1e8),30),'HEOS', True),
    #     # (['n-Hexane','n-Octane'], np.linspace(300, 600, 20), np.logspace(np.log10(1e4),np.log10(1e7),20),'HEOS',False),
    #     # (['Methane','n-Propane'], np.arange(200, 350, 10), np.logspace(np.log10(1e4),np.log10(1e7),30),'PR', True)
    #     ]:
    #     plot_isolines(fluids, Tvec, pvec, backend, only_px = only_px)