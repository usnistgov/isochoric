import ctypes as ct, os, timeit, json
import matplotlib.pyplot as plt 
import numpy as np 

here = os.path.abspath(os.path.dirname(__file__))
root_path = r'c:\Program Files (x86)\REFPROP10\\'
def trim(s):
    return s.replace(b'\x00',b'').strip().decode('utf-8')

pth = os.path.join(here,'msvc','Release','ISOCHR32.dll')
assert(os.path.exists(pth))
DLL = ct.WinDLL(pth)
trace = getattr(DLL, "trace")

for forwards in [0]:
    for T in np.arange(313,501,20):
        tic = timeit.default_timer()
        Nalloc = 10002
        inputs = dict(
            imposed_variable = "T",
            imposed_value = float(T),
            fluids = "SO2&N2",
            path = root_path,
            forwards = False,
            polishing = True,
            allowable_error = 1e-5,
            unstable_termination = True,
            maximum_pressure = 100e6,
            Nallocated = Nalloc,
            # stepping_variable='STEP_IN_P',
            # hmx_path = "path/to/HMX.BNC",
            timeout = 20
        )

        # print('INPUT >>>>>>>>>>>>>>>')
        # print(json.dumps(inputs,indent=2))
        JSON_in = ct.create_string_buffer(json.dumps(inputs).encode('utf-8'), 1000)
        T = (Nalloc*ct.c_double)()
        p = (Nalloc*ct.c_double)()
        rhoL = (Nalloc*ct.c_double)()
        rhoV = (Nalloc*ct.c_double)()
        x0 = (Nalloc*ct.c_double)()
        y0 = (Nalloc*ct.c_double)()
        JSON_out = ct.create_string_buffer(1000)
        JSON_out_size = ct.c_double(1000.0)
        errcode = ct.c_double(0.0)

        trace(JSON_in,T,p,rhoL,rhoV,x0,y0,ct.byref(errcode),JSON_out,JSON_out_size)
        # print('OUTPUT >>>>>>>>>>>>>>>')
        # print(any(np.isnan(p)))
        print(list(p))
        
        out = json.loads(trim(JSON_out.raw))
        if errcode.value != 0:
            print('errcode:',errcode.value)
            print(json.dumps(json.loads(trim(JSON_out.raw)),indent=2))
            
        toc = timeit.default_timer()
        print(toc-tic,'s elapsed')
        
        if 'N' in out: 
            N = out['N']
            plt.plot(list(x0)[0:N], list(p)[0:N],'-',lw=1)
            plt.plot(list(y0)[0:N], list(p)[0:N],'-',lw=1)

plt.xlabel('$x_{1}$')
plt.ylabel('$p$ (Pa)')
plt.yscale('log')
plt.tight_layout(pad=0.2)
plt.savefig('px.pdf')
plt.show()