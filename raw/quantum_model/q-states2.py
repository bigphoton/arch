"""
JCA 2022

"""
import time

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

try:
    import colored_traceback.auto
except ImportError:
    pass
    
import abc
import arch.port as port
from arch.port import var, print_state
from arch.block import Block
from arch.connectivity import Connectivity
from arch.models import Model, SymbolicModel, NumericModel, SourceModel
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder, Waveguide
from arch.blocks.sources import LaserCW
from arch.blocks.qontrol import Qontrol
from arch.blocks.detectors import BasicSPD, PhotoDiode
from arch.blocks.wire import Wire
from arch.architecture import Architecture
from arch.simulations import length_to_ps, ps_to_length, ng_fibre, ng_siwg, ng_TL, get_delay_map, BasicDynamicalSimulator
import networkx as nx


# from arch.raw.quantum_model.quantum_model import *



if __name__=='__main__':

    # def forig(x):
        # if x < 0 and -1 <= x:
            # return 1.0
        # if -1 <= x and x < 1:
            # return 1/np.sqrt(2) 
        # else:
            # return 0
            
            

    # st = ChronocyclicState(func_t=forig, tc=0, dt=0.01, df=0.1, nonzero_width_t=4.0)

    # print(f"Initial: t={st.integral_t()}, t2={st.integral_t2()}")
    # print(f"Initial: f={st.integral_f()}, f2={st.integral_f2()}")
     # abs. sqaure not equal



    # qstate = [{'modes' : ['wg_1', 'wg_2'], 'pos' : [1, 1], 'occ' : [1,1]},
              # {'modes' : ['wg_1', 'wg_2'], 'pos' : [1, 1], 'occ' : [0,2]},
              # {'modes' : ['wg_1', 'wg_2'], 'pos' : [1, 1], 'occ' : [0,2]}]
              
    print ("Welcome to the new, NEW arch")

    components = []
    laser = LaserCW()
    wg1 = Waveguide()
    bs0 = Beamsplitter()
    wg2 = Waveguide()
    wg3 = Waveguide()
    ps0 = PhaseShifter()
    ps1 = PhaseShifter()
    wg4 = Waveguide()
    wg5 = Waveguide()
    bs1 = Beamsplitter()
    wg6 = Waveguide()
    wg7 = Waveguide()
    det0 = PhotoDiode()
    det1 = BasicSPD()
    wire0 = Wire()
    wire1 = Wire()
    qtrl = Qontrol(nch=1)

    # at c, 1 ps is ~70 microns in silicon
    
    laser.delay = 10
    wg1.delay = 10
    wg2.delay = 10
    wg3.delay = 10
    wg4.delay = 10
    wg5.delay = 10
    wg6.delay = 10
    wg7.delay = 10
    ps0.delay = 10
    ps1.delay = 10
    bs0.delay = 10
    bs1.delay = 10
    det0.delay = 10
    det1.delay = 10
    wire0.delay = 10
    wire1.delay = 10
    qtrl.delay = 10
    
    
    connections = Connectivity( [
                        (laser.out, wg1.inp),
                        (wg1.out,   bs0.in0),
                        
                        (bs0.out0,  wg2.inp),
                        (wg2.out,   ps0.inp),
                        (ps0.out,   wg4.inp),
                        (wg4.out,   bs1.in0),
                        
                        (bs0.out1,   wg3.inp),
                        (wg3.out,   ps1.inp),
                        (ps1.out,   wg5.inp),
                        (wg5.out,   bs1.in1),
                        
                        (bs1.out0,   wg6.inp),
                        (wg6.out,   det0.inp),
                        
                        (bs1.out1,  wg7.inp),
                        (wg7.out,   det1.inp),
                        
                        (det0.out,  wire0.inp),
                        (det1.out,  wire1.inp),
                        (wire0.out, ps0.phi),
                        # (wire1.out, ps1.phi),
                        
                        (qtrl.out_0, ps1.phi)
                        ] )

    components = [b.name for b in connections.blocks]

    cm = laser.model.compound("compound name", connections.models, connections)
    
    
    state = cm.default_input_state
    print("\n\ndefault input state",state)
    
    eval_time = time.time()
    eval_time = time.time() - eval_time
    state = cm.out_func(state)
    
    print("\n\ndefault output state",state)
    
    print(f"Took {eval_time} s")
    
    
    
    
    connections.draw(draw_ports=False)
    
    print(cm)
    
    
    print(cm.in_ports)
    print(cm.out_ports)
    
    state[ps0.phi] = 0
    state[ps1.phi] = 0

    
        # Functions for producing time series
    def constant(v):
        return lambda t : v
        
    def step(v0, v1, t_step):
        return lambda t : v0 if t < t_step else v1
        
    def sinusoid(amp, offset, t_period, phase):
        from math import sin, pi
        return lambda t : (amp/2)*sin(2*pi*t/t_period + phase) + offset
        
    def ramp(v0, v1, t_period):
        from math import pi
        return lambda t : (v1-v0)*(t%t_period)/t_period + v0
    
    
    from math import pi

    

    model_state = cm.default_input_state
    
    dm = get_delay_map(connections)

    
        
    print("Setting up simulator...")
    sim = BasicDynamicalSimulator(
                    blocks=connections.blocks,
                    connectivity=connections,
                    t_start=0,
                    t_stop=200, 
                    t_step=0.1,
                    in_time_funcs={
                        laser.P: step(0.0, 0.9, 20.0),
                        qtrl.qv_0: ramp(0, 2*pi, 50) 
                        })

    print("Simulating...")
    sim.run()

    print(f"Computed {len(sim.times)} time steps.")
    print("Final state is:")
    print_state(sim.time_series[-1])

    sim.plot_timeseries(ports=[laser.out, ps0.phi, ps1.phi, wg6.out, wg7.out], style='stack')


    # """
    # for each component
        # delete & prepend inputs -> buffer[0]
        # delete & prepend buffer[-1] -> ouputs
    # check buffers and outputs are length preserved
    
    # if  using a buffer, when is the operation applied? at the output?
    
    
    # this is inherently discrete time
    # """
    
    # state = [{'block' : bs1, 'mode1' : [0 for i in range(int(bs1.delay))],  'mode2' : [0 for i in range(int(bs1.delay))]}]
    
    # model_state_ts = []
    # for t in range(200):
            
        # for op in cm.out_ports:
            
            # Get delayed inputs
            # for ip in cm.in_ports:
                # if ip in in_time_funcs:
                    # state |= {ip:in_time_funcs[ip](t - dm[op][ip])}

                
            # Update output using delayed inputs
            # state |= cm.out_func(state)
            # t_out_func += time.time()-t0
            # n_out_func_calls += 1
            
            # Update inputs to current time
            # t0 = time.time()
            # for ip in cm.in_ports:
                # if ip in in_time_funcs:
                    # state |= {ip:in_time_funcs[ip](t)}
            
            # states_ts.append((t,state.copy()))
            # t_close_copy += time.time()-t0
            
            
            
    # from matplotlib import pyplot
    
    # pyplot.plot([s[0] for s in model_state_ts], [s[1][laser.P] for s in model_state_ts])
    # pyplot.plot([s[0] for s in model_state_ts], [s[1][ps0.phi] for s in model_state_ts])
    # pyplot.plot([s[0] for s in model_state_ts], [s[1][ps1.phi] for s in model_state_ts])
    # pyplot.show()
    
    # pyplot.close()
    # pyplot.plot([s[0] for s in model_state_ts], [abs(s[1][wg6.out])**2 for s in model_state_ts])
    # pyplot.plot([s[0] for s in model_state_ts], [abs(s[1][wg7.out])**2 for s in model_state_ts])
    # pyplot.show()