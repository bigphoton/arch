
# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))



from arch.port import print_state
from arch.connectivity import Connectivity
from arch.blocks.optics import Beamsplitter, PhaseShifter
from arch.blocks.sources import LaserCW
from arch.simulations.simulations import BasicDynamicalSimulator
from math import pi
from arch.simulations.drive import constant, step, ramp


print("Setting up blocks...")
laser = LaserCW()
ps = PhaseShifter()
bs0 = Beamsplitter()
bs1 = Beamsplitter()

print("Setting delays...")
laser.delay = 20.0
ps.delay = 10.0

print("Setting up connectivity...")
connections = Connectivity( [
				(laser.out, bs0.in0),
				(bs0.out0, ps.inp),
				(ps.out, bs1.in0),
				(bs0.out1, bs1.in1) ] )

print("Setting up simulator...")
sim = BasicDynamicalSimulator(
				blocks=connections.blocks,
				connectivity=connections,
				t_start=0,
				t_stop=200, 
				t_step=0.1,
				in_time_funcs={
					laser.P: step(0.0, 1.1, 50.0),
					ps.phi: ramp(0, 2*pi, 50, 0) } )

print("Simulating...")
sim.run()

print(f"Computed {len(sim.times)} time steps.")
print("Final state is:")
print_state(sim.time_series[-1])

print("Visualising outputs...")
connections.draw()
sim.plot_timeseries(ports=[laser.P, laser.out, ps.phi, bs1.out0, bs1.out1], style='stack')
