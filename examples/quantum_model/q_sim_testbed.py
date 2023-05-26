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
	###
import abc
import arch.port as port
from arch.port import var, print_state
from arch.block import Block
from arch.connectivity import Connectivity
from arch.models import Model, SymbolicModel, NumericModel, SourceModel
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder, Waveguide, Vacuum
from arch.blocks.sources import BasicPhotonPairSource, BasicSinglePhotonSource
from arch.blocks.sources import LaserCW
from arch.blocks.qontrol import Qontrol
from arch.blocks.detectors import BasicSPD, PhotoDiode
from arch.blocks.wire import Wire
from arch.architecture import Architecture
from arch.simulations import length_to_ps, ps_to_length, ng_TL, get_delay_map, BasicDynamicalSimulator,QuantumDynamicalSimulator
import networkx as nx
import thewalrus as tw
import numpy as np
import scipy as scp
from collections import defaultdict
import copy
from math import pi, sin
import arch.qfunc


	# Functions for producing time series
def constant(v):
	return lambda t : v
	
def step(v0, v1, t_step):
	return lambda t : v0 if t < t_step else v1
	
def sinusoid(amp, offset, t_period, phase):
	return lambda t : (amp/2)*sin(2*pi*t/t_period + phase) + offset
	
def ramp(v0, v1, t_period):
	return lambda t : (v1-v0)*(t%t_period)/t_period + v0

def gaussian(x, mu, sig):
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def pulsgaus(v, reprate, sigma, cut):
	return lambda t : gaussian((t % reprate), cut/2, sigma) if (t % reprate) < cut else 0.



if __name__=='__main__':

   

	
	print ("Welcome to the q_systems arch!")
	#source params
	wg_loss=0.99
	components = []
	laser = LaserCW()
	vac  = Vacuum()
	sps0 = BasicSinglePhotonSource()
	sps1 = BasicSinglePhotonSource()
	wg0 = Waveguide(eta = wg_loss)
	wg0_0 = Waveguide(eta = wg_loss)
	wg1 = Waveguide(eta = wg_loss)
	wg1_0 = Waveguide(eta = wg_loss)
	bs0 = Beamsplitter(R = 1/2)
	wg2 = Waveguide(eta = wg_loss)
	wg3 = Waveguide(eta = wg_loss)
	ps0 = PhaseShifter()
	ps1 = PhaseShifter()
	wg4 = Waveguide(eta = wg_loss)
	wg5 = Waveguide(eta = wg_loss)
	bs1 = Beamsplitter(R = 1/2)
	bse = Beamsplitter(R = 1/2)
	wg6 = Waveguide(eta = wg_loss)
	wg7 = Waveguide(eta = wg_loss)
	det0 = BasicSPD()
	det1 = BasicSPD()
	wire0 = Wire()
	wire1 = Wire()
	qtrl = Qontrol(nch=1)

	
	vac.delay = 10.
	laser.delay = 10.
	wg0.delay = 10.
	wg0_0.delay = 10.
	sps0.delay = 10.
	sps1.delay = 10.
	wg1.delay = 10.
	wg1_0.delay = 10.
	wg2.delay = 10.
	wg3.delay = 10.
	wg4.delay = 10.
	wg5.delay = 10.
	wg6.delay = 10.
	wg7.delay = 10.
	ps0.delay = 10.
	ps1.delay = 10.
	bs0.delay = 10.
	bs1.delay = 10.
	bse.delay = 10.
	det0.delay = 5.
	det1.delay = 5.
	wire0.delay = 25.
	wire1.delay = 25.
	qtrl.delay = 10.
	
	laser.eta = 9.9e-1
	wg0.eta = 9.9e-1
	wg0_0.eta = 9.9e-1
	sps0.eta = 9.9e-1
	sps1.eta = 9.9e-1
	wg1_0.eta = 9.9e-1
	wg1.eta = 9.9e-1
	wg2.eta = 9.9e-1
	wg3.eta = 9.9e-1
	wg4.eta = 9.9e-1
	wg5.eta = 9.9e-1
	wg6.eta = 9.9e-1
	wg7.eta = 9.9e-1
	ps0.eta = 9.9e-1
	ps1.eta = 9.9e-1
	bs0.eta = 9.9e-1
	bs1.eta = 9.9e-1
	bse.eta = 9.9e-1
	det0.eta = 9.9e-1
	det1.eta = 9.9e-1
	wire0.eta = 9.9e-1
	wire1.eta = 9.9e-1
	qtrl.eta = 9.9e-1
	
	det0.deadtime = 0
	det1.deadtime = 0
	
	
	connections = Connectivity( [
						(vac.out, sps0.inp),
						(laser.out, bse.in0),
						(laser.out, bse.in1),
						(bse.out0, wg0_0.inp),
						(bse.out1, wg0.inp),
						
						(wg0.out,   sps0.inp),
						(sps0.out, wg1.inp),
						(wg1.out,   bs0.in0),
						
						(wg0_0.out,   sps1.inp),
						(sps1.out, wg1_0.inp),
						(wg1_0.out,   bs0.in1),
						
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
						
						(qtrl.out_0, ps1.phi)
						] )

	print(connections)
	component_names = [b.name for b in connections.blocks]
	blocks = [b for b in connections.blocks]

	print('\n')

	for b in blocks:
		print(b.name,"   ",b.model)

	print('\n')
	print('\n')

	for b in blocks:
		if hasattr(b.model, 'U'):
			print(b.name,"   ",b.model,"   ",b.model.U,"   ",b.delay)

	connections.draw(draw_ports=False)
	print('\n')	

	det0.vout = 0.1
	det1.vout = 0.1

	reprate  = 30
	
	sps0.reprate = reprate
	sps0.pos = 0
	sps0.freq = 's'
	sps0.hg = 0
	sps0.amp = np.sqrt(0.9)
	
	sps1.reprate = reprate
	sps1.pos = 0
	sps1.freq = 's'
	sps1.hg = 0
	sps1.amp = np.sqrt(0.9)
	
		
	print("Setting up simulator...")
	sim = QuantumDynamicalSimulator(
					q_sources = [sps0, sps1],
					q_dets = [det0, det1],
					photon_no_cutoff = 2,
					blocks = connections.blocks,
					connectivity = connections,
					t_start = 0,
					t_stop = 300, 
					t_step = 1,
					verbose = False,
					in_time_funcs={
						laser.P: pulsgaus(1., reprate, 2, 20)
						})

	print("Simulating...")
	
	[cstate, qstate_r, timetags] = sim.run()
	
	print('\nqstate is:')
	arch.qfunc.printqstate(qstate_r)
	print('\n')

	print(f"Computed {len(sim.times)} time steps.")
	print("Final state is:")
	print_state(sim.time_series[-1])

	sim.plot_timeseries(ports=[laser.out, ps0.phi, ps1.phi, wg6.out, wg7.out, det0.out, det1.out], style='stack')

