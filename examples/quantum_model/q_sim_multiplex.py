"""
JCA 2022

"""
import time, datetime

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
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder, Waveguide, Vacuum, WavelengthDivision
from arch.blocks.interface.electrical  import Comparator, FourChLogPlex
from arch.blocks.sources import BasicPhotonPairSource, BasicSinglePhotonSource
from arch.blocks.sources import LaserCW
from arch.blocks.qontrol import Qontrol
from arch.blocks.detectors import BasicSPD, PhotoDiode
from arch.blocks.wire import Wire
from arch.architecture import Architecture
from arch.simulations import length_to_ps, ps_to_length, ng_TL, get_delay_map, BasicDynamicalSimulator,QuantumDynamicalSimulator
import arch.qfunc
import networkx as nx
import thewalrus as tw
import numpy as np
import scipy as scp
from collections import defaultdict
import copy
from math import pi, sin
import csv

print('1')

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

def pulsgaus(amp, reprate, sigma, cut):
	return lambda t : amp*gaussian((t % reprate), cut/2, sigma) if (t % reprate) < cut else 0.
# def pulsgaus(v, reprate, sigma, cut):
	# return lambda t : gaussian((t % reprate), cut/2, sigma) if (t % reprate) < cut else 0.




if __name__=='__main__':

	
	print ("Welcome to the new, NEW, _NEW_ arch")
	#source params
	
	wg_loss = 1
	components = []
	
	vac = Vacuum()

	laser1  = LaserCW()
	laser2  = LaserCW()
	laser3  = LaserCW()
	laser4  = LaserCW()
	
	sfwm1 = BasicPhotonPairSource()
	sfwm2 = BasicPhotonPairSource()
	sfwm3 = BasicPhotonPairSource()
	sfwm4 = BasicPhotonPairSource()
	
	wdm1 = WavelengthDivision()
	wdm2 = WavelengthDivision()
	wdm3 = WavelengthDivision()
	wdm4 = WavelengthDivision()
	
	hdet1 = BasicSPD()
	hdet2 = BasicSPD()
	hdet3 = BasicSPD()
	hdet4 = BasicSPD()
	
	hwire1 = Wire()
	hwire2 = Wire()
	
	# thresh = 0.05
	# c1 = Comparator( threshold=thresh, hysteresis=0.01)
	# c2 = Comparator( threshold=thresh, hysteresis=0.01)
	# c3 = Comparator( threshold=thresh, hysteresis=0.01)
	# c4 = Comparator( threshold=thresh, hysteresis=0.01)
	
	logic = FourChLogPlex(vout = np.pi)

	wgh1 = Waveguide()
	wgh2 = Waveguide()
	wgh3 = Waveguide()
	wgh4 = Waveguide()
	
	wg00 = Waveguide()
	wg01 = Waveguide()
	wg10 = Waveguide()
	wg11 = Waveguide()
	
	bs12_1 = Beamsplitter()
	ps12 = PhaseShifter()
	wg12 = Waveguide()
	bs12_2 = Beamsplitter()
	
	bs34_1 = Beamsplitter()
	ps34 = PhaseShifter()
	wg34 = Waveguide()
	bs34_2 = Beamsplitter()
	
	bso_1 = Beamsplitter()
	pso = PhaseShifter()
	wgo = Waveguide()
	bso_2 = Beamsplitter()
	
	wg00o = Waveguide()
	wg11o = Waveguide()
	
	wg000 = Waveguide()
	wg111 = Waveguide()
	wg111o = Waveguide()
	wg1111 = Waveguide()
	
	odet1 = BasicSPD()
	odet2 = BasicSPD()
	odet3 = BasicSPD()
	odet4 = BasicSPD()
	
	
	
	connections = Connectivity( [
						(vac.out, wdm1.in1),
						
						(laser1.out, sfwm1.inp),
						(laser2.out, sfwm2.inp),
						(laser3.out, sfwm3.inp),
						(laser4.out, sfwm4.inp),
						
						(sfwm1.out, wdm1.in0),
						(sfwm2.out, wdm2.in0),
						(sfwm3.out, wdm3.in0),
						(sfwm4.out, wdm4.in0),
						
						(wdm1.out0, hdet1.inp),
						(wdm2.out0, hdet2.inp),
						(wdm3.out0, hdet3.inp),
						(wdm4.out0, hdet4.inp),
						
						(hdet1.out, logic.in0),
						(hdet2.out, logic.in1),
						(hdet3.out, logic.in2),
						(hdet4.out, logic.in3),
						
						(logic.out0, hwire1.inp ),
						(hwire1.out, ps12.phi ),
						(hwire1.out, ps34.phi ),
						
						(logic.out1, hwire2.inp ),
						(hwire2.out, pso.phi ),
						
						(wdm1.out1, wg00.inp),
						(wdm2.out1, wg01.inp),
						(wdm3.out1, wg10.inp),
						(wdm4.out1, wg11.inp),
						
						(wg00.out, bs12_1.in0),
						(wg01.out, bs12_1.in1),
						(wg10.out, bs34_1.in0),
						(wg11.out, bs34_1.in1),
						
						(bs12_1.out0, ps12.inp),
						(bs12_1.out1, wg12.inp),
						(ps12.out, bs12_2.in0),
						(wg12.out, bs12_2.in1),
						(bs12_2.out0, odet1.inp),
						(bs12_2.out1, wg000.inp),
						
						(bs34_1.out0, ps34.inp),
						(bs34_1.out1, wg34.inp),
						(ps34.out, bs34_2.in0),
						(wg34.out, bs34_2.in1),
						(bs34_2.out0, wg111.inp),
						(bs34_2.out1, odet2.inp),
						
						(wg000.out, bso_1.in0),
						(wg111.out, bso_1.in1),
						
						(bso_1.out0, pso.inp),
						(bso_1.out1, wgo.inp),
						(pso.out, bso_2.in0),
						(wgo.out, bso_2.in1),
						(bso_2.out0, odet3.inp),
						(bso_2.out1, odet4.inp),
						
						] )

	component_names = [b.name for b in connections.blocks]
	blocks = [b for b in connections.blocks]



	for comp in connections.blocks:
		comp.delay = 10
		comp.eta = 0.96
	hwire1.delay = 5
	hwire2.delay = 40
	
	
	bss = [bs12_1, bs12_2, bs34_1, bs34_2, bso_1, bso_2]
	for bs in bss:
		bs.R = 1/2
		bs.delay = 5
		
	pso.delay = 5
	ps12.delay = 5
	ps34.delay = 5
	
	wg12.delay = 5
	wg34.delay = 5
	wgo.delay = 5
				
	inbuilt_delay = 45
	wg00.delay = inbuilt_delay
	wg01.delay = inbuilt_delay
	wg10.delay = inbuilt_delay
	wg11.delay = inbuilt_delay
		
		
		
	print('\n')

	for b in blocks:
		print(b.name,"   ",b.model)

	print('\n')

	for b in blocks:
		if hasattr(b.model, 'U'):
			print(b.name,"   ",b.model,"   ",b.model.U,"   ",b.delay)

	connections.draw(draw_ports=False)
	print('\n')	
	
	


	qsources = [sfwm1, sfwm2, sfwm3, sfwm4]
	qdets = [hdet1, hdet2, hdet3, hdet4, odet1, odet2, odet3, odet4]

	for source in qsources:
		reprate  = 145
		source.reprate = reprate
		source.xi = 0.18
		source.lcutoff = 0
		source.cutoff = 3
		source.pos = [0,0]
		source.freq = ['s', 'i']
		source.hg = [0,0]
		
	for det in qdets:
		det.deadtime = 50
		det.vout = 1
		
	logic.threshold =0.1
	logic.vout = np.pi
	logic.hyst =0.01
	
	t_start = 0 
	t_stop = 8 * reprate
	t_step = 5

	print("Setting up simulator...")
	sim = QuantumDynamicalSimulator(
					q_sources = qsources,
					q_dets = qdets,
					photon_no_cutoff = 4,
					blocks = connections.blocks,
					connectivity = connections,
					t_start = t_start,
					t_stop = t_stop, 
					t_step = t_step,
					verbose = True,
					in_time_funcs = {
						laser1.P: pulsgaus(10., reprate, 2, 20),
						laser2.P: pulsgaus(10., reprate, 2, 20),
						laser3.P: pulsgaus(10., reprate, 2, 20),
						laser4.P: pulsgaus(10., reprate, 2, 20)
						})

	print("Simulating...")
	
	[cstate, qstate_r, timetags] = sim.run()
	
	print('\nqstate is:')
	arch.qfunc.printqstate(qstate_r)
	print('\n')

	print(f"Computed {len(sim.times)} time steps.")
	print("Final state is:")
	# print_state(sim.time_series[-1])

	print_state(sim.time_series[-1])	
	
	sim.plot_timeseries(ports=[sfwm1.out, logic.in0, logic.in1, logic.in2, logic.in3, ps12.out, ps12.phi, ps34.out, ps34.phi, pso.out, pso.phi, odet1.out, odet2.out, odet3.out, odet4.out, odet4.inp, bso_2.out1], style='stack') 
	# sim.plot_timeseries(ports=[sfwm1.out, wdm1.out1, wg00.out, bs12_1.out0, ps12.out, bs12_2.out0, bso_1.out0, pso.out, bso_2.out1], style='stack') 




    ###### SAVE DATA! ######

	# output csv of timetags 
	datahead = ['times', 'freq', 'hg', 'occ', 'pos', 'tran', 'wg', 'detphotno']
	data = [ [timetags['times'][i], timetags['modes'][i][0], timetags['modes'][i][1], timetags['modes'][i][2], timetags['modes'][i][3], timetags['modes'][i][4], timetags['modes'][i][5], timetags['detno'][i]] for i in range(len(timetags['times']))  ] 
	
	t = time.localtime()
	current_time = time.strftime("%H-%M-%S", t)
	filename_globaltags = 'timetags__T=' + str(t_stop) + '__dt=' + str(t_step) + '__R=' + str(reprate)
	save_path = 'C:/onedrive/OneDrive - University of Bristol/quantum/projects/arch_project/data/4ch_multiplex/' + str(datetime.date.today()) + '__' +  current_time 
	os.makedirs(save_path)
	
	with open(os.path.join(save_path,filename_globaltags) + '___global.csv', 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(datahead)
		writer.writerows(data)
		
	detdats = [0 for i in range(len(qdets))]
	for i,det in enumerate(qdets):
		detdats[i] = [ [timetags['times'][i], timetags['modes'][i][0], timetags['modes'][i][1], timetags['modes'][i][2], timetags['modes'][i][3], timetags['modes'][i][4], timetags['modes'][i][5], timetags['detno'][i]] for i in range(len(timetags['times'])) if timetags['modes'][i][5][0:3] == det.name ] 
	
		with open(os.path.join(save_path,filename_globaltags+'___'+det.name + '.csv') , 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(datahead)
			writer.writerows(detdats[i])
			
    # output classical data stream
	cdata = []		
	for i in range(len(sim.time_series)):		
		l = [{'kind':str(p.kind),'port name':str(p.local_name),'block name':str(p.block.name),'port':p,'value':v} for p,v in sim.time_series[i].items()]
		l.sort(key=(lambda e : (e['block name'],e['kind'],e['port name'])))
		cdata.append(   [e['value'] for e in l]  )
		classical_datahead = [str(e['port']) for e in l]
			
	filename_globaltags = 'classica__T=' + str(t_stop) + '__dt=' + str(t_step) + '__R=' + str(reprate)
	with open(os.path.join(save_path,filename_globaltags+'___classical data.csv') , 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(classical_datahead)
		writer.writerows(cdata)
			
