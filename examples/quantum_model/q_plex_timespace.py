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
from arch.blocks.interface.electrical  import Comparator, FourTimeSpatPlex
from arch.blocks.sources import BasicPhotonPairSource, BasicSinglePhotonSource
from arch.blocks.sources import LaserPulse
from arch.blocks.qontrol import Qontrol
from arch.blocks.detectors import BasicSPD, PhotoDiode
from arch.blocks.wire import Wire
from arch.architecture import Architecture
from arch.simulations import ng_TL, get_delay_map, BasicDynamicalSimulator,QuantumDynamicalSimulator
import arch.qfunc
import networkx as nx
import thewalrus as tw
import numpy as np
import scipy as scp
from collections import defaultdict
import copy
from math import pi, sin
import csv


	# Functions for producing time series
def constant(v):
	return lambda t : v
	
def step(v0, v1, t_step):
	return lambda t : v0 if t < t_step else v1
	
def square(v0, v1, t_period):
	# return lambda t : v0 if (t % t_period) < t_period/2 and t > 0 else (v1 if t > 0 else  -1)
	return lambda t : v0 if (t % t_period) < t_period/2  else (v1 )
	
def sinusoid(amp, offset, t_period, phase):
	return lambda t : (amp/2)*sin(2*pi*t/t_period + phase) + offset
	
def ramp(v0, v1, t_period):
	return lambda t : (v1-v0)*(t%t_period)/t_period + v0

def gaussian(x, mu, sig):
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def pulsgaus(amp, reprate, sigma, cut):
	return lambda t : amp*gaussian((t % reprate), 0, sigma) if (t % reprate) < cut else 0.
	
def pulsgaus_plex(amp, plexrate, bins, usedbins, sigma, cut):
	return lambda t : amp*gaussian((t % plexrate), 0, sigma) if (t % (bins*plexrate)) < ((usedbins)*plexrate) else 0.




if __name__=='__main__':

	
	print ("Welcome to the q_systems arch!")
	
	#source params
	
	components = []
	
	vac = Vacuum() # needed too keep the vacuum state in the model!

	#source
	laser1  = LaserPulse()
	laser2  = LaserPulse()
	laser3  = LaserPulse()
	laser4  = LaserPulse()
	
	sfwm1 = BasicPhotonPairSource()
	sfwm2 = BasicPhotonPairSource()
	sfwm3 = BasicPhotonPairSource()
	sfwm4 = BasicPhotonPairSource()
	
	#split photons
	wdm1 = WavelengthDivision()
	wdm2 = WavelengthDivision()
	wdm3 = WavelengthDivision()
	wdm4 = WavelengthDivision()
	
	#herald and logic
	hdet1 = BasicSPD()
	hdet2 = BasicSPD()
	hdet3 = BasicSPD()
	hdet4 = BasicSPD()
	
	prehwire = Wire()
	
	hwires0 = Wire()
	hwires1 = Wire()
	hwiret2 = Wire()
	hwiret3 = Wire()
	hwiret4 = Wire()
	hwiret5 = Wire()
	
	swire0 = Wire()
	swire1 = Wire()
	swire2 = Wire()
	swire3 = Wire()
	
	logic = FourTimeSpatPlex(vout = np.pi)

	
	#delay wgs
	wg00 = Waveguide()
	wg01 = Waveguide()
	wg10 = Waveguide()
	wg11 = Waveguide()
	
	#temporal plex
	bst11 = Beamsplitter()
	pst1  = PhaseShifter()
	wgt1  = Waveguide()
	bst12 = Beamsplitter()
	wgdelay1  = Waveguide()
	
	bst21 = Beamsplitter()
	pst2  = PhaseShifter()
	wgt2  = Waveguide()
	bst22 = Beamsplitter()
	wgdelay2  = Waveguide()
	
	bst31 = Beamsplitter()
	pst3  = PhaseShifter()
	wgt3  = Waveguide()
	bst32 = Beamsplitter()
	wgdelay3  = Waveguide()
	
	bst41 = Beamsplitter()
	pst4  = PhaseShifter()
	wgt4  = Waveguide()
	bst42 = Beamsplitter()
	wgdelay4  = Waveguide()
	
	#space plex
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
	
	#output waveguides
	wg00o = Waveguide()
	wg11o = Waveguide()
	
	wg000 = Waveguide()
	wg111 = Waveguide()
	wg111o = Waveguide()
	wg1111 = Waveguide()
	
	#det
	odet1 = BasicSPD()
	odet2 = BasicSPD()
	odet3 = BasicSPD()
	odet4 = BasicSPD()
	
	
	
	connections = Connectivity( [
						(vac.out, wdm1.in1),
						
						#source
						(laser1.out, sfwm1.inp),
						(laser2.out, sfwm2.inp),
						(laser3.out, sfwm3.inp),
						(laser4.out, sfwm4.inp),
						(laser4.clko, prehwire.inp),
						(prehwire.out, logic.clkl),
						(logic.clko, logic.clki),
						(logic.clks, logic.clkis),
						
						(sfwm1.out, wdm1.in0),
						(sfwm2.out, wdm2.in0),
						(sfwm3.out, wdm3.in0),
						(sfwm4.out, wdm4.in0),
						
						(logic.storedo0, logic.storedi0),
						(logic.storedo1, logic.storedi1),
						(logic.storedo2, logic.storedi2),
						(logic.storedo3, logic.storedi3),
						
						# (swire0.out, logic.storedi0),
						# (swire1.out, logic.storedi1),
						# (swire2.out, logic.storedi2),
						# (swire3.out, logic.storedi3),
						
						#split
						(wdm1.out0, hdet1.inp),
						(wdm2.out0, hdet2.inp),
						(wdm3.out0, hdet3.inp),
						(wdm4.out0, hdet4.inp),
						
						#herald and det
						(hdet1.out, logic.in0),
						(hdet2.out, logic.in1),
						(hdet3.out, logic.in2),
						(hdet4.out, logic.in3),
						
						(logic.outs0, hwires0.inp ),
						(hwires0.out, ps12.phi ),
						(hwires0.out, ps34.phi ),
						
						(logic.outs1, hwires1.inp ),
						(hwires1.out, pso.phi ),
						
						(logic.outt0, hwiret2.inp ),
						(hwiret2.out, pst1.phi ),
						
						(logic.outt1, hwiret3.inp ),
						(hwiret3.out, pst2.phi ),
						
						(logic.outt2, hwiret4.inp ),
						(hwiret4.out, pst3.phi ),
						
						(logic.outt3, hwiret5.inp ),
						(hwiret5.out, pst4.phi ),
						
						#delay
						(wdm1.out1, wg00.inp),
						(wdm2.out1, wg01.inp),
						(wdm3.out1, wg10.inp),
						(wdm4.out1, wg11.inp),
						
						(wg00.out, bst11.in1),
						(wg01.out, bst21.in1),
						(wg10.out, bst31.in1),
						(wg11.out, bst41.in1),
						
						#temp multiplex
						(bst11.out0, pst1.inp),
						(bst11.out1, wgt1.inp),
						(pst1.out, bst12.in0),
						(wgt1.out, bst12.in1),
						(bst12.out0, wgdelay1.inp),
						(wgdelay1.out, bst11.in0),
						(bst12.out1, bs12_1.in0),
						
						(bst21.out0, pst2.inp),
						(bst21.out1, wgt2.inp),
						(pst2.out, bst22.in0),
						(wgt2.out, bst22.in1),
						(bst22.out0, wgdelay2.inp),
						(wgdelay2.out, bst21.in0),
						(bst22.out1, bs12_1.in1),
						
						(bst31.out0, pst3.inp),
						(bst31.out1, wgt3.inp),
						(pst3.out, bst32.in0),
						(wgt3.out, bst32.in1),
						(bst32.out0, wgdelay3.inp),
						(wgdelay3.out, bst31.in0),
						(bst32.out1, bs34_1.in0),
						
						(bst41.out0, pst4.inp),
						(bst41.out1, wgt4.inp),
						(pst4.out, bst42.in0),
						(wgt4.out, bst42.in1),
						(bst42.out0, wgdelay4.inp),
						(wgdelay4.out, bst41.in0),
						(bst42.out1, bs34_1.in1),
						
						#space multiplex
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
						
						#outp
						(bso_1.out0, pso.inp),
						(bso_1.out1, wgo.inp),
						(pso.out, bso_2.in0),
						(wgo.out, bso_2.in1),
						(bso_2.out0, odet3.inp),
						(bso_2.out1, odet4.inp),
						
						] )

	t_plex_time = 100
	
	component_names = [b.name for b in connections.blocks]
	blocks = [b for b in connections.blocks]
	
	for comp in connections.blocks:
		comp.delay = 10
		comp.eta = 0.98
		
	swires = [swire0, swire1, swire2, swire3]
	for swirer in swires:
		swirer.delay = 10
		
	prehwire.delay = 80

	delays = [wgdelay1, wgdelay2, wgdelay3, wgdelay4]
	for wgdelay in delays:
		wgdelay.delay = 70
		
	# wdms = [ wdm1, wdm2, wdm3, wdm4] 
	# for wdm in wdms:
		# wdm.delay = 10
		
	spatwgs = [wg00,wg01,wg10,wg11]
	for wg in spatwgs:
		wg.delay = 40
		
	
	# hwirets = [hwiret2,hwiret3,hwiret4,hwiret5]
	# for hwire in hwirets:
		# hwire.delay = 10 
		
		
	hwires0.delay = hwiret2.delay  
	hwires1.delay = hwires0.delay 


		
	print('\n')

	for b in blocks:
		print(b.name,"   ",b.model)

	print('\n')

	for b in blocks:
		if hasattr(b.model, 'U'):
			print(b.name,"   ",b.model,"   ",b.model.U,"   ",b.delay)

	# connections.draw(draw_ports=False)
	print('\n')	
	
	


	qsources = [sfwm1, sfwm2, sfwm3, sfwm4]
	qdets = [hdet1, hdet2, hdet3, hdet4, odet1, odet2, odet3, odet4]
	reprate = 500
	for source in qsources:
		source.reprate = t_plex_time
		source.xi = 0.18
		source.lcutoff = 0
		source.cutoff = 2
		source.pos = [0,0]
		source.freq = ['s', 'i']
		source.hg = [0,0]
		
	for det in qdets:
		det.deadtime = 80
		det.vout = 1
		
	logic.threshold =0.1
	logic.vout = np.pi
	logic.hyst = 0.01
	
	t_start = 0 
	t_stop = reprate*1000
	t_step = 5
	
	bss = [bst11, bst12, bst21, bst22, bst31, bst32, bst41, bst42, bs12_1, bs12_2, bs34_1, bs34_2, bso_1, bso_2]
	for bs in bss:
		bs.R = 1/2.


	print("Setting up simulator...")
	sim = QuantumDynamicalSimulator(
					q_sources = qsources,
					q_dets = qdets,
					photon_no_cutoff = 2,
					blocks = connections.blocks,
					connectivity = connections,
					t_start = t_start,
					t_stop = t_stop, 
					t_step = t_step, 
					verbose = False,
					in_time_funcs = {
						laser1.P: pulsgaus_plex(10., t_plex_time, 5, 4, 2, 20),
						laser1.clk: square(0, 1, t_plex_time),
						laser2.P: pulsgaus_plex(10., t_plex_time, 5, 4, 2, 20),
						laser2.clk: square(0, 1, t_plex_time),
						laser3.P: pulsgaus_plex(10., t_plex_time, 5, 4, 2, 20),
						laser3.clk: square(0, 1, t_plex_time),
						laser4.P: pulsgaus_plex(10., t_plex_time, 5, 4, 2, 20),
						laser4.clk: square(0, 1, t_plex_time),
						})

	print("Simulating...")

	[cstate, qstate_r, timetags] = sim.run()
	
	# print('\nqstate is:')
	# arch.qfunc.printqstate(qstate_r[-1])
	# print('\n')

	# print(f"Computed {len(sim.times)} time steps.")
	# print("Final state is:")
	# print_state(sim.time_series[-1])	
	
	# print(timetags)
	
	#plot time series	
	# sim.plot_timeseries(ports=[laser1.clk,
								# sfwm1.inp, 
								# wdm1.in0,
								# wg00.inp,
								# bst11.in1,
								# pst1.inp,
								# wgt1.inp,
								# bst12.in1,
								# bs12_1.in0, 
								# wg12.inp,
								# bs12_2.in0,
								# wg000.inp,
								# bso_1.in0, 
								# pso.inp,
								# bso_2.in0,
								# odet3.inp,odet3.out], style='stack') 

	# sim.plot_timeseries(ports=[sfwm1.inp,
								# logic.clkis,
								##logic.clkl,
								# logic.in0,
								# logic.in1,
								# logic.in2,
								# logic.in3,
								# ps12.phi,
								# pso.phi,
								##logic.lastbin,
								# pst1.phi,
								# pst2.phi,
								# pst3.phi,
								# pst4.phi,
								# logic.storedo0,
								# logic.storedo1,
								# logic.storedo2,
								# logic.storedo3,
								# odet1.out,
								# odet2.out,
								# odet3.out,
								# odet4.out,
								# ], style='stack') 
								
	# sim.plot_timeseries(ports=[sfwm1.inp,
								# odet1.inp,
								# odet2.inp,
								# odet3.inp, 
								# odet4.inp, 
								# odet1.out, 
								# odet2.out, 
								# odet3.out, 
								# odet4.out], style='stack') 



    ###### SAVE DATA! ######

	# output csv of timetags `
	datahead = ['times', 'freq', 'hg', 'occ', 'pos', 'tran', 'wg', 'detphotno']
	data = [ [timetags['times'][i], timetags['modes'][i][0], timetags['modes'][i][1], timetags['modes'][i][2], timetags['modes'][i][3], timetags['modes'][i][4], timetags['modes'][i][5], timetags['detno'][i]] for i in range(len(timetags['times']))  ] 
	
	t = time.localtime()
	current_time = time.strftime("%H-%M-%S", t)
	filename_globaltags = 'timetags__T=' + str(t_stop) + '__dt=' + str(t_step) + '__R=' + str(reprate)
	
	
	dir_path = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname(os.path.realpath(__file__)))))
	save_path = dir_path + '/arch_raw-output-data/' + str(datetime.date.today()) + '__' +  current_time 
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
	# cdata = []		
	# for i in range(len(sim.time_series)):		
		# l = [{'kind':str(p.kind),'port name':str(p.local_name),'block name':str(p.block.name),'port':p,'value':v} for p,v in sim.time_series[i].items()]
		# l.sort(key=(lambda e : (e['block name'],e['kind'],e['port name'])))
		# cdata.append(   [e['value'] for e in l]  )
		# classical_datahead = [str(e['port']) for e in l]
			
	# filename_globaltags = 'classica__T=' + str(t_stop) + '__dt=' + str(t_step) + '__R=' + str(reprate)
	# with open(os.path.join(save_path,filename_globaltags+'___classical data.csv') , 'w', encoding='UTF8', newline='') as f:
		# writer = csv.writer(f)
		# writer.writerow(classical_datahead)
		# writer.writerows(cdata)
			
