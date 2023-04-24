"""
Functions and objects describing methods of simulation such as monte-carlo.
"""

import abc
import numpy as np
import scipy as sp
from sympy import Matrix,N
import string
import random
import copy
import math
from collections import defaultdict
import tabulate
import itertools

import importlib.util
from arch.connectivity import Connectivity
import arch.qfunc
try:
	import thewalrus
except:
	print("Unable to import `thewalrus`. Using (slower) permanent backup function." )


class Simulator(abc.ABC):
	"""
	Base class for simulations.
	"""
	
	def __init__(self, blocks=[], connectivity=Connectivity(), **kwargs):
		
		self.blocks = blocks
		self.connectivity = connectivity
		
		self.define(**kwargs)
	
	
	@property
	def connectivity(self):
		return self.__connectivity
	
	
	@connectivity.setter
	def connectivity(self, con):
		self.__connectivity = con
		
		# Update our port trackers to match the new connectivity
		self.ports = con.external_ports
		self.internal_ports = con.internal_ports
		self.all_ports = con.internal_ports | con.external_ports
		self.in_ports = con.external_in_ports
		self.out_ports = con.external_out_ports
	

	@property
	def default_state(self):
		"""Dictionary of default values keyed by input port"""
		return {p:p.default for p in self.all_ports}
	
	
	@abc.abstractmethod
	def define(self, **kwargs):
		"""
		Propagate input state to output state.
		
		Subclasses can implement this method to perform their own intialisation.
		"""
		pass

	@abc.abstractmethod
	def run(self):
		"""
		Propagate input state to output state.
		
		Subclasses must implement this method.
		"""
		pass




class DynamicalSimulator(Simulator):
	"""
	Base class for simulations which evolve with time.
	"""
	
	def define(self, t_start=0, t_stop=0, t_step=1.0,
				in_time_funcs={}, q_sources = None, q_dets = None, photon_no_cutoff = 99, verbose = False,
				get_delay_func=(lambda b : b.delay if hasattr(b,'delay') else 0.0),
				**kwargs):
		"""
		t_start: simulation start time, seconds
		t_stop: simulation stop time, seconds
		t_step: simulation time step size, or scale for adaptive simulators
		in_time_funcs: dict of funcs of one variable (time, seconds)
		get_delay_func: function of one variable (Block) to get that block's delay
		"""
		
		self.t_start = t_start
		self.t_stop = t_stop
		self.t_step = t_step
		
		self.photon_no_cutoff = photon_no_cutoff
		self.verbose = verbose
		self.in_time_funcs = in_time_funcs
		self.q_sources = q_sources
		self.q_dets = q_dets
		self.get_delay_func = get_delay_func
	
	
	def _uniform_time_range(self):
		"""
		Return standardised time sequence from start and stop times and time step.
		"""
		return np.arange(self.t_start, self.t_stop, self.t_step, dtype=np.float64)
	
	
	def plot_timeseries(self, ports=[], style='overlap', show=True):
		"""
		Plot computed time series (after calling `run`)
		
		ports: iterable of ports to plot
		style: string in ["overlap", "stack"]
		show: bool, whether to show the plot now (or later, with `pyplot.show()`)
		"""
		
		if not hasattr(self, 'time_series'):
			raise RuntimeError("Must call DynamicalSimulator.run() first.")
		
		from matplotlib import pyplot as plt
		from arch.port import norm
		
		
		if style == 'overlap':
			for p in ports:
				plt.plot(self.times, [norm(p,s[p]) for s in self.time_series])
				
		elif style == 'stack':
			fig, axs = plt.subplots(len(ports))
			for i,p in enumerate(ports):
				axs[i].set_title(str(p))
				axs[i].plot(self.times, [norm(p,s[p]) for s in self.time_series])
				
		else:
			raise AttributeError("Plot style '{:}' not recognised. See help(plot_timeseries) for available options".format(style))
		
		try:
			plt.tight_layout()
		except:
			pass
		
		if show:
			plt.show()


class QuantumDynamicalSimulator(DynamicalSimulator):
	"""
	Time-stepping simulator that does no model compounding, handles
	continuous-time delays on each block.
	"""
	
	def firesource(self, qstate, source, idx):
		source_name = source.name + '_' + '0'#str(idx)
		if source.reference_prefix == 'SV':
			sourcestate = arch.qfunc.sqz_vac_hack(xi = source.xi, 
																	 pos = [source.pos[0],source.pos[1]], 
																	 wgs = [source_name, source_name], 
																	 freq = [source.freq[0], source.freq[1]], 
																	 hg = [source.hg[0], source.hg[1]], 
																	 cutoff = source.cutoff,
																	 lcutoff = source.lcutoff  )
																	 
		if source.reference_prefix == 'SPS':
			sourcestate = arch.qfunc.sps(amp = source.amp, wgs = source_name, pos = source.pos, freq = source.freq, hg = source.hg)	  
		qstate = arch.qfunc.concat_vec(qstate, sourcestate)
		# arch.qfunc.printqstate(qstate)
		qstate = arch.qfunc.cleanzeroes(qstate)
		
		return qstate
		
				
		
	def applyloss_keep_all_lossmodes(self, qstate, crunch_comps, state, verbose):
		"""
		depreciated method to unitarily apply loss (exponential sized vector space! do not use)
		JCA 2023
		"""
		for comp in crunch_comps:
			# if qstate[j]['wg'][i][0:2] != 'L-' and qstate[j]['freq'][i][0:3] != 'vac':
			# print(qstate[j]['wg'][i])
			# curr_comp_name, curr_comp_idx = qstate[j]['wg'][i].split('_') # which component are we in? take from qstate
			# curr_comp = [comp for comp in self.blocks if comp.name == curr_comp_name][0] # find comp in blocks
			Unp =  np.cdouble(N(comp.model.U.subs(state)))
			Unp = np.array(Unp)
			for k in range(len(Unp)):
				eta = comp.eta
				rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(4)) 
				U = arch.qfunc.lossU(eta)
				lossermodes = [comp.name + '_' + str(k), 'L-' + rs1 + '_0']
				# lossermodes = [comp.name + '_' + str(k), '']
				qstate = arch.qfunc.qcruncher(qstate, U, lossermodes)   
		
		return qstate
			
		
	def applyloss_old(self, qstate, verbose):
		"""
		depreciated method to apply loss - nonunitary without explicit loss modes so does not keep normalisation with vacuum mode
		JCA 2023
		"""
		lossvecs=[]
		vac = qstate[0]
		for j,vec in enumerate(qstate):
			if sum(vec['occ']) > 0:		#don't bother with vacuum
				occs_plus1 = [occ+1 for occ in vec['occ']]  # for correct coutning with range()
				occs = vec['occ']
				
				vec_comp_names = [vec['wg'][i].split('_') for i in range(len(occs_plus1))] # which components  are we in? take from qstate
				vec_comp_names = [comp[0] for comp in vec_comp_names] #get first part
				vec_comps_set = [comp for comp in self.blocks if comp.name in vec_comp_names]
				vec_comps = []
				idx=0
				for i in range(len(vec_comp_names)):
					vec_comps.append(vec_comps_set[idx])
					if vec_comp_names[i] != vec_comp_names[(i+1) % len(vec_comp_names)] :
						idx+=1
						
				etas = [vec_comps[i].eta for i in range(len(occs_plus1))]

				idxs = [range(occs_plus1[i]) for i in range(len(occs_plus1))]
				idx_perms = list(itertools.product(*idxs))
				amp_perms = [ 0 for i in range(np.prod(occs_plus1))]
				for idx,perm in enumerate(idx_perms):
					print(j,"	", perm, "	", occs)
					amp_perms[idx] = [ np.sqrt(sp.special.binom(occs[i], perm[i]) * etas[i]**perm[i] * (1-etas[i])**(occs[i]-perm[i])  ) for i in range(len(occs))] #
				print(amp_perms)
				amp_perms = [np.prod(perm) for perm in amp_perms]   # amplitudes for vecs with idx_perms photons in
				print(amp_perms)
				for idx,amp in enumerate(amp_perms):
					addvec = copy.deepcopy(vec)
					addvec['occ'] = list(idx_perms[idx])
					addvec['amp'] = amp*vec['amp']
					addvec['pos'] = [pos if addvec['occ'][idx] > 0 else -1 for idx,pos in enumerate(addvec['pos']) ]
					lossvecs.append(addvec)
					
					
		qstate =  [vac] + lossvecs 
		print('\nlossvec:')
		arch.qfunc.printqstate(qstate)
		qstate = arch.qfunc.cleanzeroes(qstate) # clean before simplify works better
		
		print('\ncleanedzeroes:')
		arch.qfunc.printqstate(qstate)
		
		print('\nsimplified:')
		qstate = arch.qfunc.q_simplify(qstate)
		arch.qfunc.printqstate(qstate)
		
		
		return qstate
		
			
		
	def applydetection(self, qstate, t, timetags, active_dets, photon_det_event_coords, verbose):
		"""
		applies quantum measurement - applied when photons hit block with "detector" property
		JCA 2023
		"""
		vacwg = 'V0_0'
		
		# print('-------- BEGIN DET ---------')

		det_photon_modelabels = [[] for i in range(len(photon_det_event_coords))] 
		for _,det_coord in enumerate(photon_det_event_coords):	   # get mode labels of detected photons
			for key in sorted(qstate[det_coord[0]].keys() - ['amp'] ):
				det_photon_modelabels[_].append (qstate[det_coord[0]][key][det_coord[1]])

		det_photon_modelabels_unq = set(tuple(i) for i in det_photon_modelabels) # which photons are the same and being detected?

		outcome_modes = []
		outcome_amps = []
		idx_of_same_parts = []
		mode_of_same_parts = []
		for modelabel in det_photon_modelabels_unq:  #for those which are the same, get their amplitudes
			idx_of_same_parts.append([photon_det_event_coords[i][0] for i, x in enumerate(det_photon_modelabels) if x == list(modelabel)])
			mode_of_same_parts.append([photon_det_event_coords[i][1] for i, x in enumerate(det_photon_modelabels) if x == list(modelabel)])
			outcome_amps.append( [qstate[i]['amp'] for i in idx_of_same_parts[-1]] )
			outcome_modes.append(modelabel)
		
		#what is the prob of no detection? which bits of state does this correspond to? find prob, 
		idx_of_no_det  = [ x for x in range(len(qstate)) if x not in list(zip(*photon_det_event_coords))[0] ]
		no_click_prob  = sum([np.abs(qstate[i]['amp'])**2 for i in idx_of_no_det])


		probs = [] # lets compute the probabilities of these events
		for ampset in outcome_amps:
			probs.append (sum([np.abs(x)**2 for x in ampset]))
		probs.append(no_click_prob)				
		norm = sum(probs)
		probs = [prob/norm for prob in probs] #pesky numpy function needs normalised probability distributoin

		state_collapse_compononent_idxs = idx_of_same_parts + [idx_of_no_det]
		state_collapse_mode_idxs = mode_of_same_parts #+ [mode_of_no_det]
		
		if verbose: #verbose:
			print(state_collapse_compononent_idxs)
			print(state_collapse_mode_idxs)
			print(outcome_modes)
			print(probs)
			print('sum of probs at measurement was:', sum(probs))

		#collapsee state!
		clicks = []
		detection_result = np.random.choice(len(probs), 1, p=probs)[0]
		if detection_result in range(len(probs) - 1):
			if verbose:
				print('KK-KLICK! - A PHOTON WAS DETECTED HERE! result index was: ', detection_result)
			timetags['modes'].append(outcome_modes[detection_result])
			timetags['times'].append(t)
			
			photno = outcome_modes[detection_result][2]
			phottran = outcome_modes[detection_result][4]
			photno_probs = [sp.special.binom(photno,i) * phottran**i * (1 - phottran)**(photno-i) for i in range(photno+1)]
			loss_result = np.random.choice(len(photno_probs), 1, p=photno_probs)[0]
			timetags['detno'].append(loss_result)

			clicks.append([outcome_modes[detection_result][5], loss_result])

		if verbose:
			print('—————————   VACUUM WAS DETECTED HERE! result index was: ', detection_result)
		
		qstate = [qstate[i] for i in state_collapse_compononent_idxs[detection_result]]  # does the collapse!

		#delete detected modes
		if detection_result < len(probs) - 1:
			for j,vec in enumerate(qstate):
				for key in qstate[j].keys() - ['amp']:
					qstate[j][key] = [mode for idx,mode in enumerate(qstate[j][key]) if idx != state_collapse_mode_idxs[detection_result][j]  ] #delete detected photon
					
			qstate  = [vec for vec in qstate if len(vec['occ']) != 0]  #the above process leaves empty list states, remove!
		
		if len(qstate) == 0:
			rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(3)) # add vacuum state back in if empty state
			qstate.append( {'amp' : np.cdouble(1),
							'freq' : ['vac_'+rs1],
							'hg'   : [-1],
							'pos'   : [-1],
							'occ'   : [0],
							'wg'   : [vacwg],
							'tran': [0]	   } )
		
		#normalise
		new_norm = sum([np.abs(qstate[i]['amp'])**2 for i in range(len(qstate))]) #renormalise
		for vec in qstate:
			vec['amp'] = vec['amp']/np.sqrt(new_norm)
		qstate = arch.qfunc.cleanzeroes(qstate)
		
		#hand back any detection events that still need processing
		active_dets = []
		photon_det_event_coords = []
		# print(qstate)
		for j,vec in enumerate(qstate):
			for i,pos in enumerate(qstate[j]['pos']):
				curr_comp_name, curr_comp_idx = qstate[j]['wg'][i].split('_') # which component are we in? take from qstate
				curr_comp = [comp for comp in self.blocks if comp.name == curr_comp_name][0] # find comp in blocks
				if 'detector' in curr_comp.model.properties:
					photon_det_event_coords.append([j,i])
					active_dets.append(curr_comp)	# detectors that might fire - measurements to perform 
					print('photon hit detector at vec component ', [j,i])



		return qstate,timetags,clicks,active_dets,photon_det_event_coords
	
	
	
	def applyunitaries(self, qstate, crunch_comps, state, verbose):
		"""
		applies unitaries to quantum state in components that have them
		JCA 2023
		"""
		# if verbose:
			# print('\ncomponents to crunch: ', crunch_comps)

		for comp in crunch_comps:
			if comp.name[0:3] != 'WDM':
				crunchmodes = []
				Unp =  np.cdouble(N(comp.model.U.subs(state)))
				Unp = np.array(Unp)
				for k in range(len(Unp)):
					crunchmodes.append(comp.name + '_' + str(k))

				# if verbose: print('crunchmodes: ', crunchmodes)
				qstate = arch.qfunc.qcruncher(qstate, Unp, crunchmodes)   
				
		#hack to make WDM split photons
		for j,vec in enumerate(qstate):
			for i,pos in enumerate(qstate[j]['pos']):
				if qstate[j]['wg'][i][0] != 'L':
					curr_comp_name, curr_comp_idx = qstate[j]['wg'][i].split('_') # which component are we in? take from qstate
					curr_comp = [comp for comp in self.blocks if comp.name == curr_comp_name][0] # find comp in blocks
					if curr_comp_name[0:3] == 'WDM':
						if qstate[j]['freq'][i] == 'i':
							moder = qstate[j]['wg'][i][0:-1]
							qstate[j]['wg'][i] = moder + '1'
		
		return qstate
		
		
	
	def propagate(self, qstate, lconns, verbose):
		"""
		propogates photons 'pos' index with time, connects quantum states to next component
		JCA 2023
		"""		  
				  
		for j,vec in enumerate(qstate):
			for i,pos in enumerate(qstate[j]['pos']):
			
				curr_comp_name, curr_comp_idx = qstate[j]['wg'][i].split('_') # which component are we in?
				if pos >= 0: 
					curr_comp_idx = int(curr_comp_idx)
				curr_comp = [comp for comp in self.blocks if comp.name == curr_comp_name]
				
				if curr_comp != []:
					curr_comp = curr_comp[0]
					endcompq = math.isclose(qstate[j]['pos'][i] , curr_comp.delay)		
				   
					if endcompq:# we have reached the end of a component
						# what output port are we connected to? we must find it in conns
						connected_blocks = []
						outport = curr_comp.out_ports[curr_comp_idx]
						conto = [pair for pair in lconns if pair.count(outport) > 0][0]
						nex_port = conto[(conto.index(outport) + 1) % 2]
						nex_comp = nex_port.block
						out_port_idx = nex_port.block.in_ports.index(nex_port)
						nex_comp_name = nex_port.block.name + '_' + str(out_port_idx)
						# if verbose:
							# print('\ncomponent ',curr_comp.name,' of index ', i, ' ended at time: ', t)
							# print('connection: ',conto)
							# print('next component: ',nex_comp)
							# print('next component input port: ',nex_port)
							# print('next component input port index: ',out_port_idx)
						qstate[j]['pos'][i] = 0
						qstate[j]['wg'][i] = nex_comp_name
						qstate[j]['tran'][i] = qstate[j]['tran'][i] * curr_comp.eta

		for vec in qstate: # incrememnt 'pos'
			vec['pos'] = [pos + self.t_step if vec['occ'][idx] > 0 else -1 for idx,pos in enumerate(vec['pos']) ]


			
		return qstate
		
	
	def run(self):
		con = self.connectivity
		models = {b.model for b in self.blocks}
		conns = self.connectivity._Connectivity__conns	  
		lconns = list(conns)
		# Range of times
		ts = self._uniform_time_range()
		
		
		# Init state history
		state_history = [self.default_state]
		
		# Compute integer delays for each port
		# port_delay_ints  = {p:0 for p in self.ports}
		port_delay_ints  = {}
		port_delay_ints |= {p:round(self.get_delay_func(b)/self.t_step) for b in self.blocks for p in b.out_ports}
		
		def delayed_port_value(port):
			d = port_delay_ints[port]
			t = len(state_history)
			if d < t:
				return state_history[-d][port]
			else:
				return state_history[0][port]
			
		from arch.port import print_state
		
		qstate = arch.qfunc.sqz_vac(0.001, ['V0_0','V0_0'], pos = [-1,-1], freq = ['s', 'i'], hg = [0, 0],  cutoff = 1, lcutoff = 0)
		# print(qstate)
		# Step through times
		timetags = {'times' : [], 'modes' : [], 'detno' : [] }
		
		deadcounters = {det.name : 0. for det in self.q_dets}
		for t in ts:

			#######################
			#### CLASSICAL SIM ####
			#######################

			state = state_history[-1].copy() # State at `t`
			
			# Update inputs for use at `t`
			# These are the values that would've been present in the past (at `t-delay`)
			# to cause a change in the output value now (at `t`).
			for p in self.in_ports:
				if p in self.in_time_funcs:
					state[p] = self.in_time_funcs[p](t) #-self.get_delay_func(p.block)
			for p in self.out_ports:
				state[p] = delayed_port_value(p)
			
			# Step through models and calculate output port values
			for m in models:
				# Update output using delayed inputs
				o = m.out_func(state)
				state |= o
			
			
			# Propagate values along connectivity
			state |= {pi:delayed_port_value(po) for po,pi in con if po in state}
			
			# Store state in time series
			state_history.append(state)

			
			#####################
			#### QUANTUM SIM ####
			#####################

			
			### SOURCES ###
			for idx,source in enumerate(self.q_sources):
				if state[source.out] > 1.4:
				# if math.isclose(t % source.reprate , 0):	
					qstate = self.firesource(qstate, source, idx)
			# remove higher photon numbers beyond cutoff
			qstate = list(filter(lambda x: np.sum(x['occ']) <= self.photon_no_cutoff, qstate))
				

			
			### APPLY DETECTION ###
			
			# reset dead detectors if deadcounter > deadtime		  
			for det in self.q_dets: 
				deadcounters[det.name] += self.t_step
				if deadcounters[det.name] > det.deadtime:
					state[det.click_in] = 0
					
					
			#do any detectors have photons hitting them?
			# which components are we in? ###
			crunch_comps = set()  # used in applying unitaries
			photon_det_event_coords=[]
			active_dets = []
			clicks = []
			
			for j,vec in enumerate(qstate):
				for i,pos in enumerate(qstate[j]['pos']):
					if qstate[j]['wg'][i][0] != 'L':
					
						curr_comp_name, curr_comp_idx = qstate[j]['wg'][i].split('_') # which component are we in? take from qstate
						curr_comp = [comp for comp in self.blocks if comp.name == curr_comp_name][0] # find comp in blocks
						
						if 'detector' in curr_comp.model.properties:
							photon_det_event_coords.append([j,i])
							active_dets.append(curr_comp)	# detectors that might fire - measurements to perform 
							# print('photon hit detector at vec component ', [j,i])
						else:
							crunch_comps.add(curr_comp)	# components to be churned through in applying unitaries
							
			#apply dark counts -> needs real units	
			this_tick_clicks = []			
			pdarkcount = 0.001
			for det in self.q_dets:
				if pdarkcount > np.random.random(1):
					dclick = [(det.name +'_1'), 1]
					timetags['modes'].append(('d','d','d','d','d',det.name +'_1'))
					timetags['times'].append(t)
					timetags['detno'].append(1)  
					this_tick_clicks.append(dclick)

			while active_dets != []:
				qstate, timetags, clicks, active_dets, photon_det_event_coords = self.applydetection(qstate, t, timetags, active_dets, photon_det_event_coords, self.verbose)
				this_tick_clicks = this_tick_clicks + clicks
			
			#which detectors clicked? forward this to classical state
			if this_tick_clicks != []:
				for click in this_tick_clicks:
					this_click = click[0].split('_')[0] 
					clickdet  =  [ det for det in self.q_dets if det.name == this_click ][0]
					state[clickdet.click_in] = click[1]
					deadcounters[clickdet.name] = 0
					
			
			qstate = self.applyunitaries(qstate, crunch_comps, state, self.verbose)

			qstate = self.propagate(qstate, lconns, self.verbose)

			if self.verbose:
				print('')
				print('-------------------------------------------------------------')
				print('--------------ITERATION COMPLETE, time is: ', t,'-------------')
				print('----------------	 state printed below	 ----------------')
				arch.qfunc.printqstate(qstate)   

  


		state_history.pop(0)
		
		self.time_series = state_history
		self.times = ts
		
		return state_history, qstate, timetags
		
	
	
class BasicDynamicalSimulator(DynamicalSimulator):
	"""
	Time-stepping simulator that does no model compounding, handles
	continuous-time delays on each block.
	"""
	
	
	def run(self):
		
		con = self.connectivity
		models = {b.model for b in self.blocks}
		
		# Range of times
		ts = self._uniform_time_range()
		
		# Init state history
		state_history = [self.default_state]
		
		# Compute integer delays for each port
		# port_delay_ints  = {p:0 for p in self.ports}
		port_delay_ints  = {}
		port_delay_ints |= {p:round(self.get_delay_func(b)/self.t_step) for b in self.blocks for p in b.out_ports}
		# print(port_delay_ints)
		def delayed_port_value(port):
			d = port_delay_ints[port]
			t = len(state_history)
			if d < t:
				return state_history[-d][port]
			else:
				return state_history[0][port]
			
		from arch.port import print_state
		
		# Step through times
		for t in ts:
			
			# State at `t`
			state = state_history[-1].copy()
			
			# Update inputs for use at `t`
			# These are the values that would've been present in the past (at `t-delay`)
			# to cause a change in the output value now (at `t`).
			for p in self.in_ports:
				if p in self.in_time_funcs:
					state[p] = self.in_time_funcs[p](t) #-self.get_delay_func(p.block)
			for p in self.out_ports:
				state[p] = delayed_port_value(p)
			
			# Step through models and calculate output port values
			for m in models:
				# Update output using delayed inputs
				o = m.out_func(state)
				state |= o
			
			# Update `out_state` to contain external in values at time `t` (not delayed)
			# for p in self.in_ports:
				# if p in self.in_time_funcs:
					# state[p] = self.in_time_funcs[p](t)
			
			# Propagate values along connectivity
			state |= {pi:delayed_port_value(po) for po,pi in con if po in state}
			
			# Store state in time series
			state_history.append(state)
		
		state_history.pop(0)
		
		self.time_series = state_history
		self.times = ts
		
		return state_history
		
	
	



class InterferometerSimulator(Simulator):

	"Simulating the output quantum state of an interferometer using permanents."

	def define(self, unitary, input_state, **kwargs):
		"""
		Unitary: Sympy matrix associated with the interferometer
		Input state: Dict of state vector elements and complex amplitudes
		"""
		self.input_state = input_state
		self.unitary_matrix = unitary
		
	


	def create_full_state_unitary(self,unitary_matrix, input_state, modes_list ):
		""" 
		The unitary of the interferometer needs to be extended to act with the identity
		on modes which are not input to the interferometer.
		"""
		total_mode_number=len(list(input_state.keys())[0])
		
		full_state_unitary=np.identity(total_mode_number,dtype=complex)


		for k in range(len(modes_list)):
			for l in range(len(modes_list)):
				full_state_unitary[modes_list[k],modes_list[l]]=unitary_matrix[k][l]

		return full_state_unitary

		
	def create_transition_matrix(self,unitary,input_vector,output_vector, d=complex):
		""" Function to make appropriate changes to unitary so that it represents the desired transition
			from this we can then find the permanent representing the probability of this transition.
			This function must be called for every transition probability required to be calculated.
		"""
		no_photons=int(np.sum(input_vector))
		col_swapped_matrix=np.zeros(shape=(no_photons,no_photons),dtype=d)

		#If there are more or less input photons than output channels we must reshape the matrix slightly for the following to work
		#Definitely exists a more efficient way to do this

		reshaped_unitary=np.zeros(shape=(no_photons,no_photons),dtype=d)
		col_count=0
		row_count=0

		for i in range(len(input_vector)):
			for j in range(len(input_vector)):

				if (no_photons-len(input_vector))>=0:
					reshaped_unitary[i,j]=unitary[i,j]

				elif (no_photons-len(input_vector))<0:
			
					if input_vector[i]!=0 and output_vector[j]!=0:
						reshaped_unitary[row_count,col_count]=unitary[i,j]
						col_count+=1
						row_count+=1

		#Special case of matrix with only 1 photon in and out
		if len(reshaped_unitary)==1:
			return reshaped_unitary[0]


		#Make the column swaps required for the given input vector.
		col_counter=0
		for k in range(len(input_vector)):
			if input_vector[k]==0:
				continue
			else:
				for j in range(input_vector[k]):
					col_swapped_matrix[:,col_counter+j]=copy.deepcopy(reshaped_unitary[:,k])
				col_counter+=1+j


		#Make the row swaps required for a given output vector
		transition_matrix=copy.deepcopy(col_swapped_matrix)
		row_counter=0
		for p in range(len(output_vector)):
			if output_vector[p]==0:
				continue
			else:
				for r in range(output_vector[p]):
					transition_matrix[row_counter+r,:]=copy.deepcopy(col_swapped_matrix[p,:])
				row_counter+=1+r

		
		return transition_matrix

	def calculate_permanent(self, M):
		""" Manual permanent function for cases where thewalrus
		fails to install. As of 04/02/21 no thewalrus wheel
		for python 3.9. Slower than thewalrus, taken from:
		https://github.com/scipy/scipy/issues/7151"""
		
		n = M.shape[0]
		d = np.ones(n)
		j =  0
		s = 1
		f = np.arange(n)
		v = M.sum(axis=0)
		p = np.prod(v)

		while (j < n-1):
			v -= 2*d[j]*M[j]
			d[j] = -d[j]
			s = -s
			prod = np.prod(v)
			p += s*prod
			f[0] = 0
			f[j] = f[j+1]
			f[j+1] = j+1
			j = f[0]
		return p/2**(n-1)


	def calculate_output_amplitudes(self, unitary, input_vector):
		"""Using the probability expression in 'Permanents in linear optical networks' Scheel 2004,
		we calculate the probability of each transition and store it in an array.
		In the fully quantum case we need to calculate all possible contributions to the output state
		that is we need to loop over every element in the input state with a non 0 amplitude
		and calculate every transition probability for that element.
		"""
		state_vector_elements=[list(key) for key in input_vector]
		input_amplitudes=list(input_vector.values() )
		output_amplitudes=np.zeros(shape=(len(input_amplitudes)), dtype=complex)

		#If the walrus not installed use manual permanent calc
		is_walrus_alive = importlib.util.find_spec(name='thewalrus')
	

	
		#For every element of the input state vector
		
		for i in range(len(state_vector_elements)):
			input_element=state_vector_elements[i]
			#Loop over every possible outcome
			for k in range(len(state_vector_elements)):
				element=state_vector_elements[k]

				#If it has a non zero amplitude
				#only consider photon number preserving transitions as these should evaluate to 0 anyway (true?)
				if input_amplitudes[i] != 0 and np.sum(input_element)==np.sum(element): 
				
					#print('The transition being calculated is ', input_element, element )

					trans_matrix=self.create_transition_matrix(unitary, input_element, element)


					if len(trans_matrix)==1:
						output_amplitudes[i]+=(np.abs(trans_matrix[0])**2)*input_amplitudes[i]
						
					else:
						prefactor=1

						if is_walrus_alive is None:
							perm=self.calculate_permanent(trans_matrix)
						else:
							perm=thewalrus.perm(trans_matrix)
						
						for m in range(len(input_element)):
							prefactor=prefactor*(1/math.sqrt(math.factorial(input_element[m])))*(1/math.sqrt(math.factorial(element[m])))
						
						output_amplitudes[k]+=np.around(perm*prefactor, decimals=6)*input_amplitudes[i]
						
		
		return output_amplitudes


	def run(self):
		""" 
		Take the input state and the unitary and
		calculate the full output state
		"""


		unitary_matrix=self.unitary_matrix
		unitary_matrix=np.array(unitary_matrix.tolist()).astype(np.complex)
		input_state=self.input_state
		output_state=input_state
	
		#create appropriate unitary to act on the global state from
		#full_unitary=self.create_full_state_unitary(unitary_matrix, input_state, modes_list)
		
		#calculate the output state amplitudes 
		output_amplitudes=self.calculate_output_amplitudes(unitary_matrix, input_state)
		
		#update the output state dictionary with the new amplitudes
		it=0
		for key in output_state:
			output_state[key]=output_amplitudes[it]
			it+=1

		self.output_state = output_state
		
	




def get_delay_map(connectivity, default_delay=0, block_delay_func=lambda b:b.delay):
	"""
	Build map between the external output ports of `connectivity` and its input ports
	mapping the delay along each path between those ports. Checks that all paths have
	matching delays.
	
	connectivity: `Connectivity` object describing configuration
	default_delay: value to be taken as default when no delay is present
	block_delay_func: function to compute delay from block value; exceptions use default
	"""
	
	def _integrate_path_delay(path):
		"""Add up delay attributes of blocks in list `path`"""
		
		total_delay = 0
		for block in path:
			try:
				delay = block_delay_func(block)
			except:
				delay = default_delay
			total_delay += delay
		
		return total_delay
	
	delay_map = dict()
	for op in connectivity.external_out_ports:
		delay_map |= {op:dict()}
		for ip in connectivity.external_in_ports:
			
			if ip.block is op.block:
				# If input port and output port are from the same block, no delay
				paths = [[ip.block]]
			else:
				# Find all paths between input and output blocks, integrate delays
				paths = list(nx.algorithms.simple_paths.all_simple_paths(
									connectivity.block_graph, ip.block, op.block))
			
			delays = {_integrate_path_delay(p) for p in paths}
			
			# Do checks
			if not len(delays):
				# No path exists
				delays = [default_delay]
			
			assert len(delays) == 1, f"Varying delays found between {ip} and {op}"
			
			# Update map
			delay_map[op] |= {ip:next(iter(delays))}
			
	return delay_map	



	
c = 299792458 # m/s
ng_siwg  = 4.2
ng_fibre = 1.45
ng_TL = 1.45

def length_to_ps(length, ng):
	"""
	converts circuit length to ps delay. rounds to nearest integer
	"""
	return np.around(length * ng/c * 1e12)
	
def ps_to_length(ps, ng):
	"""
	convert ps delay to wire length to 
	"""
	return ps* 1e-12 * c/ng