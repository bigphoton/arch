"""
Functions and objects describing optical components.
"""

import abc
import scipy as sp
import random
import string
from sympy import N
from ..connectivity import Connectivity
import arch.port as port
import arch.qfunc
import numpy as np
import math
from arch.models import Model

class QuantumNumericalModel:

	def __init__(self, blocks, lconns, t_step):
		self.blocks = blocks
		self.lconns = lconns
		self.t_step = t_step
		self.qstate = [{'amp' :1., 'wg' : ['V0_0'], 'pos' : [-1], 'occ' : [0], 'tran' : [1.], 'freq' : ['v'], 'hg' : [0]}]
	
	def firesource(self, qstate, source, idx):
		"""
		fires quantum photonic photon sources. Currently only two-mode photon pair source and 
		quasi-deterministic single photon source are implemented
		should take source variables from DynamicalSimulator
		JCA 20223
		"""
		source_name = source.name + '_' + '0'#str(idx)
		if source.reference_prefix == 'SV':
			sourcestate = arch.qfunc.sqz_vac(xi = source.xi, 
														 pos = [source.pos[0],source.pos[1]], 
														 wgs = [source_name, source_name], 
														 freq = [source.freq[0], source.freq[1]], 
														 hg = [source.hg[0], source.hg[1]], 
														 cutoff = source.cutoff,
														 lcutoff = source.lcutoff  )
																	 
		if source.reference_prefix == 'SPS':
			sourcestate = arch.qfunc.sps(amp = source.amp, wgs = source_name, 
											pos = source.pos, freq = source.freq, hg = source.hg)	  
		qstate = arch.qfunc.concat_vec(qstate, sourcestate)
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
		depreciated method to apply loss - nonunitary without explicit loss modes so does not keep normalisation!
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
		vacwg = 'V0_0' # corresponds to vacuum in the block connectivity - needed too keep the vacuum state in the model!
		
		
		
		# what modes are the detected photons in? get their modelabels
		det_photon_modelabels = [[] for i in range(len(photon_det_event_coords))] 
		for _,det_coord in enumerate(photon_det_event_coords):	   # get mode labels of detected photons
			for key in sorted(qstate[det_coord[0]].keys() - ['amp'] ):
				det_photon_modelabels[_].append (qstate[det_coord[0]][key][det_coord[1]])

		det_photon_modelabels_unq = set(tuple(i) for i in det_photon_modelabels) # which photons are the same and being detected?



		#for those which are the same, get their amplitudes
		outcome_modes = []
		outcome_amps = []
		idx_of_same_parts = []
		mode_of_same_parts = []
	
		for modelabel in det_photon_modelabels_unq:  
			idx_of_same_parts.append([photon_det_event_coords[i][0] for i, x in enumerate(det_photon_modelabels) if x == list(modelabel)])
			mode_of_same_parts.append([photon_det_event_coords[i][1] for i, x in enumerate(det_photon_modelabels) if x == list(modelabel)])
			outcome_amps.append( [qstate[i]['amp'] for i in idx_of_same_parts[-1]] )
			outcome_modes.append(modelabel)
		
		
		
		#what is the prob of no detection? which bits of state does this correspond to? find prob, 
		idx_of_no_det  = [ x for x in range(len(qstate)) if x not in list(zip(*photon_det_event_coords))[0] ]
		no_click_prob  = sum([np.abs(qstate[i]['amp'])**2 for i in idx_of_no_det])


		# lets compute the probabilities of these events
		probs = [] 
		for ampset in outcome_amps:
			probs.append (sum([np.abs(x)**2 for x in ampset]))
		probs.append(no_click_prob)				
		norm = sum(probs)
		probs = [prob/norm for prob in probs] # numpy function needs normalised probability distributoin

		state_collapse_compononent_idxs = idx_of_same_parts + [idx_of_no_det]
		state_collapse_mode_idxs = mode_of_same_parts 
		
		
		
		if verbose == 2: # TODO: implement levels of verbosity
			print(state_collapse_compononent_idxs)
			print(state_collapse_mode_idxs)
			print(outcome_modes)
			print(probs)
			print('sum of probs at measurement was:', sum(probs))



		#collapse the state!
		clicks = []
		detection_result = np.random.choice(len(probs), 1, p=probs)[0]  # pick a state for collapse using random number
		if detection_result in range(len(probs) - 1):
			if verbose:
				print('KK-KLICK! - A PHOTON WAS DETECTED HERE! result index was: ', detection_result)
			timetags['modes'].append(outcome_modes[detection_result])	# append to timtags variable for data storage and ouput
			timetags['times'].append(t)
			
			photno = outcome_modes[detection_result][2]   
			phottran = outcome_modes[detection_result][4]  # all loss is applied at detection stage by keeping track of total loss each photon has experienced, in vec['tran'] 
			photno_probs = [sp.special.binom(photno,i) * phottran**i * (1 - phottran)**(photno-i) for i in range(photno+1)]  # lost photon distribution is binomial
			loss_result = np.random.choice(len(photno_probs), 1, p=photno_probs)[0]	 # pick with random number
			timetags['detno'].append(loss_result)  # save data

			clicks.append([outcome_modes[detection_result][5], loss_result])

		if verbose:
			print('—————————   VACUUM WAS DETECTED HERE! result index was: ', detection_result)
		
		qstate = [qstate[i] for i in state_collapse_compononent_idxs[detection_result]]  # does the collapse, keeps only the part of wavefunction that had the detector photon in it




		# vector states will still have this photon in it, even thought its been detected! delete them!
		if detection_result < len(probs) - 1:
			for j,vec in enumerate(qstate):
				for key in qstate[j].keys() - ['amp']:
					qstate[j][key] = [mode for idx,mode in enumerate(qstate[j][key]) if idx != state_collapse_mode_idxs[detection_result][j]  ] #delete detected photon
					
			qstate  = [vec for vec in qstate if len(vec['occ']) != 0]  #the above process leaves empty list states, remove!
		
		
		
		
		 # add vacuum state back in if empty state
		if len(qstate) == 0:
			rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(3))
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
		
		
		
		# hand back any detection events that still need processing, then we run this routine again
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

				qstate = arch.qfunc.qcruncher(qstate, Unp, crunchmodes)   
				
		#hack to make WDM split photons TODO: FIXME add model for WDM - make unitarys apply to arbitrary mode index
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
		
		
	
	
	def propagate(self, qstate, state, lconns, verbose):
		"""
		propogates photons 'pos' index with time, connects quantum states to next component
		JCA 2023
		"""		  
		#apply unitary transformation
		next_comps = []
		for j,vec in enumerate(qstate):
			for i,pos in enumerate(qstate[j]['pos']):
				# 'wg' takes form COMPONENTx_y where x is block index, y is mode (output port) index 

				curr_comp_name, curr_comp_idx = qstate[j]['wg'][i].split('_') # which component are we in?
				if pos >= 0: 
					curr_comp_idx = int(curr_comp_idx)
				curr_comp = [comp for comp in self.blocks if comp.name == curr_comp_name]
				
				if curr_comp != []:
					curr_comp = curr_comp[0]
					endcompq = math.isclose(qstate[j]['pos'][i] , curr_comp.delay, abs_tol = 0.1*self.t_step)		
					
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
						
						#propagate to next component in connectivity
						qstate[j]['pos'][i] = 0
						qstate[j]['wg'][i] = nex_comp_name
						qstate[j]['tran'][i] = qstate[j]['tran'][i] * curr_comp.eta
						
						if  "optical" in nex_comp.model.properties:
							next_comps.append(nex_comp)


		for vec in qstate: # incrememnt 'pos'
			vec['pos'] = [pos + self.t_step if vec['occ'][idx] > 0 else -1 for idx,pos in enumerate(vec['pos']) ]
		
		next_comps = [*set(next_comps)]
		# print(next_comps)
		qstate = self.applyunitaries(qstate, next_comps, state, verbose)
		
		return qstate
	
		