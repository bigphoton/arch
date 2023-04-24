try:
	import colored_traceback.auto
except ImportError:
	pass
	
import time
import thewalrus as tw
import networkx as nx
import numpy as np
import scipy as scp
from collections import defaultdict
import copy
import cmath
import operator
import string
import random
import tabulate


def printqstate(qstate):
	"""
	Uses tabulate to print the quantum state in a human-readable table
	JCA 2023
	"""
	my_qstate = copy.deepcopy(qstate)
	for state in my_qstate:
		amp = state['amp']
		del state['amp']
		state['|amp|'] = np.abs(amp)
		state['phase (pi)'] = np.angle(amp)/np.pi
	header = my_qstate[0].keys()
	rows =  [x.values() for x in my_qstate]
	print(tabulate.tabulate(rows, header))



def sqz_vac(xi, wgs, pos = [0,0], freq = ['sig', 'idl'], hg = [0, 0],  cutoff = 2, lcutoff = 0):
	"""
	Generates a squeezed vacuum state in the dictionary format at position pos
	JCA 2023
	"""
	sqz_state = []
	for i in range(lcutoff, cutoff):
		zi = np.cdouble(1j*np.exp(1j*np.angle(xi))*np.tanh(xi))
		amp = np.cdouble(np.sqrt(1-abs(zi)**2)*((-zi)**i))
		sqz_state.append({'amp' : amp, 
					'wg' : [wgs[0], wgs[1]], 
					'occ' : [i,i],
					'pos' : [pos[0], pos[1]],
					'freq' : [freq[0], freq[1]],
					'hg' : [hg[0], hg[1]],
					'tran' : [1,1]})

	sqz_state = [vec for vec in sqz_state if not len(vec['occ']) == 0]

	rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(3))
	if lcutoff == 0:
		sqz_state[0]['freq'] = ['vac_'+rs1]
		sqz_state[0]['hg'] = [-1]
		sqz_state[0]['pos'] = [-1]
		sqz_state[0]['occ'] = [0]
		sqz_state[0]['wg'] = [wgs[0]]
		sqz_state[0]['tran'] = [0]

	return sqz_state
	


def sqz_vac_hack(xi, wgs, pos = [0,0], freq = ['sig', 'idl'], hg = [0, 0],  cutoff = 2, lcutoff = 0):
	"""
	#determinnistic pair photon source for testing / clock tree matching
	JCA 2023
	"""
	sqz_state = []
	amp = np.cdouble(0.999)
	sqz_state.append({'amp' : amp, 
				'wg' : [wgs[0], wgs[1]], 
				'occ' : [1,1],
				'pos' : [pos[0], pos[1]],
				'freq' : [freq[0], freq[1]],
				'hg' : [hg[0], hg[1]],
				'tran' : [1,1]})
	# sqz_state = [vec for vec in sqz_state if not len(vec['occ']) == 0]
	
	rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(3))
	vac = {'amp' : 1-amp, 
			'wg' : [wgs[0]], 
			'occ' : [0],
			'pos' : [-1],
			'freq' : ['vac_'+rs1],
			'hg' : [-1],
			'tran' : [-1]}
	sqz_state = [vac] + sqz_state

	return sqz_state
	
	
	
def sps(amp = 1, wgs = 'wg1', pos = 0, freq = 'idl', hg = 0):
	"""
	Generates a idealised quasi-deterministic single photon state with amplitude amp in the dictionary format in component wg at position pos 
	JCA 2023
	"""
	sps_state = []
	
	rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(3))
	sps_state.append({'amp' : np.sqrt(1. - np.cdouble(amp)**2), 
					'wg' : [wgs], 
					'occ' : [0],
					'pos' : [-1],
					'freq' : ['vac_'+rs1],
					'hg' : [-1],
					'tran' : [0]})
					
	sps_state.append({'amp' : np.cdouble(amp), 
					'wg' : [wgs], 
					'occ' : [1],
					'pos' : [pos],
					'freq' : [freq],
					'hg' : [hg],
					'tran' : [1]})

	return sps_state   



def state_checker(qstate):
	"""
	Checks that a given state (stored as a dictionary) is well formed.
	JCA 2023
	"""
	for vec in qstate:
		assert isinstance(vec['amp'], np.complex128), "quantum state must have float amplitude"
		assert isinstance(vec['wg'], list), "quantum state must have list of modes"
		assert isinstance(vec['occ'], list), "quantum state must have list of modes"
		assert isinstance(vec['pos'], list), "quantum state must have list of modes"
		coord_keys = list(vec.keys())
		coord_keys.remove('amp')
		modelabelens = []
		for key in coord_keys:
			# print(vec[key])
			assert isinstance(vec[key], list), "quantum state mode labels must be list"
			modelabelens.append(len(vec[key]))
			
		assert (modelabelens == [len(vec['wg']) for i in range(len(modelabelens))]), "quantum state unequal mode label lengths"

		#check for duplicate modes in vector
		modelabs = []
		if sum(vec['occ']) > 0:   #preserve vacuum state
			for key in vec.keys() - ['amp']:
				modelabs.append(vec[key])
			modelabs = list(zip(*modelabs))

			for _ in range (len(modelabs)):
				for __ in range (_):
					assert modelabs[_] != modelabs[__]; 'duplicate mode entry in vector'

	# print("state passed validity tests")



def genfockslist_core(kk, nn):
	""" 
	Genrates a list of fock states with k excitations in n modes
	from https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
	JCA 2023
	"""
	if kk == 1:
		yield [nn,]
	else:
		for value in range(nn + 1):
			for permutation in genfockslist_core(kk - 1, nn - value):
				yield [value,] + permutation
			
			
			
def genfockslist(k, n):
	"""
	generates fock states with n photons in k modes
	JCA 2023
	"""						
	ans = list(genfockslist_core(k, n))
	ans.reverse()
	ans = np.array(ans)
	return ans



def gendictoutputs(wg_mode_names, ivec, focks, outputpos):
	"""
	Generates a list of dictionarys composing a set of fock states for a given set of waveguide modes fock states as list, as given by the above genfockslist
	uses the above 
	JCA 2023
	"""
	dict = {}
	occs = []
	outpdict_list = []
	for fock in focks:
		fock_occ = [num for num in fock if num]
		occs.append(fock_occ)
		
		fock_pos = np.nonzero(fock > 0)[0]
		outpdict_list.append( {'wg' : [wg_mode_names[index] for index in fock_pos], 'occ' : fock_occ, 'pos' : [outputpos for i in range(len(fock_occ))]  }  )
		# for key in ivec.keys() - ['amp', 'wg', 'occ', 'pos']:
			# outpdict_list[-1][key] = ivec[key]
	return outpdict_list
		



def q_simplify (qstate, tol = 1e-10, verbose = False):
	"""
	adds together wavevectors with the same coordinates - does quantum inteference
	qstate must be in the list of dictionarys format
	JCA 2022
	"""
	for kk in range(len(qstate)):
		qstate[kk] = dict(sorted(qstate[kk].items(), reverse = True))	 # sorts modes lexicographically by mode name
	   
		# very hacky way to get rid of exchange symetry! we sort other labels by which waveguide,  they are in
		mylabels = [ qstate[kk][key] for key in sorted((qstate[kk].keys() - ['amp']), reverse = True)]
		keylist = [ key for key in sorted((qstate[kk].keys() - ['amp']), reverse = True)]
		mysortedlabels2 = list(zip(*sorted(zip(*mylabels), key=operator.itemgetter(0,1,2,3,4), reverse = True))) 
		for i in range(len(qstate[kk].keys() - ['amp'])): #degeneracy of representation / exchange symmetry (e.g. [wg0, wg0], [1,1], [idl, sig] <-> [wg0, wg0], [1,1], [sig, idl] )
			qstate[kk][keylist[i]] = list(mysortedlabels2[i])

  
	qstate_out = []
	my_coords = []
	sets = {}
	if verbose: print('state before simplify: \n', np.array(qstate))
	# which vecs are the same?
	for idx,vec in enumerate(qstate) :
		
		coord_keys = list(vec.keys())
		coord_keys.remove('amp')
		coords = [vec[coord_keys[k]] for k in range(len(coord_keys))]

		if coords not in my_coords :
			my_coords.append(coords)
			sets[str(my_coords[-1])] =  [idx]
		else :
			new_val=list(sets[str(coords)])
			new_val.append(idx)
			sets[str(coords)] = new_val
	
	setslist = list(sets.values())
	# print(setslist)
	if verbose: 
		print('\nsets: ', np.array(sets))
		print('\nsetslist: ',setslist)

	# add them and re-assemble qstate (interference)
	qstate_out = [0 for i in range(len(setslist))]
	for i in range(len(setslist)):
		qstate_out[i] = copy.deepcopy(qstate[setslist[i][0]])
		# qstate_out[i] = qstate[setslist[i][0]]
		qstate_out[i]['amp'] = 0.
		for idx in setslist[i]:
			qstate_out[i]['amp'] += qstate[idx]['amp']

	#delete zero amp, or zero length vecs
	qstate_out = [vec for vec in qstate_out if not len(vec['occ']) == 0]
	qstate_out = [vec for vec in qstate_out if not cmath.isclose(np.abs(vec['amp']), 0., abs_tol = tol)]
	
	for kk in range(len(qstate_out)):
		# print('\nbeforesort: ',qstate[kk])
		qstate_out[kk] = dict(sorted(qstate_out[kk].items(), reverse = True))

	return qstate_out	 




def cleanzeroes(qstate, tol = 1e-10):
	"""
	deletes zero-photon and empty-list parts of qstate
	JCA 2023
	"""

	vacwg = qstate[0]['wg'][0]
	vacampsum = np.sum([vec['amp'] for vec in qstate if sum(vec['occ']) == 0 ])
	# vacampsum2 = np.sum([vec['amp'] for vec in qstate if vec['wg']) == 0 ])
	# for j,vec in enumerate(qstate): #delete vacuum modes by -1 pos signifier
		# for i in len(range(qstate[j]['wg']:  
			# qstate[j]['wg'][i]
			
	for j,vec in enumerate(qstate): #delete vacuum modes by -1 pos signifier
		for key in qstate[j].keys() - ['amp', 'pos']:	
			qstate[j][key] = [mode for idx,mode in enumerate(qstate[j][key]) if qstate[j]['pos'][idx] != -1]
	
	for j,vec in enumerate(qstate):
		qstate[j]['pos'] = [ pos for pos in qstate[j]['pos'] if pos != -1] #remove extraneous pos indexes
				
				
	rs1 = ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(3)) # add vacuum state back in if empty state
	vacstate =  {'amp' : vacampsum,
				 'wg'   : [vacwg], 
				 'occ'   : [0],
				 'pos'   : [-1],
				 'freq' : ['vac_'+rs1],
				 'hg'   : [-1],
				 'tran' : [0]}
							
					
	qstate = [vacstate] + qstate

	# emptystate? no ampltude? delete!
	qstate = [vec for vec in qstate if not len(vec['occ']) == 0]
	qstate = [vec for vec in qstate if not cmath.isclose(np.abs(vec['amp']), 0., abs_tol = tol)]

	return qstate



def concat_vec (qstate1, qstate2):
	"""
	does tensor product of two states which have non-interfering components
	JCA 2022
	"""
	outp = []
	if qstate1 == []:
		outp = qstate2
		return outp
	if qstate2 == []:
		outp = qstate1
		return outp
		
	for i in range(len(qstate1)):
		qvec1 = copy.deepcopy(qstate1[i])
		# qvec1 = qstate1[i]
		amp1 = qvec1['amp']
		qvec1.pop('amp')
		keys1 = list(qvec1.keys())

		for j in range(len(qstate2)):
			qvec2 = copy.deepcopy(qstate2[j])
			# qvec2 = qstate2[j]
			amp2 = qvec2['amp']
			qvec2.pop('amp')
			keys2 = list(qvec2.keys())
			# print(keys1,"  ",keys2)
			assert set(keys1) == set(keys2); 'vectors must have the same keys'
			for key in keys1:
				qvec2[key] += qvec1[key]

			qvec2['amp'] = amp1*amp2
			outp.append(qvec2)
			
	return outp


def norm(qstate):   
	"""
	Computes the norm of the quantum state in our list of dictoinarys format
	JCA 2023
	"""
	return np.sum([np.abs(vec['amp'])**2 for vec in qstate])
	
	

def qcruncher(qstate, U, myU_modes, outputfocks = None, thresh = 0.01, verbose = False):
	"""
	takes in fock states and a unitary on spatial modes ('wg') and computes the result
	TODO: this is a beast! refactor this into smaller chunks so that it is legible 
	JCA 2023
	"""
	assert len(U) == len(myU_modes), "Error! Unitary and mode list incorrectly specified"
	
	# are we normalised?
	inp_norm = 0
	for vec in qstate:
		inp_norm += np.abs(vec['amp'])**2
	if verbose:
		if inp_norm < 1 - thresh:
			print("warning, state norm less than {}".format(1 - thresh))
		print("L-2 norm of input was: {}".format(inp_norm))

	
	#check qstate has modes in common with U 
	vecmodes = []
	for vec in qstate:
		vecmodes += vec['wg']
		vecmodes = list(dict.fromkeys(vecmodes))
	if vecmodes == []:
		print('no modes in common between vec and U')
		return qstate

	# process each vector (dictionary) in input state one by one, later run over all possible output vecs and compute transition amplitude
	outputstate = []
	for ivec in qstate:
		vecwgs = ivec['wg']
		unqvecwgs = sorted(list(set(ivec['wg']).difference(myU_modes))) # vectors not involved in U
		intvecs = sorted(list(set(ivec['wg']).intersection(myU_modes))) # vectors not involved in U
		
		if verbose:
			print('\nunqvecs', unqvecwgs)
			print('intvecs', intvecs)
		
		
		# No modes in common? bypass
		if intvecs == []:
			outputstate.append(ivec)

		# one-dimensional unitary? easy!
		elif len(U) == 1:
			outputstate.append(ivec)
			outputstate[-1]['amp'] = ivec['amp'] * U[0][0]
		elif sum (ivec['occ']) == 0:
			outputstate.append(ivec)
			
			
		# non-trivial U? find mode overlaps, generate sub U and compute permanents
		else:	
			modelabels=[]
			ridx=0
			int_idx = []
			wg_ccomp = []
			
			# what are the mode labels in the input state? which parts will interfere?
			if verbose:
				print('vector components involved in U')
			for idx,wg in enumerate(vecwgs) :
				if vecwgs[idx] in myU_modes:
					modelabels.append([])
					wg_ccomp.append(wg)
					activemodelabels = ivec.keys() - ['amp', 'occ', 'wg']		   # any mode label can provide (non)orthogonality. 'wg' is inbuilt to unitary it applies to
					for key in activemodelabels:
						modelabels[ridx].append (ivec[key][idx])
					if verbose:
						print(idx, "	",wg,"	", activemodelabels, "   ", modelabels[ridx])
					int_idx.append(idx)
					ridx += 1

			# find parts of input state that will interfere (have same mode labels)
			samemodeq = [[0 for j in range(len(modelabels) )] for i in range(len(modelabels))]		
			for i in range(len(modelabels)):
				for j in range(len(modelabels)):
					samemodeq[i][j] = int(modelabels[i] == modelabels[j])

			samemodeq = np.transpose(np.array(samemodeq))
			g = nx.from_numpy_matrix(samemodeq)
			g_connex = nx.connected_components(g)	
			g_connex_compnts = [list(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
			if verbose:
				print('\nconnected components: ',g_connex_compnts)	
			# connected components of state will interfere. There may be many independent ones. These are handled independently and joined together later with concat_vec() (tensor product)
			# computes permanents of many small (typically 2x2) matrices, rather than permanents of sparse, larger matrices.

			cc_states = [[] for ccomp in g_connex_compnts]
			for cc_states_idx, ccomp in enumerate(g_connex_compnts) :
				
				occinUmodes = 0 
				for cc in ccomp: 
					occinUmodes += ivec['occ'][int_idx[cc]]   

				# what are the possible output states?
				focks = genfockslist(len(myU_modes), occinUmodes)
				outputfocks = gendictoutputs(myU_modes, ivec, focks, -5) # put them in dictionary format

				# keep track of number of photons in this connected component
				subU_iidx = []
				i_occs = [] 
				for cc in ccomp:   
					i_occs.append(ivec['occ'][int_idx[cc]] )
					for i in range(ivec['occ'][int_idx[cc]] ):
						subU_iidx.append(myU_modes.index(ivec['wg'][int_idx[cc]])) 

				# compute transition amplitude for each output possible output vector
				for ovec in outputfocks:
					subU_oidx = []
					for idx,wgs in enumerate(ovec['wg']):
						for j in range(ovec['occ'][idx]):			# handle repeated row/columns
							subU_oidx.append(myU_modes.index(wgs))


					subU = U[np.ix_(subU_oidx,subU_iidx)] # find submatrix
					if verbose:
						print("input fock: ",ivec)
						print("activemodelabels", activemodelabels,  modelabels[list(ccomp)[0]])
						print("component occupation: ", occinUmodes)
						print("outputfocklist: ")
						print(np.array(outputfocks))
						print("\n")
						print("ioccs: ",i_occs)
						print("\noutput fock : ",ovec)
						print("input index: ",subU_iidx)
						print("output index: ",subU_oidx)
						print(subU)
					
					# boson sampling part transision amplitude calculation
					denom1 = np.prod(list(map(np.math.factorial, ovec['occ'])))
					denom2 = np.prod(list(map(np.math.factorial, i_occs)))	   

					if len(subU) == 1 and len(subU[0]) == 1:
						trans_amp = subU[0][0]/np.sqrt(denom1*denom2)
					else:
						trans_amp = tw.perm(subU)/np.sqrt(denom1*denom2)
					
					if verbose:
						print("transition amplitude was:", trans_amp)

					 # rebuilds output vector with corrct mode labesl
					if abs(trans_amp) != 0:		
						outputvecdict = {'amp' : trans_amp, 
						'wg' : ovec['wg'], 
						'occ' : ovec['occ']}

						for idx,modelabel in enumerate(list(activemodelabels)):
							outputvecdict[modelabel] = [modelabels[list(ccomp)[0]][idx] for _ in range(len(list(ovec['wg'])))]
					   
						cc_states[cc_states_idx].append(outputvecdict)

			# re-label the different connected components so they can be joined
			for i in range(len(cc_states[0])):

				cc_states[0][i]['amp'] =  ivec['amp']*cc_states[0][i]['amp']
				nonampmodes = ivec.keys() - ['amp']
				re_build_list = [x for x in range(len(ivec['occ'])) if x not in int_idx]
				
				for mode in nonampmodes :
					for _ in re_build_list :
						cc_states[0][i][mode] += [ivec[mode][_]]
							

			# recconnect all of the un-interfereing parts (different connected components) into one big state
			for l in range(1, len(cc_states)):
				cc_states[0] = concat_vec(cc_states[0], cc_states[l])

			# add components to outp
			for i in range(len(cc_states[0])):
				outputstate.append(cc_states[0][i])
	# simplify state (does the quantum interference, potentially unnormalised before this)
	outputstate = cleanzeroes(outputstate)
	outputstate = q_simplify(outputstate,  verbose = False )

	# check norm of output state
	outp_norm = 0.
	if verbose:
		for vec in outputstate:
			outp_norm += np.abs(vec['amp'])**2 
		print("L-2 norm of output was: {:}".format(outp_norm),"\n")   
	state_checker(outputstate)
	return outputstate