"""
Functions and objects describing methods of simulation such as monte-carlo.
"""

import abc
import numpy as np
from sympy import Matrix
from arch.connectivity import Connectivity
import copy
import math
import importlib.util
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
		
	




		


class DynamicalSimulator(Simulator):
	"""
	Base class for simulations which evolve with time.
	"""
	
	def define(self, t_start=0, t_stop=0, t_step=1.0,
				in_time_funcs={}, 
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
		
		self.in_time_funcs = in_time_funcs
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
		port_delay_ints  = {p:0 for p in self.all_ports}
		port_delay_ints |= {p:round(self.get_delay_func(b)/self.t_step) 
									for b in self.blocks for p in b.in_ports}
		
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
			for p in self.ports:
				if p in self.in_time_funcs:
					state[p] = self.in_time_funcs[p](t-self.get_delay_func(p.block))
				else:
					state[p] = delayed_port_value(p)
			
			# Step through models and calculate output port values
			for m in models:
				# Update output using delayed inputs
				o = m.out_func(state)
				state |= o
			
			# Update `out_state` to contain external in values at time `t` (not delayed)
			for p in self.in_ports:
				if p in self.in_time_funcs:
					state[p] = self.in_time_funcs[p](t)
			
			# Propagate values along connectivity
			state |= {pi:state[po] for po,pi in con if po in state}
			
			# Store state in time series
			state_history.append(state)
		
		state_history.pop(0)
		
		self.time_series = state_history
		self.times = ts
		
		return state_history
		
		
		
		
import networkx as nx

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