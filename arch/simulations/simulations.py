"""
Functions and objects describing methods of simulation such as monte-carlo.
"""

import abc
import numpy
from arch.connectivity import Connectivity


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
		return numpy.arange(self.t_start, self.t_stop, self.t_step, dtype=numpy.float64)
	
	
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