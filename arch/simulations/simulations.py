"""
Functions and objects describing methods of simulation such as monte-carlo.
"""

import abc
import numpy


class Simulator(abc.ABC):
	"""
	Base class for simulations.
	"""
	
	def __init__(self, blocks=[], connectivity=Connectivity(), **kwargs):
		
		self.blocks = blocks
		self.connectivity = connectivity
		
		self.define(**kwargs)
	
	
	@connectivity.setter
	def connectivity(self, con):
		self.__connectivity = con
		
		# Update our port trackers to match the new connectivity
		self.ports = con.external_ports
		self.in_ports = con.external_in_ports
		self.out_ports = con.external_out_ports
	
	
	@property
	def connectivity(self):
		return self.__connectivity
	
	
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
	
	def define(self, t_start=0, t_stop=0, t_step=1.0, in_time_funcs={}, **kwargs):
		"""
		t_start: simulation start time, seconds
		t_stop: simulation stop time, seconds
		t_step: simulation time step size, or scale for adaptive simulators
		in_time_funcs: dict of funcs of one variable (time, seconds)
		"""
		
		self.t_start = t_start
		self.t_stop = t_stop
		self.t_step = t_step
	
	
	@classmethod
	def _time_range(cls, t_start, t_stop, t_step):
		"""
		Return standardised time-sequence from start and stop times and time step.
		"""
		return numpy.arange(t_start, t_stop, t_step, dtype=numpy.float64)
		

	@abc.abstractmethod
	def run(self):
		"""
		Propagate input state to output state.
		
		Subclasses must implement this method.
		"""
		pass