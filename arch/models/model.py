"""
Functions and objects describing optical components.
"""

import abc


class model(abc.ABC):
	"""
	Model base class.
	"""
	
	@abc.abstractmethod
	def update_params(self, new_params):
		"""
		Update compact model (e.g. matrix) with new parameters, such that model.compute() gives
		result based on these new parameters.
		
		Subclasses must implement this method.
		"""
		pass


	@abc.abstractmethod
	def compute(self):
		"""
		Propagate input state to output state.
		
		Subclasses must implement this method.
		"""
		pass
	
	
	@property
	@abc.abstractmethod
	def n_inputs(self):
		"""
		Computed number of model inputs.
		
		Subclasses must implement this method.
		"""
		pass
	
	
	@property
	@abc.abstractmethod
	def n_outputs(self):
		"""
		Computed number of model outputs.
		
		Subclasses must implement this method.
		"""
		pass



from collections import deque


class delayed_model(object):
	"""
	Wrapper class for adding fixed delay to each input.
	
	original_model: class of original model to be instantiated
	delays: singleton int (all inputs same), or list of int number of delay time-steps
			with length same as input length.
	args, kwargs: args and kwargs for original model initialisation
	"""
	
	def __init__(self, original_model, delays, *args, **kwargs):
		
		self._model = original_model(*args, **kwargs)
		
		# Start at t=0 
		# Integer
		self.t = 0
		
		self.delays = delays
		for i in range(len(self.delays)):
			self.delays[i] += 1
			if self.delays[i] < 1:
				raise AttributeError("All elements of delays must be > 0.")
		
		# This could be expanded spectrally by making `nt` a tuple
		# Element 0 is the newest input value
		# Element -1 is the oldest input value
		self.input_time_series = [deque(maxlen=d) for d in delays]
		
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def compute(self, input_vector):
		
		# Advance the time by one step
		self.t += 1
		
		
		# Get values from ports' historical values
		# Handle each time series (for each input/output port)
		vin = []
		for i in range(self.n_modes):
			
			# Get the final input vector to multiply
			vin.append(self.time_serieses[i].pop())
		
			# Store the input vector
			self.time_serieses[i].insert(0, input_vector[i].value)
		
		m = self.model_matrix
		
		# Do matrix multiplication
		vout = m @ vin
		
		return vout.flat