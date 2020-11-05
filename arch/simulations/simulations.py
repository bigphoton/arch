"""
Functions and objects describing methods of simulation such as monte-carlo.
"""

import abc


class simulations(abc.ABC):
	"""
	Simulations base class.
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
	def simulate(self):
		"""
		Propagate input state to output state.
		
		Subclasses must implement this method.
		"""
		pass