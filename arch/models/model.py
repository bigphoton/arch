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