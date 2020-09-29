"""
Functions and objects describing optical components.
"""

class model(object):
	"""
	Model base class.
	"""
	
	def update_params(self, new_params):
		"""
		Update compact model (e.g. matrix) with new parameters, such that model.compute() gives
		result based on these new parameters.
		
		Subclasses must implement this method.
		"""
		pass
	
	
	def compute(self):
		"""
		Propagate input state to output state.
		
		Subclasses must implement this method.
		"""
		pass