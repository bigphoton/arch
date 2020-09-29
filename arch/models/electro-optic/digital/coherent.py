"""
Functions and objects describing digital-input coherent-effect switches.
"""

import numpy
from arch.models import model

class basic(model.model):
	"""
	Digital input {1,0} single-mode unitary output model.
	unitary_matrix_func: function returning n-by-n complex numpy array; should be unitary
	"""
	
	def __init__(self, unitary_matrix_func, model_params):
		super(type(self), self).__init__()
		
		self.unitary_matrix_func = unitary_matrix_func
		self.model_matrix = self.unitary_matrix_func(**model_params)
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def compute(self, input_vector):
		# Get values from ports
		vin = numpy.array([e.value for e in input_vector])
		
		m = self.model_matrix
		
		# Do matrix multiplication
		vout = m @ vin
		
		return vout.flat