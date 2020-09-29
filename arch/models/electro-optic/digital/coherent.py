"""
Functions and objects describing digital-input coherent-effect switches.
"""

import numpy
from arch.models import model

class switch_basic(model.model):
	"""
	Digital input single-mode unitary output model.
	matrix_func_list: list or dict of functions returning n-by-n complex numpy array;
						one for each digital input; must be subscriptable with digital values
	
	Last element in input vector is digital switch value (type int).
	"""
	
	def __init__(self, matrix_func_list, model_params):
		super(type(self), self).__init__()
		
		self.matrix_func_list = matrix_func_list
		
		n = len(self.matrix_func_list)
		self.model_matrix_list = [self.matrix_func_list[i](**model_params) for i in range(n)]
	
	
	def update_params(self, new_params):
		n = len(self.matrix_func_list)
		self.model_matrix_list = [self.matrix_func_list[i](**model_params) for i in range(n)]
	
	
	def compute(self, input_vector):
		# Get digital value from port
		switch_value = int(input_vector[-1].value)
		
		# Get optical values from ports
		vin = numpy.array([e.value for e in input_vector[:-1]])
		
		m = self.model_matrix_list[switch_value]
		
		# Do matrix multiplication
		vout = m @ vin
		
		return vout.flat