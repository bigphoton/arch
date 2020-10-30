"""
Functions and objects describing optical components.
"""

import numpy as np
from arch.models import model

class on_click_pair_source(model):
	"""
	Simple model for an on demand pair source. If digital input signal recieved, output a pair of photons in two spatial modes.
	"""
	
	def __init__(self,  model_params):
		super(type(self), self).__init__()
		
		self.model_matrix = model_params
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def compute(self, input_vector):
		# Get values from ports
		vin = np.array([e.value for e in input_vector])

		if vin==1:
			vout=np.array([1,1])
		elif vin==0:
			vout=np.array([0,0])
		else:
			raise Exception(' On click source can only currently take binary input')
		
		return vout.flat