"""
Functions and objects describing electro-optic components.
"""

from ..block import base_block
import numpy as np
	
	
class switch(base_block):
	def __init__(self):
		self.reference_prefix = "X"
		self.supported_models = ["incoherent"]
		self.n_spatial_modes = 2
		
		super(type(self), self).__init__()
	
	def model_matrix(self, state = 0):
		if state == 0:
			m = np.array([[1.0, 0.0], [0.0, 1.0]])
		elif state == 1:
			m = np.array([[0.0, 1.0], [1.0, 0.0]])
		
		return m