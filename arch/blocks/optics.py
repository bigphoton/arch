"""
Functions and objects describing optical components.
"""


import numpy as np
from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.optical.classical.linear import linear_basic
from arch.models.optical.quantum.permanents import permanent
from arch.models.optical.quantum.permanents_quantum import permanent_quantum

class beamsplitter(base_block):

	"""Requires the user to pick a model of computation. Options are 'linear', 'monte_carlo', 'full_quantum'
	Defaults to linear"""
	
	reference_prefix = "BS"
	
	def define(self, model_choice, reflectivity=0.5):
		
		self.model_choice=model_choice

		# Setup ports
		w = generic_box.box_width
		h = generic_box.box_height
		l = generic_port.port_length
		x0,y0 = self.position
		
		
		# Add two input ports
		n_in = 2
		for n in range(n_in):
			name = "IN"+str(n)
			self.ports.add(port(name, "optical", True, self, 1, (-w/2+x0-l,-h/2+(n+1/2)*h/n_in+y0), 0))
			self.in_port_order.append(name)
		
		# ...and two outputs
		n_out = 2
		for n in range(n_out):
			name = "OUT"+str(n)
			self.ports.add(port(name, "optical", False, self, 1, (+w/2+x0+l,-h/2+(n+1/2)*h/n_out+y0), 180))
			self.out_port_order.append(name)
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		
		# Setup model matrix
		def model_matrix_func(reflectivity):
			r = np.sqrt(reflectivity)
			t = np.sqrt(1-r**2) * 1j
			m = np.array([ [r, t], 
						   [t, r] ])
			
			return m
		
		# Model parameter(s)
		self.model_params.update({'reflectivity':reflectivity})
		
		# Set model
		if model_choice=='monte_carlo':
			self.model = permanent(model_matrix_func, self.model_params,model_choice)
		elif model_choice=='linear':
			self.model= linear_basic(model_matrix_func, self.model_params)
		elif model_choice=='full_quantum':
			self.model= permanent_quantum(model_matrix_func, self.model_params, model_choice)
		else:
			raise Exception('This is not a valid model choice for a beamsplitter')

		
class phase_shift(base_block):
	
	reference_prefix = "P"
	
	def define(self, phase=0):
		
		# Internal variable(s)
		self.phase = phase
		
		# Setup ports
		w = generic_box.box_width
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add ports
		self.ports.add(port("IN", "optical", True,  self, 1, (-w/2+x0-l,y0), 0))
		self.ports.add(port("OUT", "optical", False, self, 1, (+w/2+x0+l,y0), 180))
		
		self.in_port_order = ["IN"]
		self.out_port_order = ["OUT"]
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		
		# Setup model matrix
		def model_matrix_func(phase):
			return np.array([[np.exp(1j*phase)]])
		
		# Model parameter(s)
		self.model_params.update({'phase':phase})
		
		# Set model
		self.model = linear_basic(model_matrix_func, self.model_params)