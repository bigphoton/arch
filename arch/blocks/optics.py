"""
Functions and objects describing optical components.
"""


import numpy as np
from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.optical.classical.linear import linear_basic

class beamsplitter(base_block):
	
	reference_prefix = "BS"
	
	def define(self, reflectivity=0.5):
		
		# Setup ports
		w = generic_box.box_width
		h = generic_box.box_height
		l = generic_port.port_length
		x0,y0 = self.position
		
		
		# Add two input ports
		n_in = 2
		for n in range(n_in):
			name = "IN"+str(n)
			self.ports.add(port(name, "optical", True, self, None, 1, (-w/2+x0-l,-h/2+(n+1/2)*h/n_in+y0), 0))
			self.in_port_order.append(name)
		
		# ...and two outputs
		n_out = 2
		for n in range(n_out):
			name = "OUT"+str(n)
			self.ports.add(port(name, "optical", False, self, None, 1, (+w/2+x0+l,-h/2+(n+1/2)*h/n_out+y0), 180))
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
		self.model = linear_basic(model_matrix_func, self.model_params)
		
		
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
		self.ports.add(port("IN", "optical", True,  self, None, 1, (-w/2+x0-l,y0), 0))
		self.ports.add(port("OUT", "optical", False, self, None, 1, (+w/2+x0+l,y0), 180))
		
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