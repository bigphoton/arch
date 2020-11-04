"""
Functions and objects describing electro-optic components.
"""


import numpy as np
from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.electro_optic.digital.coherent import switch_basic

class switch_2x2(base_block):
	"""
	extinction_ratio: ratio of desired signal to undesired signal from wrong port
	loss_dB: positive number of decibels of loss (0 dB -> 100% tx; 10 dB -> 10% tx)
	"""
	
	reference_prefix = "SW"
	
	def define(self, loss_dB = 3.0, extinction_ratio=1000.0):
		
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
		
		# ...and a digital input
		name = "DIG"
		self.ports.add(port(name, "digital", True, self, None, 1, (x0,h/2+y0), 0))
		self.in_port_order.append(name)
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		
		# Setup model matrix list
		def model_matrix_func_off(loss_dB=loss_dB, extinction_ratio=extinction_ratio):
			leak = 1/extinction_ratio
			tx = 10**(-loss_dB/10)
			r = np.sqrt(tx*(1-leak))
			t = np.sqrt(tx*leak) * 1j
			m = np.array([ [r, t], 
						   [t, r] ])
			return m
		
		def model_matrix_func_on(loss_dB=loss_dB, extinction_ratio=extinction_ratio):
			leak = 1/extinction_ratio
			tx = 10**(-loss_dB/10)
			r = np.sqrt(tx*leak)
			t = np.sqrt(tx*(1-leak)) * 1j
			m = np.array([ [r, t], 
						   [t, r] ])
			return m
			
		model_matrix_func_list = [model_matrix_func_off, model_matrix_func_on]
		
		# Model parameter(s)
		self.model_params.update({'extinction_ratio':extinction_ratio})
		
		# Set model
		self.model = switch_basic(model_matrix_func_list, self.model_params)