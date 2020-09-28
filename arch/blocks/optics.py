"""
Functions and objects describing optical components.
"""


from ..block import base_block
import numpy as np
from ..vis.generic import generic_box, generic_port
from ..port import port

class beamsplitter(base_block):
	
	reference_prefix = "BS"
	
	def define(self, reflectivity=0.5):
		
		# Internal variable(s)
		self.reflectivity = reflectivity
		
		# Setup ports
		w = generic_box.box_width
		h = generic_box.box_height
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add two input ports
		n_in = 2
		for n in range(n_in):
			self.ports.add(port("IN"+str(n), "optical", True,  self, 1, (-w/2+x0-l,-h/2+(n+1/2)*h/n_in+y0), 0))
		
		# ...and two outputs
		n_out = 2
		for n in range(n_out):
			self.ports.add(port("OUT"+str(n), "optical", False, self, 1, (+w/2+x0+l,-h/2+(n+1/2)*h/n_out+y0), 180))
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
	
	def compute(self):
		# Model matrix
		r = np.sqrt(self.reflectivity)
		t = np.sqrt(1-r**2) * 1j
		m = np.array([ [r, t], 
					   [t, r] ])
		
		# Input vector
		vin = np.array([[self.ports['IN0'].value],
						[self.ports['IN1'].value]])
		
		# Output vector
		vout = m @ vin
		
		# Set output port values
		self.ports['OUT0'].value = vout.flat[0]
		self.ports['OUT1'].value = vout.flat[1]
		
		
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
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
	
	def compute(self):
		
		self.ports['OUT'].value = np.exp(1j*self.phase) * self.ports['IN'].value