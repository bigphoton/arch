"""
Functions and objects describing single photon sources
"""

import numpy as np
from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.optical.quantum.sources import on_click_pair_source

class black_box_pair_photon_source(base_block):

	"""
	A simple button click pair photon source.
	Digital button click input, two single photon output channels

	Desired class attributes: 
	"""

	reference_prefix="SPS"

	def define(self):
		#setup display box
		w=generic_box.box_width
		h=generic_box.box_height
		l=generic_port.port_length
		x0,y0=self.position

		#add single optical input
		name="IN"
		self.ports.add(port(name,"digital", True, self, 1 ,(x0,h/2+y0),0))
		self.in_port_order.append(name)

		# ...and two outputs
		n_out = 2
		for n in range(n_out):
			name = "OUT"+str(n)
			self.ports.add(port(name, "optical", False, self, 1, (+w/2+x0+l,-h/2+(n+1/2)*h/n_out+y0), 180))
			self.out_port_order.append(name)


		#setup graphic
		self.graphic=generic_box(self.reference_designator,position=self.position)


		#set model
		self.model=on_click_pair_source(self.model_params)