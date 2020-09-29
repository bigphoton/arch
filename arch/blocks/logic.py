"""
Functions and objects describing electrical and electronic physics and
components.

Wishlist
--------
 * Electronic delays
 * Signal distortion
 * SPICE integration
"""


from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.electrical.digital.logic import combinatorial


class not_gate(base_block):
	"""
	A not gate: 0 -> 1, 1 -> 0
	"""
	
	reference_prefix = "U"
	
	def define(self):
		
		# Setup ports
		w = generic_box.box_width
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add ports
		self.ports.add(port("IN", "digital", True,  self, 1, (-w/2+x0-l,y0), 0))
		self.ports.add(port("OUT", "digital", False, self, 10, (+w/2+x0+l,y0), 180))
		
		self.in_port_order = ["IN"]
		self.out_port_order = ["OUT"]
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		# Set model
		self.model = combinatorial([1,0], 1)
		

class and_gate(base_block):
	"""
	An and gate: 00 -> 0, 01 -> 0, 10 -> 0, 11 -> 1
	"""
	
	reference_prefix = "U"
	
	def define(self):
		
		# Setup ports
		w = generic_box.box_width
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add ports
		self.ports.add(port("IN0", "digital", True,  self, 1, (-w/2+x0-l,+20+y0), 0))
		self.ports.add(port("IN1", "digital", True,  self, 1, (-w/2+x0-l,-20+y0), 0))
		self.ports.add(port("OUT", "digital", False, self, 10, (+w/2+x0+l,y0), 180))
		
		self.in_port_order = ["IN0","IN1"]
		self.out_port_order = ["OUT"]
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		# Set model
		self.model = combinatorial([0,0,0,1], 1)


class or_gate(base_block):
	"""
	An or gate: 00 -> 0, 01 -> 1, 10 -> 1, 11 -> 1
	"""
	
	reference_prefix = "U"
	
	def define(self):
		
		# Setup ports
		w = generic_box.box_width
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add ports
		self.ports.add(port("IN0", "digital", True,  self, 1, (-w/2+x0-l,+20+y0), 0))
		self.ports.add(port("IN1", "digital", True,  self, 1, (-w/2+x0-l,-20+y0), 0))
		self.ports.add(port("OUT", "digital", False, self, 10, (+w/2+x0+l,y0), 180))
		
		self.in_port_order = ["IN0","IN1"]
		self.out_port_order = ["OUT"]
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		# Set model
		self.model = combinatorial([0,1,1,1], 1)


class nand_gate(base_block):
	"""
	A nand gate: 00 -> 1, 01 -> 1, 10 -> 1, 11 -> 0
	"""
	
	reference_prefix = "U"
	
	def define(self):
		
		# Setup ports
		w = generic_box.box_width
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add ports
		self.ports.add(port("IN0", "digital", True,  self, 1, (-w/2+x0-l,+20+y0), 0))
		self.ports.add(port("IN1", "digital", True,  self, 1, (-w/2+x0-l,-20+y0), 0))
		self.ports.add(port("OUT", "digital", False, self, 10, (+w/2+x0+l,y0), 180))
		
		self.in_port_order = ["IN0","IN1"]
		self.out_port_order = ["OUT"]
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		# Set model
		self.model = combinatorial([1,1,1,0], 1)