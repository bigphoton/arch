"""
Functions and objects describing abstract components.
 
Structure
=========

base_block
 * Contains compact model
 * Takes input parameters
 * Has input and output ports
 * Can compute output from input state

complex_optic
 * Inherits from base_block
 * Contains any number of base_block and complex_optic's
 * Contains a map of internal io ports
 * Can compactly model contents
 
"""

from .vis.generic import generic_box
from .vis.graphic import v2
from .port import port_set
import numpy as np


class base_block:
	"""
	Base class for optical components.
	
	Subclass this class to implement different devices.
	
	Subclasses should implement:
	----------------------------
	 * reference_prefix: str, class property
	 * graphic: graphic
	 * ports: port_set
	 * model: dict
	 * in_port_map: dict
	 * out_port_map: dict
	 * supported_models: list
	 
	Instance data
	=============
	
	
	Class data
	==========
	reference_designators: dict(set). Keys are reference_prefix's, sets are indices
	
	Wishlist
	========
	 * Calculates its own compact model
	"""
	
	# Dictionary of reference designators for all blocks
	reference_designators = dict()
	
	# To be overridden by subclasses:
	reference_prefix = "_"
	
	def __init__(self, **kwargs):
		
		# Model parameters to track
		# This must go first, so we can compare incoming sets against it, later in __init__
		self.model_params = dict()
		
		# Handle reference designator generation
		try:
			existing_indices = base_block.reference_designators[self.reference_prefix]
			self.reference_index = max(existing_indices) + 1
			
		except KeyError:
			self.reference_index = 0
			base_block.reference_designators.update( {self.reference_prefix:set()} )
		
		base_block.reference_designators[ self.reference_prefix ].add( self.reference_index )
		
		self.reference_designator = self.reference_prefix + str(self.reference_index)
		
		# Placeholder ports set
		self.__ports = port_set()
		
		# Placeholder port maps
		self.in_port_order = []
		self.out_port_order = []
		
		# TODO: set position intelligently, or hide graphic before its position is set
		self.__position = v2(0,0)
		
		# Run subclass define routine
		#  Pass kwargs to define to initialise variables as required
		# TODO: check everything is set up correctly by self.define()
		# TODO: check no model_params are named stupidly (e.g. position)
		self.define(**kwargs)
		
		# Handle graphics
		# Set default graphic if none present
		if not hasattr(self, "graphic"):
			self.graphic = generic_box(self.reference_designator, position=self.position, angle=0)
		
		# Ensure model_params are set as attributes
		for key in self.model_params:
			self.__setattr__(key, self.model_params[key])
		
	
	def __setattr__(self, name, value):
		# Store the original version of this attribute
		try:
			old_value = self.__getattribute__(name)
		except:
			old_value = None
		
		# Set the attribute in super
		object.__setattr__(self, name, value)
		
		# If the attribute is a model parameter, update the model
		if name in self.model_params and value != old_value:
			self.model_params[name] = value
			self.model.update_params(self.model_params)
	
	
	@property
	def position(self):
		return self.__position
	
	
	@position.setter
	def position(self, new_position):
		new_position = v2(*new_position)
		# Update our graphic's position
		self.graphic.position = new_position
		# Update our ports' positions
		for port in self.ports:
			port.graphic.position -= self.__position
			port.graphic.position += new_position
			for con_port in port.connected_ports:
				con_port.graphic.update_path()
		# Update our position
		self.__position = new_position
	
	
	@property
	def ports(self):
		return self.__ports
	
	@ports.setter
	def ports(self, new_ports):
		self.__ports = port_set(new_ports)
		
	
	@property
	def in_ports(self):
		"""
		Convenience access to a filtered dict of input ports only
		"""
		return port_set({p for p in self.ports if p.is_input})
	
	
	@property
	def out_ports(self):
		"""
		Convenience access to a filtered list of output ports only
		"""
		return port_set({p for p in self.ports if p.is_output})
	
	
	@property
	def in_state(self):
		"""
		Convenience access to values across input ports.
		"""
		return {p.name:p.value for p in self.ports if p.is_input}
	
	
	@property
	def out_state(self):
		"""
		Convenience access to values across output ports.
		"""
		return {p.name:p.value for p in self.ports if p.is_output}
	
	
	def define(self, **kwargs):
		"""
		Method to be overridden by subclasses.
		
		Must populate:
		 - self.reference_prefix
		 - self.ports
		 - self.in_port_order
		 - self.out_port_order
		
		Optionally populate:
		 - self.graphic
		"""
		pass
	
	
	def compute(self):
		"""
		Method to propagate state from component input ports to output ports.
		"""
		#TODO: Have a more appropriate method top propogate through a state in the fully quantum case
		# - deoesn't really make sense to have two output ports with distinct values 

		# Compose the input vector based on the input port order
		# TODO: Would be great if this happened automatically when the input ports were updated
		vin = np.array([self.ports[name] for name in self.in_port_order])
		# Compute the output vector from the model
		vout = self.model.compute(vin)
		# Update the output ports based on the vector elements and the output port order
		for i,value in enumerate(vout):
			self.ports[self.out_port_order[i]].value = value
	
	
class complex_block(base_block):
	"""
	Optical component combining several other optical components.
	"""
	pass