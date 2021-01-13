"""
Functions and objects describing abstract components.
 
Structure
=========

New-style blocks
----------------

Block
 * Represents 'abstract' or 'prototype' block
 * Specifies compact model
 * Specifies possible connections/params and their properties
 * Specifies connection/parameter default values
 * Designed to be subclassed by users

BlockReference
 * Created by the Block.ref method
 * Represents 'physical' block
 * Can be multiple BlockReference's from same Block
 * Tracks parameter values
 * Tracks connections to other references
 * Provides methods for accessing subsets of connections
 * Provides methods for computing output


Old-style blocks
----------------

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

import abc

import arch.port as port
from .port import var
from arch.models import Model

import copy

class Block(abc.ABC):
	"""
	 * Represents 'abstract' or 'prototype' block
	 * Specifies compact model
	 * Specifies possible connections/params and their properties
	 * Specifies connection/parameter default values
	 
	 Subclasses must:
	 - Define the reference_prefix attribute
	 - Implement the define method
	 """
	 
	# Dictionary of reference designators for all blocks
	names = dict()
	
	# To be overridden by subclasses:
	reference_prefix = "_"
	
	def __init__(self, _copy=False, **kwargs):
		
		# Handle reference designator generation
		self._setup_new_name()
		
		# Store the init kwargs for use by copier
		self.__init_kwargs = kwargs
		
		# Placeholder lists
		self.__ports = list()
		self.__models = list()
		
		# Run subclass define routine
		if not _copy:
			self.define(**kwargs)
		
		# Prevent post-define() modification
		self._inited = True
		
		# Assign an order to input and output ports and models
	
	
	@abc.abstractmethod
	def define(self, **kwargs):
		"""
		Method overridden by subclasses to implement the block. kwargs are
		passed directly from __init__.
		"""
		pass
	
	
	def _setup_new_name(self):
		
		assert self.reference_prefix != "_", "reference_prefix must be set by all Block subclasses."
		
		try:
			existing_indices = Block.names[self.reference_prefix]
			self.reference_index = max(existing_indices) + 1
			
		except KeyError:
			self.reference_index = 0
			Block.names.update( {self.reference_prefix:set()} )
		
		Block.names[ self.reference_prefix ].add( self.reference_index )
		
		self.name = self.reference_prefix + str(self.reference_index)
	
	
	def add_port(self, name, kind, direction, default=None):
		"""
		Initialise a port, use it in this block.
		"""
		assert type(kind) is port.kind
		assert type(direction) is port.direction
		if hasattr(self, '_inited'):
			raise RuntimeError("Block modifications not allowed outside define().")
		
		if default is None:
			default = port.KIND_DEFAULTS[kind]
		
		sym = var(name, block=self,
					kind=kind, direction=direction, default=default)
		
		# Make this data accessible through the ports list, as an attribute, and as return
		self.__ports.append(sym)
		self.__setattr__(name, sym)
		return sym
	
	
	def use_port(self, name, original):
		"""
		Repurpose `original` port for use in this block. Call it `name`.
		This method is designed to allow subclasses which contain other Block instances to
		expose the ports of those instances at the top level.
		"""
		assert isinstance(name, str)
		assert isinstance(original, var)
		if hasattr(self, '_inited'):
			raise RuntimeError("Block modifications not allowed outside define().")
		
		original.block = self
		original.name = var.new_name(self.name, name)
		
		self.__ports.append(original)
		self.__setattr__(name, original)
		
		return original
		
	
	def add_model(self, model):
		assert isinstance(model, Model)
		if hasattr(self, '_inited'):
			raise RuntimeError("Block modifications not allowed outside define().")
		
		self.__models.append(model)
	
	
	def __copy__(self):
		"""Copy routine. Copying is forbidden."""
		raise RuntimeError("Shallow copying (using copy.copy) is not allowed for objects "
				"of type Block. Use Block.copy instead.")
	
	
	def __deepcopy__(self):
		"""Deep copy routine. Copying is forbidden."""
		raise RuntimeError("Deep copying (using copy.deepcopy) is not allowed for "
				"objects of type Block. Use Block.copy instead.")
	
	
	def copy(self):
		"""Routine to copy this block."""
		cls = self.__class__
		new_self = cls.__new__(cls)
		new_self.__init__(_copy=True)
		
		port_map = dict()
		for p in self.ports:
			port_map[p] = new_self.add_port(name=p.local_name, kind=p.kind, 
							direction=p.direction)
		
		for m in self.models:
			print("This needs to have its ports changed")
			new_self.add_model(m.copy(port_map = port_map))
		
		return new_self
		
	
	@property
	def ports(self):
		return self.__ports
	
	# TODO: Full suite of port filtering functions, probably in arch.port -JWS
	
	
	@property
	def in_ports(self):
		return [p for p in self.__ports if p.direction == port.direction.inp]
	
	
	@property
	def out_ports(self):
		return [p for p in self.__ports if p.direction == port.direction.out]
	
	
	@property
	def models(self):
		return self.__models
	
	
	@property
	def model(self):
		try:
			return self.__models[0]
		except IndexError:
			return None



class AutoCompoundBlock(Block):
	
	reference_prefix = "A"
	
	def define(self, blocks, connectivity):
		
		self.blocks = blocks
		self.connectivity = connectivity
		
		for b in blocks:
			for p in b.ports:
				# Only expose ports that are not part of our connectivity graph
				if not connectivity.test(p):
					self.use_port(name=p.name.replace('.','_'), original=p)
		
		verbose = False # Set me to True to print compounding debug info
		def printv(*args):
			if verbose:
				print(*args)
		
		# Make compounds
		uncompounded_models = {m for b in blocks for m in b.models}
		model_types_of_blocks = dict()
		for b in blocks:
			for m in b.models:
				t = type(m)
				try:
					model_types_of_blocks[t].append(b)
				except KeyError:
					model_types_of_blocks[t] = [b]
		printv(model_types_of_blocks)
		
		model_blocks = {m:b for b in blocks for m in b.models}
		
		def models_and_their_friends(models):
			b_models = set()
			for m in models:
				try:
					b = model_blocks[m]
					b_models |= set(b.models)
				except KeyError:
					pass
			return set(models) | b_models
		
		def model_str(model):
			return str(type(model).__name__+' '+model.name)
		
		compound_num = 0
		iter_num = 0
		while len(uncompounded_models) > 1 and iter_num < 3:
			iter_num += 1
			printv("\nStarting iter number",iter_num)
			
			# First try to compound groups of same model type
			printv("Trying to compound models with MATCHING types")
			for t in {type(mu) for mu in uncompounded_models}:
				models_to_compound = {m for m in uncompounded_models if type(m) == t}
				printv("Compound",[model_str(m) for m in models_to_compound])
				try:
					compound = t.compound('auto compound same '+str(compound_num), block=self,
								 models=models_to_compound, connectivity=connectivity)
					compound_num += 1
					printv("Starting iter number",iter_num)
					for m in models_and_their_friends(models_to_compound):
						uncompounded_models.remove(m)
					uncompounded_models.add(compound)
					break
				except Exception as error:
					pass
			
			# Next try to compound different models together
			printv("Trying to compound models with DIFFERENT types")
			from itertools import combinations
			combos = {c for n in range(1,len(uncompounded_models)) for c in combinations(uncompounded_models,n)}
			printv("Finding compounds in combinations",[model_str(m) for models in combos for m in models])
			for models_to_compound in combos:
				try:
					printv("Trying to compound combo", [model_str(m) for m in models_to_compound])
					compound = t.compound('auto compound diff '+str(compound_num), block=self, models=models_to_compound, connectivity=connectivity)
					compound_num += 1
					printv("successfully made compound",model_str(compound))
					for m in models_and_their_friends(models_to_compound):
						uncompounded_models.remove(m)
					uncompounded_models.add(compound)
					break
				except NotImplementedError as error:
					pass
				except AttributeError as error:
					print("AttributeError caught when combining models", models_to_compound)
			printv("Finished round with {:} model(s)".format(len(uncompounded_models)))
		
		printv("done compounding")
		
		self.add_model(next(iter(uncompounded_models)))



#########
## OLD ##
#########

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
	
	
	@property
	def io_signature(self):
		"""Get a signature of the input and output ports and port types"""
		
		return {'in':{p.name:p.type for p in self.in_ports}, 'out':{p.name:p.type for p in self.out_ports} }
	
	
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