"""
Input and output ports.
"""



import numpy as np
from .vis.generic import generic_port

from enum import Enum     # Req. Python >= 3.4


class kind(Enum):
	optical = 0
	photonic = 0
	digital = 1
	temperature = 2
	voltage = 3
	current = 4
	real = 10
	complex = 11
	integer = 12

# Default symbol values for each kind of port. Put None for no default.
# TODO: Would be nice to integrate this as a property of each kind as kind.default
KIND_DEFAULTS = {
		kind.optical:0.0,
		kind.digital:0, 
		kind.temperature:300.0, 
		kind.voltage:0.0,
		kind.current:0.0,
		kind.real:0.0,
		kind.complex:0.0,
		kind.integer:0}

KIND_ASSUMPTIONS = {
		kind.optical:{'complex':True},
		kind.digital:{'integer':True, 'nonnegative':True}, 
		kind.temperature:{'positive':True}, 
		kind.voltage:{'real':True},
		kind.current:{'real':True},
		kind.real:{'real':True},
		kind.complex:{'complex':True},
		kind.integer:{'integer':True}
		}

class direction(Enum):
    inp = 0
    out = 1



import sympy.core.symbol

class var(sympy.core.symbol.Symbol):
	"""
	A version of sympy.Symbol with attached attributes.
	
	block: block to which this port was initially attached, Block
	kind: kind of port, port.kind
	direction: sense of port, port.direction
	default: default value if not set, kind-specific
	
	data: dictionary of attached data, for use by models
	"""
	def __new__(self, local_name, block=None, 
					kind=None, direction=None, default=None):
		"""
		We need to intercept __new__ rather than __init__ because Symbol uses it 
			instead of __init__.
		"""
		if block is None:
			my_name = local_name
		else:
			my_name = var.new_name(block.name, local_name)
		obj = sympy.core.symbol.Symbol.__new__(self, my_name, **KIND_ASSUMPTIONS[kind])
		obj.local_name = local_name
		obj.block = block
		obj.kind = kind
		obj.direction = direction
		obj.default = default
		obj.data = dict()
		return obj
	
	@classmethod
	def new_name(cls, block_name, local_name):
		"""
		The name is set based on the *last* block to which we are associated.
		"""
		return block_name + '.' + local_name

class port:
	"""
	Class encapsulating an input or output port attached to a component.
	
	name: name of port
	type: name of port type; connected ports must have matching type
	is_input: port is an input port
	is_output: port is an output port (not is_output)
	connected_ports: list of other port instances connected to this one
	max_connections: maximum number of ports connected to this one
	value: the value emitted by the output port and received by the input port in a connection
	value_info: dict containing information about value (e.g. model type, dimensionality, etc.)
	"""
	
	def __init__(self, name, type, is_input, owner, value=None, max_connections = 1, position=(0,0), angle=0):
		
		self.name = name
		self.type = type
		self.is_input = is_input
		self.connected_ports = set()
		self.__value = value
		self.value_info = dict()
		self.max_connections = max_connections
		self.owner = owner
		
		self.graphic = generic_port(name=name, is_input=is_input, position=position, angle=angle)
	
	def connect(self, other_port):
		
		# Only run if we're not already connected
		if other_port in self.connected_ports:
			return
		
		# Do checks
		if len(self.connected_ports) >= self.max_connections:
			raise RuntimeError("Maximum number of connections reached for port '{:}' ({:} connections).".format(
						self.name, self.max_connections) )
		
		if self.type != other_port.type:
			raise RuntimeError("Port being connected of type '{:}' does not match this port (type '{:}').".format(other_port.type, self.type))
		
		if self.is_input == other_port.is_input:
			raise RuntimeError("Port directions do not match.")
		
		# Connect
		# TODO: Disconnect connected ports as required to stay within self.max_connections -JWS 24/09/2020
		self.connected_ports.add(other_port)
		
		# Tell other port we're married
		other_port.connect(self)
		
		# Connect graphically
		if self.is_input:
			self.graphic.connected_port = other_port
			self.graphic.update_path()
	
	
	@property
	def is_output(self):
		return not self.is_input
		
	@is_output.setter
	def is_output(self, new):
		self.is_input = not new
	
	
	@property
	def value(self):
		return self.__value
	
	@value.setter
	def value(self, new_value):
		self.__value = new_value
		
		# Propagate value from this output port to connected input ports
		if self.is_output:
			for p in self.connected_ports:
				if p.is_input:
					p.value = new_value
		
		
	def __str__(self):
		s = "port"
		s += (" '"+self.name+"'" if self.name != "" else "")
		s += (" input" if self.is_input else "")
		s += (" output" if self.is_output else "")
		s += (" connected to {:}".format(self.connected_ports) if self.connected_ports != set() else "")
		
		try:
			s += (" value is {:}".format(port.phasor(self.value) ) )
		except:
			s += (" value is {:}".format(self.value) )
		
		return s
	
	
	def __repr__(self):
		if self.is_output:
			return "port('{:}', type={:}, value={:})".format(self.name, self.type, str(self.value))
		else:
			return "port('{:}', type={:})".format(self.name, self.type)
	
	
	@classmethod
	def phasor(cls, value):
		"""Convenience complex formatting"""
		
		return "{:.3f}\u2220{:.1f}\u00B0".format(abs(value), np.degrees(np.angle(value)))
		

class input_port(port):
	"""
	Convenience subclass of port with is_input = True.
	"""
	
	def __init__(self, *args, **kwargs):
		port.__init__(self,args,kwargs)
		self.is_input = True
		
class output_port(port):
	"""
	Convenience subclass of port with is_output = True.
	"""
	
	def __init__(self, *args, **kwargs):
		port.__init__(self,args,kwargs)
		self.is_input = True

class port_set(set):
	"""
	Subscriptable container class for ports.
	
	Build like a set, access elements like a set or like a dict.
	"""
	
	# TODO: Support set addition (e.g. port_set({a_set}) + {another_set} )
	
	def __getitem__(self, key):
		val = {e for e in self if e.name == key}.pop()
		return val
