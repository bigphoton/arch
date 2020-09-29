"""
Input and output ports.
"""

import numpy as np
from .vis.generic import generic_port

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
	
	def __init__(self, name, type, is_input, owner, max_connections = 1, position=(0,0), angle=0):
		
		self.name = name
		self.type = type
		self.is_input = is_input
		self.connected_ports = set()
		self.__value = None
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
		return ("port" 
				+ (" '"+self.name+"'" if self.name != "" else "")
				+ (" input" if self.is_input else "")
				+ (" output" if self.is_output else "")
				+ (" connected to {:}".format(self.connected_ports) if self.connected_ports != set() else "") )
	
	def __repr__(self):
		if self.is_output:
			return "port('{:}',value={:.3f}\u2220{:.1f}\u00B0)".format(self.name, abs(self.value), np.degrees(np.angle(self.value)))
		else:
			return "port('{:}',type={:})".format(self.name, self.type)

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
	
	def __getitem__(self, key):
		val = {e for e in self if e.name == key}.pop()
		return val

