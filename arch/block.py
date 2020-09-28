"""
Functions and objects describing optical physics and optical components.

Wishlist
--------
 * Basic classical modeler (Monte Carlo)
 * Basic classical modeler (incoherent)
 * Classical modeler (coherent)
 * Intermediate quantum modeler (Fock on single modes)
 * Advanced quantum modeler (Fock on multiple spectro-temporal modes, Gaussian?)
 
Structure
=========

base_optic
 * Contains compact model
 * Takes input parameters
 * Has input and output ports
 * Can compute output from input state

complex_optic
 * Inherits from base_optic
 * Contains any number of base_optic and complex_optic's
 * Contains a map of internal io ports
 * Can compactly model contents
 
"""

import numpy as np
import turtle
from turtle import Vec2D as v2


MODELS = ["incoherent", 
		  "coherent", 
		  "monte_carlo",
		  "cv",
		  "fock_simple", 
		  "fock_advanced"]


class port:
	"""
	Class encapsulating an input or output port attached to a component.
	
	is_input: port is an input port
	is_output: port is an output port (not is_output)
	type: name of port type; connected ports must have matching type
	name: name of port
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


class graphic:
	"""
	Class encapsulating graphic and glyph drawing.
	
	path: list of path commands (odd elements) and parameters (even elements)
	position: origin of drawing commands
	angle: angle of drawing commands
	
	graphical_attributes: attributes watched for updates to trigger redrawing
	
	Subclasses should implement:
	 * update_path()
	 
	Subclasses can optionally implement:
	 * __init__() to modify input parameters
	 * graphical_attributes, extending
	
	Path commands
	-------------
	
	pu: pen up (None)
	pd: pen down (None)
	rt: right turn in degrees (angle)
	lt: left turn in degrees (angle)
	fd: forward move (distance)
	bk: backward move (distance)
	go: go to position (x,y tuple)
	oa: go to angle (angle)
	cl: set line colour (text or 3-tuple)
	wi: set width (width)
	tx: write text (text)
	arb: execute arbitrary argument-less method (text)
	"""
	
	debug_speed = 1
	is_synchronous = True
	
	def __init__(self, path = [], position = v2(0,0), angle = 0, debug = False):
		
		# Attributes which cause a redraw
		self.graphical_attributes = []
		
		self.position = v2(*position)
		self.angle = angle
		
		self.debug = debug
		
		# Set up turtle
		# Stop turtle animations
		graphic.run_animations(False)
		# Instantiate new turtle for our use
		self.turtle = turtle.Turtle()
		turtle.delay(100)
		
		# Set path
		self.path = path
		
		# Start watching attributes
		self.graphical_attributes.extend(["position", "angle", "path", "debug"])
		
		# Draw self
		self.draw()
	
	
	@classmethod
	def run_animations(cls, is_running):
		"""
		Method to turn graphic updates on and off globally.
		
		is_running: bool
		"""
		if is_running:
			turtle.tracer(1,0)
		else:
			turtle.tracer(0,0)
	
	
	@classmethod
	def force_update(cls):
		"""
		Method to force a global graphics update (without animating).
		"""
		turtle.update()
	
	
	@classmethod
	def update(cls):
		"""
		Method to perform a global graphics update (without animating).
		If graphic.is_synchronous is false, this will not cause an update.
		"""
		if cls.is_synchronous:
			cls.force_update()
	
	
	def __setattr__(self, name, value):
		# Store the original version of this attribute
		try:
			old_value = self.__getattribute__(name)
		except:
			old_value = None
		
		# Set the attribute in super
		object.__setattr__(self, name, value)
		
		# If we are setting 'path', do checks, and do not re-update the path
		if name == "path":
			# Check path
			if len(value) % 2:
				raise RuntimeError("New path has an odd number of elements.")
		
			# Check each element of path
			for i in range(int(len(value)/2)):
				c = value[2*i]
				if type(c) != str:
					raise RuntimeError("Command at position {:} in path is invalid.".format(2*i))
			
			# Draw
			self.draw()

		# If the attribute is on our redraw watch list and is changed, update path
		#  Updating path forces a redraw via the code above
		elif name in self.graphical_attributes and value != old_value:
			self.update_path()
	
	
	def update_path(self):
		"""
		Subclasses should override this method to update their paths when things change.
		This method should only update the path, not call graphics.draw().
		"""
		pass
		
	
	def draw(self):
		"""
		Redraw graphic based on encoded path.
		"""
		self.turtle.reset()
		self.turtle.pu()
		self.turtle.goto(self.position)
		self.turtle.lt(self.angle)
		self.turtle.pd()
		
		if self.debug:
			graphic.run_animations(True)
			self.turtle.speed(graphic.debug_speed)
			self.turtle.dot(10,'blue')
		else:
			self.turtle.hideturtle()
			self.turtle.speed(0)
		
		for i in range(int(len(self.path)/2)):
			
			c = self.path[2*i]
			a = self.path[2*i + 1]
			
			if type(c) != str:
				raise RuntimeError("Command at position {:} in path is invalid.".format(i*2))
			
			if   c == "pu":
				self.turtle.pu()
			elif c == "pd":
				self.turtle.pd()
			elif c == "rt":
				self.turtle.rt(a)
			elif c == "lt":
				self.turtle.lt(a)
			elif c == "fd":
				self.turtle.fd(a)
			elif c == "bk":
				self.turtle.bk(a)
			elif c == "go":
				self.turtle.setpos(self.transform_to_global(a))
			elif c == "oa":
				self.turtle.seth(self.angle + a)
			elif c == "wi":
				self.turtle.width(a)
			elif c == "cl":
				self.turtle.color(a)
			elif c == "tx":
				self.turtle.write(a, move=False, align='center', font=('Arial',12,'bold'))
			elif c == "arb":
				getattr(self.turtle, a)()
			else:
				raise RuntimeError("Unrecognised path command '{:}'".format(c))
		
		graphic.update()
		
		if self.debug:
			# Stop updates again
			graphic.run_animations(False)
	
	
	def transform_to_global(self, local_position):
		p_l = v2(*local_position)
		
		# Origin
		p_o = v2(*self.position)
		angle_o = -self.angle #deg
		
		# Projectors
		p_xo = v2(1,0).rotate(angle_o)
		p_yo = v2(0,1).rotate(angle_o)
		
		# Global
		p_g = v2(p_l*p_xo, p_l*p_yo) + p_o
		
		return p_g
	
	def transform_to_local(self, global_position):
		p_g = v2(*global_position)
		
		# Origin
		p_o = v2(*self.position)
		angle_o = self.angle #deg
		
		# Projectors
		p_xo = v2(1,0).rotate(angle_o)
		p_yo = v2(0,1).rotate(angle_o)
		
		# Local
		p_l = v2((p_g - p_o)*p_xo, (p_g - p_o)*p_yo)
		
		return p_l


class generic_box(graphic):
	"""
	Graphics subclass for drawing generic blocks for devices which don't draw their own.
	
	name: label to draw at centre
	n_in: number of input ports (on left side)
	n_out: number of output ports (on right side)
	"""
	
	box_width = 60
	box_height = 120
	box_linewidth = 2
	
	
	def __init__(self, name="", **kwargs):
	
		graphic.__init__(self, **kwargs)
		
		self.name = name
		
		self.graphical_attributes.extend(["name"])
		
		self.update_path()
	
	
	def update_path(self):
		"""
		Update the graphics path with latest parameters (name, n_in, n_out...)
		"""
		
		w = generic_box.box_width
		h = generic_box.box_height
	
		p = []
	
		p.extend(['wi',generic_box.box_linewidth])
	
		# Draw box
		p.extend(['pu',None])
		p.extend(['go',(+w/2,+h/2)])
		p.extend(['pd',None])
	
		p.extend(['go',(+w/2,-h/2)])
		p.extend(['go',(-w/2,-h/2)])
		p.extend(['go',(-w/2,+h/2)])
		p.extend(['go',(+w/2,+h/2)])
		
		# Draw name text
		p.extend(['pu',None])
		p.extend(['go',(0,-6)])
		p.extend(['tx',self.name])
		
		self.path = p


class generic_port(graphic):
	"""
	Graphic for drawing input/output ports.
	
	name: str, label string
	in_port: bool, port direction
	"""
	
	port_length = 20
	
	def __init__(self, name="", is_input=True, **kwargs):
	
		graphic.__init__(self, **kwargs)
		
		self.name = name
		self.is_input = is_input
		
		self.connected_port = None
		
		self.graphical_attributes.extend(["name", "is_input", "port_length"])
		
		self.update_path()
	
	
	def update_path(self):
	
		p = []
	
		p.extend(['pu',None])
		p.extend(['wi',generic_box.box_linewidth])
		
		p.extend(['lt',self.angle+90])
		p.extend(['fd',5])
		p.extend(['tx',self.name])
		p.extend(['bk',5])
		p.extend(['oa',0])
		p.extend(['pd',None])
		
		
		if self.is_input:
			p.extend(['go',(generic_port.port_length,0)])
			p.extend(['arb','stamp'])
			p.extend(['pu',None])

		else:
			p.extend(['rt',180])
			p.extend(['arb','stamp'])
			p.extend(['go',(generic_port.port_length,0)])
			p.extend(['pu',None])
			
		if self.is_input:
			try:
				u = self.transform_to_local(self.position)
				v = self.transform_to_local(self.connected_port.graphic.position)
				p.extend(['wi',1])
				p.extend(['cl','blue'])
				p.extend(['pu',None])
				p.extend(['go',u])
				p.extend(['pd',None])
				p.extend(['go',v])
			except:
				pass
		
		self.path = p
	
	


class base_optic:
	"""
	Base class for optical components.
	
	Subclass this class to implement different devices.
	
	Subclasses should implement:
	----------------------------
	 * reference_prefix: str, class property
	 * graphic: graphic
	 * ports: port_set
	 * model_parameters: dict
	 * model_matrix: list(list)
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
	
	# Dictionary of reference designators for all optics
	reference_designators = dict()
	
	# To be overridden by subclasses:
	reference_prefix = "_"
	
	def __init__(self, **kwargs):
		
		# Handle reference designator generation
		try:
			existing_indices = base_optic.reference_designators[self.reference_prefix]
			self.reference_index = max(existing_indices) + 1
			
		except KeyError:
			self.reference_index = 0
			base_optic.reference_designators.update( {self.reference_prefix:set()} )
		
		base_optic.reference_designators[ self.reference_prefix ].add( self.reference_index )
		
		self.reference_designator = self.reference_prefix + str(self.reference_index)
		
		# Placeholder ports set
		self.__ports = port_set()
		
		# TODO: set position intelligently, or hide graphic before its position is set
		self.__position = v2(0,0)
		
		# Run subclass define routine
		#  Pass kwargs to define to initialise variables as required
		self.define(**kwargs)
		
		# Handle graphics
		
		# Set default graphic if none present
		if not hasattr(self, "graphic"):
			self.graphic = generic_box(self.reference_designator, position=self.position, angle=0)
		
	
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
		return {p for p in self.ports if p.is_input}
	
	
	@property
	def out_ports(self):
		"""
		Convenience access to a filtered list of output ports only
		"""
		return {p for p in self.ports if p.is_output}
	
	
	def define(self, **kwargs):
		"""
		Method to be overridden by subclasses.
		
		Must populate:
		 - self.reference_prefix
		 - self.ports
		
		Optionally populate:
		 - self.graphic
		"""
		pass
	
	
	def compute(self):
		"""
		Method to propagate input state to output of component. Overridden by subclasses.
		
		Must take values from input ports, place values on output ports.
		"""
		pass
	
	
class complex_optic(base_optic):
	"""
	Optical component combining several other optical components.
	"""



if __name__ == "__main__":
	
	print ("Hello world")
	
	class switch(base_optic):
		def __init__(self):
			self.reference_prefix = "X"
			self.supported_models = ["incoherent"]
			self.n_spatial_modes = 2
			
			super(type(self), self).__init__()
		
		def model_matrix(self, state = 0):
			if state == 0:
				m = np.array([[1.0, 0.0], [0.0, 1.0]])
			elif state == 1:
				m = np.array([[0.0, 1.0], [1.0, 0.0]])
			
			return m
	
	
	class beamsplitter(base_optic):
		
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
			
			
	class phase_shift(base_optic):
		
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
	
	
	from time import sleep
	
	bs0 = beamsplitter(reflectivity=0.5)
	bs0.position = (-200,+20)
	
	ps = phase_shift(phase=0)
	ps.position = (0,-40)
	
	bs1 = beamsplitter()
	bs1.position = (+200,+20)
	
	bs0.ports['OUT0'].connect(ps.ports['IN'])
	ps.ports['OUT'].connect(bs1.ports['IN0'])
	bs0.ports['OUT1'].connect(bs1.ports['IN1'])
	
	
	bs0.ports['IN0'].value = 1.0
	bs0.ports['IN1'].value = 0.0
	
	sleep(0.5)
	
	
	for phase in np.linspace(0,np.pi,10):
		ps.phase = phase
		bs0.compute()
		ps.compute()
		bs1.compute()
		print("phase={:.3f}, output={:}".format(phase,bs1.out_ports))
