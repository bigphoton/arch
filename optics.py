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
		
		self.graphic = generic_port(name=name,is_input=is_input)
	
	def connect_to(self, other_port):
		
		# Do checks
		if len(self.connected_ports) >= self.max_connections:
			raise RuntimeError("Maximum number of connections reached for port '{:}' ({:} connections).".format(
						self.name, self.max_connections) )
		
		if self.type != other_port.type:
			raise RuntimeError("Port being connected of type '{:}' does not match this port (type '{:}').".format(other_port.type, self.type))
		
		if self.is_input and not other_port.is_output or self.is_output and other_port.is_input:
			raise RuntimeError("Port directions do not match.")
		
		# Connect
		self.connected_ports.add(other_port)
	
	
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
				+ (" connected to {:}".format(self.connected_port) if self.connected_port is not None else "") )


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


class optical_state:
	"""
	Class of all types of optical states across all models.
	"""
	
	def __init__(self,
					model = "incoherent", 
					n_spatial_modes = 1, 
					amplitudes = [1], 
					n_submodes = 1):
		
		# Type of model which this state applies to
		self.model = model
		
		# Check
		if self.model not in MODELS:
			raise AttributeError("Model type {:} is not one of the possible options: {:}".format(self.model, MODELS))
		
		# Number of spatial modes covered by state
		self.n_spatial_modes = n_spatial_modes
		
		# Check
		if type(self.n_spatial_modes) is not int:
			raise AttributeError("Number of spatial modes {:} must be an integer.".format(self.n_spatial_modes))
		if self.n_spatial_modes < 1:
			raise AttributeError("Number of spatial modes {:} must be greater than zero.".format(self.n_spatial_modes))
		
		# Number of other modes per spatial mode
		self.n_submodes = n_submodes
		
		# Check
		if type(self.n_submodes) is not int:
			raise AttributeError("Number of sub-modes per mode {:} must be an integer.".format(self.n_submodes))
		if self.n_submodes < 1:
			raise AttributeError("Number of sub-modes per mode {:} must be greater than zero.".format(self.n_submodes))
		
		# Amplitude of state in each mode
		self.amplitudes = amplitudes
		
		# Check
		if len(self.amplitudes) != self.n_spatial_modes * self.n_submodes:
			raise AttributeError("Number of amplitudes {:} must match total number of modes {:}.".format(len(self.amplitudes), self.n_spatial_modes * self.n_submodes))


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
	ox: go to x (x coordinate)
	oy: go to y (x coordinate)
	oa: go to angle (angle)
	wi: set width (width)
	tx: write text (text)
	arb: execute arbitrary argument-less method (text)
	"""
	
	debug_speed = 1
	
	def __init__(self, path = [], position = (0,0), angle = 0, debug = False):
		
		# Attributes which cause a redraw
		self.graphical_attributes = []
		
		self.position = position
		self.angle = angle
		
		self.debug = debug
		
		# Set up turtle
		# Stop turtle animations
		turtle.tracer(0,0)
		# Instantiate new turtle for our use
		self.turtle = turtle.Turtle()
		
		# Set path
		self.path = path
		
		# Start watching attributes
		self.graphical_attributes.extend(["position", "angle", "path", "debug"])
		
		# Draw self
		self.draw()
	
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
			turtle.tracer(1,0)
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
			elif c == "ox":
				# FIXME: Simplify `ox` and `oy` commands.
				#  They use the most straightforward algorithm, but it's not efficient. 
				#   -JWS 22/09/20
				angle = np.radians(self.angle)
				x0,y0 = self.turtle.pos() - self.position
				r0 = np.sqrt(x0**2 + y0**2)
				angle0 = np.arctan2(y0,x0)
				x1,y1 = r0*np.cos(angle0 - angle), r0*np.sin(angle0 - angle)
				x2,y2 = a, y1
				r2 = np.sqrt(x2**2 + y2**2)
				angle2 = np.arctan2(y2,x2)
				x3,y3 = r2*np.cos(angle2 + angle), r2*np.sin(angle2 + angle)
				self.turtle.setpos( x3 + self.position[0], y3 + self.position[1] )
				
			elif c == "oy":
				angle = np.radians(self.angle)
				x0,y0 = self.turtle.pos() - self.position
				r0 = np.sqrt(x0**2 + y0**2)
				angle0 = np.arctan2(y0,x0)
				x1,y1 = r0*np.cos(angle0 - angle), r0*np.sin(angle0 - angle)
				x2,y2 = x1, a
				r2 = np.sqrt(x2**2 + y2**2)
				angle2 = np.arctan2(y2,x2)
				x3,y3 = r2*np.cos(angle2 + angle), r2*np.sin(angle2 + angle)
				self.turtle.setpos( x3 + self.position[0], y3 + self.position[1] )
			elif c == "oa":
				self.turtle.seth(self.angle + a)
			elif c == "wi":
				self.turtle.width(a)
			elif c == "tx":
				self.turtle.write(a, move=False, align='center', font=('Arial',12,'bold'))
			elif c == "arb":
				getattr(self.turtle, a)()
			else:
				raise RuntimeError("Unrecognised path command '{:}'".format(c))
		
		turtle.update()
		
		if self.debug:
			# Stop updates again
			turtle.tracer(0,0)


class generic_box(graphic):
	"""
	Graphics subclass for drawing generic blocks for devices which don't draw their own.
	
	name: label to draw at centre
	n_in: number of input ports (on left side)
	n_out: number of output ports (on right side)
	"""
	
	box_width = 50
	box_height = 100
	port_length = 10
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
		l = generic_box.port_length
	
		p = []
	
		p.extend(['wi',generic_box.box_linewidth])
	
		# Draw box
		p.extend(['pu',None])
		p.extend(['ox',w/2])
		p.extend(['oy',h/2])
		p.extend(['pd',None])
	
		p.extend(['oy',-h/2])
		p.extend(['ox',-w/2])
		p.extend(['oy',+h/2])
		p.extend(['ox',+w/2])
		
		# Draw name text
		p.extend(['pu',None])
		p.extend(['ox',0])
		p.extend(['oy',-6])
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
		
		self.graphical_attributes.extend(["name", "is_input", "port_length"])
		
		self.update_path()
	
	
	def update_path(self):
		
		w = generic_box.box_width
		h = generic_box.box_height
		l = generic_box.port_length
	
		p = []
	
		p.extend(['wi',generic_box.box_linewidth])
		p.extend(['pd',None])
		
		if self.is_input:
			p.extend(['arb','stamp'])
			p.extend(['rt',180])
			p.extend(['fd',generic_port.port_length])
		else:
			p.extend(['rt',180])
			p.extend(['fd',generic_port.port_length])
			p.extend(['arb','stamp'])
		
		p.extend(['pu',None])
		p.extend(['rt',90])
		p.extend(['fd',3])
		p.extend(['tx',self.name])
		
		
		self.path = p
	
	


class base_optic:
	"""
	Base class for optical components.
	
	Subclass this class to implement different devices.
	
	Subclasses should implement:
	----------------------------
	 * reference_prefix: str, class property
	 * graphic: graphic
	 * ports: set
	 * model_parameters: dict
	 * model_matrix: list(list)
	 * supported_models: list
	 
	Data
	====
	
	reference_designators: dict(set). Keys are reference_prefix's, sets are indices
	
	Wishlist
	========
	 * Draws itself
	 * Calculates its own compact model
	 * Function to combine two components
	"""
	
	# Dictionary of reference designators for all optics
	reference_designators = dict()
	
	# To be overridden by subclasses:
	reference_prefix = "_"
	
	def __init__(self):
		
		# Handle reference designator generation
		try:
			existing_indices = base_optic.reference_designators[self.reference_prefix]
			self.reference_index = max(existing_indices) + 1
			
		except KeyError:
			self.reference_index = 0
			base_optic.reference_designators.update( {self.reference_prefix:set()} )
		
		base_optic.reference_designators[ self.reference_prefix ].add( self.reference_index )
		
		self.reference_designator = self.reference_prefix + str(self.reference_index)
		
		# Run subclass define routine
		self.define()
		
		# Handle graphic
		
		# Set default graphic if none present
		if not hasattr(self, "graphic"):
			self.graphic = generic_box(self.reference_designator)
		
		
	
	
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
	
	
	def define(self):
		"""
		Method to be overridden by subclasses.
		
		Must populate:
		 - self.reference_prefix
		 - self.ports
		
		Optionally populate:
		 - self.graphic
		"""
		pass
	
	
	def compute(self, input):
		"""
		Method to propagate input state to output of component.
		"""
	
	
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
		
		reference_prefix = "B"
		
		def define(self):
			
			# FIXME: this is a placeholder; needs to be implemented by the class -JWS 23/09/2020
			self.position = (0,0)
			
			p = set()
			w = generic_box.box_width
			h = generic_box.box_height
			x0,y0 = self.position
			
			n_in = 2
			for n in range(n_in):
				print ("IN"+str(n))
				print ((-w/2+x0,-h/2+(n+1/2)*h/n_in+y0))
				p.add(port("IN"+str(n), "optical", True,  self, 1, (-w/2+x0,-h/2+(n+1/2)*h/n_in+y0), 0))
			
			n_out = 2
			for n in range(n_out):
				print ("OUT"+str(n))
				print ((+w/2+x0,-h/2+(n+1/2)*h/n_out+y0))
				p.add(port("OUT"+str(n), "optical", False, self, 1, (+w/2+x0,-h/2+(n+1/2)*h/n_out+y0), 0))
			
			self.ports = p
			
			self.graphic = generic_box(self.reference_designator, position=self.position)
			
		
		def model_matrix(self, reflectivity = 0.5):
			th = np.arctan(reflectivity)
			m = np.array([ [np.cos(th), 1j * np.sin(th)], 
						   [1j * np.sin(th), np.cos(th)] ])
			return m
	
# 	s = switch()
# 	w = switch()
# 	bsa = beamsplitter()
# 	i = switch()
	bs = beamsplitter()
	
# 	g = generic_box("H100", 2, 2)
# 	p0 = generic_port("IN0", True, position=(0,0), debug = False)
# 	p1 = generic_port("IN1", True, position=(0,50), debug = False)
# 	p2 = generic_port("OUT0", False, position=(0,100), debug = False)
	
	turtle.exitonclick()