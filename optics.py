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
	"""
	
	def __init__(self, name, type, is_input, owner, max_connections = 1):
		
		self.name = name
		self.type = type
		self.is_input = is_input
		self.connected_ports = set()
		self.__value = None
		self.max_connections = max_connections
		self.owner = None
	
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
	origin: origin of drawing commands
	angle: angle of drawing commands
	
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
	"""
	
	debug_speed = 1
	
	def __init__(self, path = [], origin = (0,0), angle = 0, debug_mode = False):
		
		self.path = path
		
		self.origin = origin
		self.angle = angle
		
		self.debug_mode = debug_mode
		
		# Set up turtle
		# Stop turtle animations
		turtle.tracer(0,0)
		# Instantiate new turtle for our use
		self.turtle = turtle.Turtle()
	
	@property
	def path(self):
		return self.__path
	
	@path.setter
	def path(self,new_path):
		
		# Check path
		if len(new_path) % 2:
			raise RuntimeError("New path has an odd number of elements.")
		
		# Check each element of path
		for i in range(int(len(new_path)/2)):
			c = new_path[2*i]
			if type(c) != str:
				raise RuntimeError("Command at position {:} in path is invalid.".format(2*i))
		
		# Set
		self.__path = new_path
		
	
	def draw(self):
		"""
		Redraw graphic based on encoded path.
		"""
		
		self.turtle.reset()
		self.turtle.pu()
		self.turtle.goto(self.origin)
		self.turtle.lt(self.angle)
		self.turtle.pd()
		
		if self.debug_mode:
			turtle.tracer(1,0)
			self.turtle.speed(graphic.debug_speed)
		else:
			self.turtle.hideturtle()
			self.turtle.speed(0)
		
		self.turtle.dot(10)
		
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
				#  They use the most straightforward algorithm, but it's not efficient. -JWS 22/09/20
				angle = np.radians(self.angle)
				x0,y0 = self.turtle.pos() - self.origin
				r0 = np.sqrt(x0**2 + y0**2)
				angle0 = np.arctan2(y0,x0)
				x1,y1 = r0*np.cos(angle0 - angle), r0*np.sin(angle0 - angle)
				x2,y2 = a, y1
				r2 = np.sqrt(x2**2 + y2**2)
				angle2 = np.arctan2(y2,x2)
				x3,y3 = r2*np.cos(angle2 + angle), r2*np.sin(angle2 + angle)
				self.turtle.setpos( x3 + self.origin[0], y3 + self.origin[1] )
				
			elif c == "oy":
				angle = np.radians(self.angle)
				x0,y0 = self.turtle.pos() - self.origin
				r0 = np.sqrt(x0**2 + y0**2)
				angle0 = np.arctan2(y0,x0)
				x1,y1 = r0*np.cos(angle0 - angle), r0*np.sin(angle0 - angle)
				x2,y2 = x1, a
				r2 = np.sqrt(x2**2 + y2**2)
				angle2 = np.arctan2(y2,x2)
				x3,y3 = r2*np.cos(angle2 + angle), r2*np.sin(angle2 + angle)
				self.turtle.setpos( x3 + self.origin[0], y3 + self.origin[1] )
			elif c == "oa":
				self.turtle.seth(self.angle + a)
			elif c == "wi":
				self.turtle.width(a)
			elif c == "tx":
				self.turtle.write(a)
		
		turtle.update()
		
		if self.debug_mode:
			# Stop updates again
			turtle.tracer(0,0)
				
			
			

class base_optic:
	"""
	Base class for optical components.
	
	Subclass this class to implement different devices.
	
	Subclasses should implement:
	----------------------------
	
	 * reference_prefix: str
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
	
	def __init__(self):
		
		# Number of spatial modes
		self.n_spatial_modes = []
		
		
		# Handle reference designator generation
		try:
			existing_indices = base_optic.reference_designators[self.reference_prefix]
			
			self.reference_index = max(existing_indices) + 1
			
		except KeyError:
		
			self.reference_index = 0
			
			base_optic.reference_designators.update( {self.reference_prefix:set()} )
		
		base_optic.reference_designators[ self.reference_prefix ].add( self.reference_index )
		
		self.reference_designator = self.reference_prefix + str(self.reference_index)
		
		print ("Hello, my reference designator is " + self.reference_designator)
		
		# Set up ports
		self.ports = set()
		
		
	
	@classmethod
	def connect(cls, comp0, comp1, connection_dict):
		"""
		Generate and return a new base_optic which connects and merges
		the components.
		"""
		# TODO
		merged_comp = comp0
		return merged_comp
		
	
	def compute(self, input):
		"""
		Method to propagate input state to output of component.
		"""
		
		if input.model == "incoherent":
			pass
		
		if input.model == "coherent":
			pass
		
		if input.model == "fock_simple":
			pass
		
		if input.model == "fock_advanced":
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
		def __init__(self):
			self.reference_prefix = "B"
			self.supported_models = ["coherent"]
			self.n_spatial_modes = 2
			
			super(type(self), self).__init__()
		
		def model_matrix(self, reflectivity = 0.5):
			th = np.arctan(reflectivity)
			m = np.array([ [np.cos(th), 1j * np.sin(th)], 
						   [1j * np.sin(th), np.cos(th)] ])
			return m
	
# 	s = switch()
# 	w = switch()
# 	bsa = beamsplitter()
# 	i = switch()
# 	bs = beamsplitter()
	
	w = 100
	h = 200
	l = 20
	ni = 3
	no = 2
	
	p = []
	
	p.extend(['wi',3])
	
	# Draw box
	p.extend(['pu',None])
	p.extend(['ox',w/2])
	p.extend(['oy',h/2])
	p.extend(['pd',None])
	
	p.extend(['oy',-h/2])
	p.extend(['ox',-w/2])
	p.extend(['oy',+h/2])
	p.extend(['ox',+w/2])
	
	# Draw ports
	p.extend(['pu',None])
	p.extend(['wi',1])
	
	# Output ports (right side)
	p.extend(['ox',+w/2])
	p.extend(['oa',0])
	
	for n in range(no):
		p.extend(['oy',-h/2 + (n+1/2)*h/no])
		p.extend(['pd',None])
		p.extend(['fd',l])
		p.extend(['pu',None])
		p.extend(['bk',l])
	
	# Input ports (left side)
	p.extend(['ox',-w/2])
	p.extend(['oa',180])
	
	for n in range(ni):
		p.extend(['oy',-h/2 + (n+1/2)*h/ni])
		p.extend(['pd',None])
		p.extend(['fd',l])
		p.extend(['pu',None])
		p.extend(['bk',l])
	
	
	p.extend(['pu',None])
	p.extend(['ox',0])
	p.extend(['oy',0])
	p.extend(['tx',"Yo1"])
	
	
	
	g1 = graphic(path = p, origin = (150,100), angle = 15)
	g2 = graphic(path = p, origin = (0,0), angle = 0)
	
	g1.debug_mode = False
	g2.debug_mode = False
	
	g1.draw()
	g2.draw()
	
	
	
	turtle.exitonclick()