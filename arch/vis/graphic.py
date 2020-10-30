"""
Core graphics functionality.
"""

import numpy as np
import turtle
from turtle import Vec2D as v2


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
		
		# Set path
		self.path = path
		
		# Start watching attributes
		self.graphical_attributes.extend(["position", "angle", "path", "debug"])
	
	
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
	
	
	@classmethod
	def wait_until_close(cls):
		"""
		Method to perform a global graphics update (without animating).
		If graphic.is_synchronous is false, this will not cause an update.
		"""
		turtle.mainloop()
	
	
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
		
		# Set up turtle if not already done
		if not hasattr(self, 'turtle'):
			# Stop turtle animations
			graphic.run_animations(False)
			# Instantiate new turtle for our use
			self.turtle = turtle.Turtle()
			turtle.delay(100)
		
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
