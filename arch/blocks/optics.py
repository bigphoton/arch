"""
Functions and objects describing optical components.
"""


import numpy as np
from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.optical.classical.linear import linear_basic
from arch.models.optical.quantum.permanents import permanent
from arch.models.optical.quantum.permanents_quantum import permanent_quantum
from arch.connectivity import Connectivity

from arch.block import Block
from arch.models.model import Linear, LinearGroupDelay
from sympy import Matrix, sqrt, exp, I
import arch.port as port

class Beamsplitter(Block):
	
	reference_prefix = "BS"
	
	def define(self, R=0.5):
		
		self.add_port(name='in0', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='in1', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='out0', kind=port.kind.optical, direction=port.direction.out)
		self.add_port(name='out1', kind=port.kind.optical, direction=port.direction.out)
		r = self.add_port(name='R', kind=port.kind.real, direction=port.direction.inp, 
							default=R)
		
		M = Matrix([
				[sqrt(r), I*sqrt(1 - r)],
				[I*sqrt(1 - r), sqrt(r)] ])
		
		self.add_model(Linear('simple R '+self.name, block=self, unitary_matrix=M))


class PhaseShifter(Block):
	
	reference_prefix = "P"
	
	def define(self, phi=None):
		
		self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		p = self.add_port(name='phi', kind=port.kind.real, direction=port.direction.inp, 
						default=phi)
		
		M = Matrix([[exp(I*p)]])
		
		self.add_model(Linear('simple phase '+self.name, block=self, unitary_matrix=M))
		self.add_model(LinearGroupDelay('group delay phase '+self.name, block=self, 
								unitary_matrix=M, delay=1))




class MachZehnder(Block):
	
	reference_prefix = "MZ"
	
	def define(self, R0=1/2, R1=1/2):
		
		bs0 = Beamsplitter(R=R0)
		bs1 = Beamsplitter(R=R1)
		ps = PhaseShifter()
		
		con = Connectivity([
					(bs0.out0, ps.inp),
					(ps.out, bs1.in0),
					(bs0.out1, bs1.in1) ])
		
		self.use_port(name='in0', original=bs0.in0)
		self.use_port(name='in1', original=bs0.in1)
		self.use_port(name='out0', original=bs1.out0)
		self.use_port(name='out1', original=bs1.out1)
		self.use_port(name='phi', original=ps.phi)
		self.use_port(name='R0', original=bs0.R)
		self.use_port(name='R1', original=bs1.R)
		
		self.add_model(
			Linear.compound(
				'compound '+self.name, 
				models=[bs0.model, ps.model, bs1.model], 
				connectivity=con))



class RingResonator(Block):
	
	reference_prefix = "RR"
	
	def define(self, R=None, phi=None):
		
		bs = Beamsplitter(R=R)
		ps = PhaseShifter()
		
		con = Connectivity([
					(bs.out0, ps.inp),
					(ps.out, bs.in0) ])
		
		self.use_port(name='in', original=bs.in1)
		self.use_port(name='out', original=bs.out1)
		self.use_port(name='phi', original=ps.phi)
		self.use_port(name='R', original=bs.R)
		
		raise NotImplementedError("TODO")



from arch.models.model import SourceModel

class LaserCW(Block):
	
	reference_prefix = "CW"
	
	def define(self):
		
		P = self.add_port(name='P', kind=port.kind.real, direction=port.direction.inp, default=1.0)
		phase = self.add_port(name='phase', kind=port.kind.real, direction=port.direction.inp, default=0.0)
		out = self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		
		self.add_model(SourceModel('laser '+self.name, block=self, 
							out_exprs={out:sqrt(P)*exp(I*phase)} ))



#########
## OLD ##
#########

class beamsplitter(base_block):

	"""Requires the user to pick a model of computation. Options are 'linear', 'monte_carlo', 'full_quantum'
	Defaults to linear"""
	
	reference_prefix = "BS"
	
	def define(self, model_choice, reflectivity=0.5):
		
		self.model_choice=model_choice

		# Setup ports
		w = generic_box.box_width
		h = generic_box.box_height
		l = generic_port.port_length
		x0,y0 = self.position
		
		
		# Add two input ports
		n_in = 2
		for n in range(n_in):
			name = "IN"+str(n)
			self.ports.add(port(name, "optical", True, self, None, 1, (-w/2+x0-l,-h/2+(n+1/2)*h/n_in+y0), 0))
			self.in_port_order.append(name)
		
		# ...and two outputs
		n_out = 2
		for n in range(n_out):
			name = "OUT"+str(n)
			self.ports.add(port(name, "optical", False, self, None, 1, (+w/2+x0+l,-h/2+(n+1/2)*h/n_out+y0), 180))
			self.out_port_order.append(name)
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		
		# Setup model matrix
		def model_matrix_func(reflectivity):
			r = np.sqrt(reflectivity)
			t = np.sqrt(1-r**2) * 1j
			m = np.array([ [r, t], 
						   [t, r] ])
			
			return m
		
		# Model parameter(s)
		self.model_params.update({'reflectivity':reflectivity})
		
		# Set model
		if model_choice=='monte_carlo':
			self.model = permanent(model_matrix_func, self.model_params,model_choice)
		elif model_choice=='linear':
			self.model= linear_basic(model_matrix_func, self.model_params)
		elif model_choice=='full_quantum':
			self.model= permanent_quantum(model_matrix_func, self.model_params, model_choice)
		else:
			raise Exception('This is not a valid model choice for a beamsplitter')

		
class phase_shift(base_block):
	
	reference_prefix = "P"
	
	def define(self, phase=0):
		
		# Internal variable(s)
		self.phase = phase
		
		# Setup ports
		w = generic_box.box_width
		l = generic_port.port_length
		x0,y0 = self.position
		
		# Add ports
		self.ports.add(port("IN", "optical", True,  self, None, 1, (-w/2+x0-l,y0), 0))
		self.ports.add(port("OUT", "optical", False, self, None, 1, (+w/2+x0+l,y0), 180))
		
		self.in_port_order = ["IN"]
		self.out_port_order = ["OUT"]
		
		# Setup graphic
		self.graphic = generic_box(self.reference_designator, position=self.position)
		
		
		# Setup model matrix
		def model_matrix_func(phase):
			return np.array([[np.exp(1j*phase)]])
		
		# Model parameter(s)
		self.model_params.update({'phase':phase})
		
		# Set model
		self.model = linear_basic(model_matrix_func, self.model_params)