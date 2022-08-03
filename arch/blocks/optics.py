"""
Functions and objects describing optical components.
"""

from arch.block import Block
from arch.connectivity import Connectivity
from arch.models.model import Linear, LinearGroupDelay
from sympy import Matrix, sqrt, exp, I, eye
import arch.port as port
import numpy as np


class Waveguide(Block):
	
	reference_prefix = "WG"
	
	def define(self, eta=1.0):
		
		self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		# self.add_state_buffer(name='state', kind=port.kind.optical, direction=port.direction.buffer)
		self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		
		r = self.add_port(name='eta', kind=port.kind.real, direction=port.direction.inp, default = eta)

		M = Matrix([[r]])
		
		self.add_model(Linear('waveguide '+self.name, block=self, unitary_matrix=M))
        
class Beamsplitter(Block):
	
	reference_prefix = "BS"
	
	def define(self, R=0.5):
		
		self.add_port(name='in0', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='in1', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='out0', kind=port.kind.optical, direction=port.direction.out)
		self.add_port(name='out1', kind=port.kind.optical, direction=port.direction.out)
		r = self.add_port(name='R', kind=port.kind.real, direction=port.direction.inp, default = R)

		M = Matrix([
				[sqrt(r), I*sqrt(1 - r)],
				[I*sqrt(1 - r), sqrt(r)] ])
		
		self.add_model(Linear('simple R '+self.name, block=self, unitary_matrix=M))


class PhaseShifter(Block):
	
	reference_prefix = "P"
	
	def define(self, phi=0):
		
		self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		p = self.add_port(name='phi', kind=port.kind.real, direction=port.direction.inp, 
						default=phi)
		
		M = Matrix([[1.*exp(I*p)]])
		self.add_model(Linear('simple phase '+self.name, block=self, unitary_matrix=M))
		self.add_model(LinearGroupDelay('group delay phase '+self.name, block=self, unitary_matrix=M, delay=1))


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




class Interferometer(Block):
	"""
	Class to calculate the evolution of a quantum state through
	an interferometer described by some unitary.
	For the moment must be instantiated with a unitary matrix.
	"""
	reference_prefix = "IF"
	
	def define(self, unitary):
		
		self.add_port(name='out0', kind=port.kind.photonic, direction=port.direction.out)
		self.add_port(name='out1', kind=port.kind.photonic, direction=port.direction.out)
		self.add_port(name='in1', kind=port.kind.photonic, direction=port.direction.inp)
		input_state=self.add_port(name='in0', kind=port.kind.photonic, direction=port.direction.inp) #Create new "quantum" port type?
		#U=self.add_port(name='unitary', kind=port.kind.real, direction=port.direction.inp)

		self.add_model(Linear('simple R '+self.name, block=self, unitary_matrix=unitary))
