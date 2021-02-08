"""
Functions and objects describing electro-optic components.
"""


from arch.block import Block
from arch.models.model import Linear
from sympy import Matrix, sqrt, exp, I
import arch.port as port

class Switch2x2(Block):
	"""
	extinction_ratio: ratio of desired signal to undesired signal from wrong port
	loss_dB: positive number of decibels of loss (0 dB -> 100% tx; 10 dB -> 10% tx)
	"""
	
	reference_prefix = "SW"
	
	def define(self, loss_dB = 3.0, extinction_ratio=1000.0):
		
		self.add_port(name='in0', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='in1', kind=port.kind.optical, direction=port.direction.inp)
		self.add_port(name='out0', kind=port.kind.optical, direction=port.direction.out)
		self.add_port(name='out1', kind=port.kind.optical, direction=port.direction.out)
		state = self.add_port(name='state', kind=port.kind.digital, 
			direction=port.direction.inp)
		
		# Lagrange polynomial
		s,er,tx = state,extinction_ratio,10**(-loss_dB/10)
		r = (s-0)/(1-0)*(1-1/er) + (s-1)/(0-1)*(1/er)
		
		M = sqrt(tx) * Matrix([
				[sqrt(r), I*sqrt(1 - r)],
				[I*sqrt(1 - r), sqrt(r)] ])
		
		self.add_model(Linear('simple switch '+self.name, block=self, unitary_matrix=M))