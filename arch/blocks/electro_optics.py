"""
Functions and objects describing electro-optic components.
"""


from arch.block import Block
from arch.models.model import Linear, SymbolicModel
from sympy import Matrix, sqrt, exp, I, pi
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



class ThermoOpticPhaseShifterBasicRT(Block):
	"""
	Due to Dario, based on https://doi.org/10.1364/OE.27.010456
	"""
	
	reference_prefix = "TOPM"
	
	def define(self, device_length=None, centre_wavelength=2.0E-6, ref_index_temp_func=lambda T:1.0*T, R=None):
		"""
		thermooptic_coeff: constant thermo-optic coefficient
		i0: input port current
		v0: input port voltage
		"""
		
		
		A,B,C,D = 1,-R,0,1
		
		M = Matrix([[A,B],[C,D]])
		
		
		inp = self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		out = self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		
		i0 = self.add_port(name='i0', kind=port.kind.voltage, direction=port.direction.inp)
		v0 = self.add_port(name='v0', kind=port.kind.current, direction=port.direction.inp)
		i1 = self.add_port(name='i1', kind=port.kind.voltage, direction=port.direction.out)
		v1 = self.add_port(name='v1', kind=port.kind.current, direction=port.direction.out)
		
		T = self.add_port(name='T', kind=port.kind.temperature, direction=port.direction.inp)
		
		
		
		oes = {
			out: exp(I* (2*pi*device_length/centre_wavelength)*ref_index_temp_func(T) )*inp,
			v1: +A*v0 + B*i0,
			i1: -C*v0 - D*i0}
		
		
		
		self.add_model(SymbolicModel('simple phase '+self.name, block=self, out_exprs=oes))
		