"""
Functions and objects describing single photon sources
"""

from arch.block import Block
import arch.port as port
from arch.models.model import SourceModel
from sympy import sqrt, exp, I



class LaserCW(Block):
	
	reference_prefix = "CW"
	
	def define(self, Power = 1.0, Wavelength= 2000):
		
		Wavelength = self.add_port(name='lambda', kind=port.kind.real, direction=port.direction.inp, default = Wavelength)
		
		P = self.add_port(name='P', kind=port.kind.real, direction=port.direction.inp, default = Power)
		phase = self.add_port(name='phase', kind=port.kind.real, direction=port.direction.inp, default=0.0)
        
		out = self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		
		self.add_model(SourceModel('laser '+self.name, block=self, out_exprs={out:sqrt(P)*exp(I*phase)} ))



class BasicPhotonPairSource(Block):

	"""
	A simple button click pair photon source.
	Digital button click input, two single photon output channels

	Desired class attributes: 
	"""

	reference_prefix="PPS"

	def define(self, model_choice):
		raise NotImplementedError()



class BasicSinglePhotonSource(Block):

	"""
	A simple button click pair photon source.
	Digital button click input, two single photon output channels

	Desired class attributes: 
	"""

	reference_prefix="SPS"

	def define(self, model_choice, max_occ=2):

		if model_choice=='Fock':

			P = self.add_port(name='P', kind=port.kind.real, direction=port.direction.inp, default=1.0)
			max_occupation = self.add_port(name='max_occ', kind=port.kind.real, direction=port.direction.inp, default=max_occ)
			out = self.add_port(name='out', kind=port.kind.photonic, direction=port.direction.out)

			self.add_model(SourceModel('Fock state source'+self.name, block=self, 
							out_exprs={out:sqrt(P)} ))
		else:

			raise NotImplementedError()