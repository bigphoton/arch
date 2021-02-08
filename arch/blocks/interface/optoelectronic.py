"""
Functions and objects describing single photon sources
"""

from arch.block import Block
import arch.port as port
from arch.models.model import SymbolicModel
from sympy import Abs



class Photodiode(Block):
	
	reference_prefix = "PD"
	
	def define(self, responsivity=1):
		
		inp = self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		resp = self.add_port(name='responsivity', kind=port.kind.real, direction=port.direction.inp, default=responsivity)
		i = self.add_port(name='i', kind=port.kind.real, direction=port.direction.out)
		
		self.add_model(SymbolicModel('photodiode '+self.name, block=self, 
							out_exprs={i:resp*Abs(inp)**2} ))

