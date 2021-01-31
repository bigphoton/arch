"""
Functions and objects describing single photon sources
"""

from arch.block import Block
import arch.port as port
from arch.models.model import SymbolicModel
from sympy import Piecewise



class Comparator(Block):
	"""
	Digitises a real input.
	
	threshold: 0 if input < threshold else 1
	hysteresis: additional inp change above or below threshold to cause out change
	"""
	
	reference_prefix = "COMP"
	
	def define(self, threshold=0.5, hysteresis=0.1):
		
		inp = self.add_port(name='inp', kind=port.kind.real, direction=port.direction.inp)
		t = self.add_port(name='threshold', kind=port.kind.real, direction=port.direction.inp, default=threshold)
		h = self.add_port(name='hyst', kind=port.kind.real, direction=port.direction.inp, default=hysteresis)
		out = self.add_port(name='out', kind=port.kind.digital, direction=port.direction.out)
		
		self.add_model(
			SymbolicModel('comparator '+self.name, block=self, 
				out_exprs={out:Piecewise(
								(0, ((inp<t-h) & out) | ((inp<t+h) & ~out)),
								(1, ((inp>t-h) & out) | ((inp>t+h) & ~out))) } ))

