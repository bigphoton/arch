"""
Functions and objects describing logic gates.
"""

from arch.block import Block
from arch.connectivity import Connectivity
from arch.models.model import DCQontrolModel
from sympy import Matrix, sqrt, exp, I, eye
import arch.port as port
import numpy as np
		
class Qontrol(Block):
	"""
	DC voltage provided by Qontrol (TM) drivers.
	"""
	
	reference_prefix = "QT"
	
	def define(self, nch = 8, Vs=[0]):
		
		while len(Vs) < nch:
			Vs.append(0)
             
		vs = [self.add_port(name='qv_'+str(i), kind=port.kind.real, direction=port.direction.inp, default = Vs[i]) for i in range(nch)]
		outp = [self.add_port(name='out_'+str(i), kind=port.kind.voltage, direction=port.direction.out) for i in range(nch)]

		out_exprs_ = {outp[i] : vs[i] for i in range(nch)}
		self.add_model(DCQontrolModel('qontrol '+self.name, block=self, out_exprs = out_exprs_))