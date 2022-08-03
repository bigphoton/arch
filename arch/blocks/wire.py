"""
Functions and objects describing logic gates.
"""
from math import pi
from arch.block import Block
from arch.connectivity import Connectivity
from arch.models.model import TransmissionLineModel
from sympy import Matrix, sqrt, exp, I, eye
import arch.port as port
import numpy as np
from sympy import asin, acos
        
class Wire(Block):
	"""
	Transmission line for voltage
	"""
	
	reference_prefix = "TL"
	
	def define(self, Eta=1., Ph_v_const=2*pi):
		inpe= self.add_port(name='inp', kind=port.kind.voltage, direction=port.direction.inp)
		eta = self.add_port(name='eta', kind=port.kind.real,    direction=port.direction.inp, default = Eta)
		ph_v_const = self.add_port(name='ph_v', kind=port.kind.real,    direction=port.direction.inp, default = Ph_v_const)
		
		# outmon = self.add_port(name='out_mon', kind=port.kind.voltage, direction=port.direction.out)
		out = self.add_port(name='out', kind=port.kind.voltage, direction=port.direction.out)
        

        
		# self.add_model(TransmissionLineModel('wire '+self.name, block=self, out_exprs={out: ph_v_const*inpe, outmon :ph_v_const*inpe}))
		self.add_model(TransmissionLineModel('wire '+self.name, block=self, out_exprs={out: ph_v_const*inpe}))