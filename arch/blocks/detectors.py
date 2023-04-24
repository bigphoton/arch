"""
Functions and objects describing single photon detectors
"""
import numpy as np
from arch.block import Block
import arch.port as port
from arch.models.model import SourceModel
from arch.models.model import DetectorModel
from sympy import sqrt, exp, I, asin, acos, cos


class BasicSPD(Block):

	"""
	Basic detector block with simple efficiency function. 
	Currently monte-carlo style detection is implemented, clicks forwarded to timetag data output


	Desired class attributes: efficiency, dead time, dark counts, jitter, bandwidth
	spectral range
	"""

	reference_prefix="PD"

	def define(self, Efficiency = 0.7, Bias = 1., Jitter = 50., Vout = 1., Deadtime = 10.):
		click_in = self.add_port(name='click_in', kind=port.kind.digital, direction=port.direction.inp)
		inp = self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		out  = self.add_port(name='out', kind=port.kind.voltage, direction=port.direction.out)

		vout = self.add_port(name='vout', kind=port.kind.real, direction=port.direction.inp, default = Vout)
		deadtime = self.add_port(name='deadtime', kind=port.kind.real, direction=port.direction.inp, default = Deadtime)
		eff = self.add_port(name='efficiency', kind=port.kind.real, direction=port.direction.inp, default = Efficiency)
		bias = self.add_port(name='bias', kind=port.kind.real, direction=port.direction.inp, default = Bias)
		jitter = self.add_port(name='jitter', kind=port.kind.real, direction=port.direction.inp, default = Jitter)
		
		self.add_model(DetectorModel('detector '+self.name, block=self, out_exprs={out : click_in*vout  } ))
		
		
		
		
class PhotoDiode(Block):

	"""
	Basic detector block with simple efficiency function. 


	Desired class attributes: efficiency, bandwidthm, noise
	spectral range
	"""

	reference_prefix="PM"

	def define(self, Efficiency = 0.7, Noise = 0.0):
		inp = self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		out  = self.add_port(name='out', kind=port.kind.voltage, direction=port.direction.out)
	
		eff = self.add_port(name='efficiency', kind=port.kind.real, direction=port.direction.inp, default = Efficiency)
		noise = self.add_port(name='noise', kind=port.kind.real, direction=port.direction.inp, default = np.random.normal(0,Noise))

		self.add_model(DetectorModel('detector '+self.name, block=self, out_exprs={out :  cos(6.64534*acos(abs(inp)**2)*(1-acos(abs(inp)**2)))**2} ))

















