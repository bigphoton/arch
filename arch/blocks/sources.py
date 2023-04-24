"""
Functions and objects describing single photon sources
"""

from arch.block import Block
import arch.port as port
from arch.models.model import SourceModel
from sympy import Matrix, sqrt, exp, I, eye
from arch.models.model import Linear, LinearGroupDelay
from sympy import sqrt, exp, I

import arch.qfunc


class LaserCW(Block):
	"""
	A laser component. Connect to other optics for classical values. Can be driven in time with in_time_funcs
	"""
	reference_prefix = "CW"
	
	def define(self, Power = 1.0, Wavelength= 2000):
		
		Wavelength = self.add_port(name='lambda', kind=port.kind.real, direction=port.direction.inp, default = Wavelength)
		
		P = self.add_port(name='P', kind=port.kind.real, direction=port.direction.inp, default = Power)
		phase = self.add_port(name='phase', kind=port.kind.real, direction=port.direction.inp, default=0.0)
		
		out = self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		
		self.add_model(SourceModel('laser '+self.name, block=self, out_exprs={out:sqrt(P)*exp(I*phase)} ))


class BasicPhotonPairSource(Block):
	"""	 
	A simple squeezed vacuum source
	JCA 2023
	"""

	reference_prefix="SV"

	def define(self, xi = 0.1, wgs = ['wg0','wg0'], pos = [0, 0], freq = ['sig', 'idl'], hg = [0, 0], cutoff = 2 , eta = 1.0, reprate = 25):
		
		self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		out = self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		# self.add_model(SourceModel('pair photon source'+self.name, block=self, out_exprs={out:sqrt(P)} ))
		
		p1 = self.add_port(name='wgs', kind=port.kind.real, direction=port.direction.inp, default = wgs)
		p2 = self.add_port(name='pos', kind=port.kind.real, direction=port.direction.inp, default = pos)
		p3 = self.add_port(name='freq', kind=port.kind.real, direction=port.direction.inp, default = freq)
		p4 = self.add_port(name='hg', kind=port.kind.real, direction=port.direction.inp, default = hg)
		p5 = self.add_port(name='cutoff', kind=port.kind.real, direction=port.direction.inp, default = cutoff)
		p6 = self.add_port(name='reprate', kind=port.kind.real, direction=port.direction.inp, default = reprate)
		
		# ostate = arch.qfunc.sqz_vac(xi, wgs, pos, freq, hg, cutoff)
		
		r = self.add_port(name='eta', kind=port.kind.real, direction=port.direction.inp, default = eta)
		M = Matrix([[1]])
		
		self.add_model(Linear('swfm_src '+self.name, block=self, unitary_matrix=M))
		
		# self.add_model(SourceModel('pair photon source'+self.name, block=self, out_exprs={out:ostate} ))


class BasicSinglePhotonSource(Block):
	"""
	A simple single photon source
	JCA 2023
	"""

	reference_prefix = "SPS"

	def define(self,  wgs = 'wg0', pos = 0, freq = 'sig', hg = 0, eta = 1.0, reprate = 25):
		
		self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
		out = self.add_port(name='out', kind=port.kind.optical, direction=port.direction.out)
		# self.add_model(SourceModel('pair photon source'+self.name, block=self, out_exprs={out:sqrt(P)} ))
		
		p1 = self.add_port(name='wgs', kind=port.kind.real, direction=port.direction.inp, default = wgs)
		p2 = self.add_port(name='pos', kind=port.kind.real, direction=port.direction.inp, default = pos)
		p3 = self.add_port(name='freq', kind=port.kind.real, direction=port.direction.inp, default = freq)
		p4 = self.add_port(name='hg', kind=port.kind.real, direction=port.direction.inp, default = hg)
		p6 = self.add_port(name='reprate', kind=port.kind.real, direction=port.direction.inp, default = reprate)
		
		# ostate = arch.qfunc.sqz_vac(xi, wgs, pos, freq, hg, cutoff)
		
		r = self.add_port(name='eta', kind=port.kind.real, direction=port.direction.inp, default = eta)
		M = Matrix([[1]])
		
		self.add_model(Linear('sps_src '+self.name, block=self, unitary_matrix=M))
		
		# self.add_model(SourceModel('pair photon source'+self.name, block=self, out_exprs={out:ostate} ))