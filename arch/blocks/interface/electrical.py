"""
Functions and objects describing single photon sources
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0]))))
import numpy as np
from arch.block import Block
import arch.port as port
from arch.models.model import SymbolicModel, FourChLogPlexModel, FourTimeSpacePlexModel
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

class Amplifier(Block):
	"""
	Digitises a real input.
	
	threshold: 0 if input < threshold else 1
	hysteresis: additional inp change above or below threshold to cause out change
	"""
	
	reference_prefix = "COMP"
	
	def define(self, gain = np.pi):
		
		inp = self.add_port(name='inp', kind=port.kind.real, direction=port.direction.inp)
		gain = self.add_port(name='gain', kind=port.kind.real, direction=port.direction.inp)

		out = self.add_port(name='out', kind=port.kind.digital, direction=port.direction.out)
		
		self.add_model(
			AmplifierModel('amplifier '+self.name, block=self, 
				out_exprs={out: gain*inp} ))
				
				
				
class FourChLogPlex(Block):

	
	reference_prefix = "LOG"
	
	def define(self, vout = np.pi, threshold = 0.1, hyst = 0.01):
		
		inp1 = self.add_port(name='in0', kind=port.kind.voltage, direction=port.direction.inp)
		inp2 = self.add_port(name='in1', kind=port.kind.voltage, direction=port.direction.inp)
		inp3 = self.add_port(name='in2', kind=port.kind.voltage, direction=port.direction.inp)
		inp4 = self.add_port(name='in3', kind=port.kind.voltage, direction=port.direction.inp)

		out1 = self.add_port(name='out0', kind=port.kind.voltage, direction=port.direction.out)
		out2 = self.add_port(name='out1', kind=port.kind.voltage, direction=port.direction.out)


		self.add_model(
			FourChLogPlexModel('logic ' + self.name, block=self, ))
				
class FourTimeSpatPlex(Block):

	
	reference_prefix = "LOG"
	
	def define(self, vout = np.pi, threshold = 0.1, hyst = 0.01):
		
		inp1 = self.add_port(name='in0', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		inp2 = self.add_port(name='in1', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		inp3 = self.add_port(name='in2', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		inp4 = self.add_port(name='in3', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		
		stored0 = self.add_port(name='storedi0', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		stored1 = self.add_port(name='storedi1', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		stored2 = self.add_port(name='storedi2', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		stored3 = self.add_port(name='storedi3', kind=port.kind.voltage, direction=port.direction.inp, default = 0)

		
		clki  = self.add_port(name='clkl', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		clki  = self.add_port(name='clki', kind=port.kind.voltage, direction=port.direction.inp, default = 0)
		clkis = self.add_port(name='clkis', kind=port.kind.voltage, direction=port.direction.inp, default = 0)

		
		out1 = self.add_port(name='outs0', kind=port.kind.voltage, direction=port.direction.out)
		out2 = self.add_port(name='outs1', kind=port.kind.voltage, direction=port.direction.out)
		
		out1 = self.add_port(name='outt0', kind=port.kind.voltage, direction=port.direction.out)
		out2 = self.add_port(name='outt1', kind=port.kind.voltage, direction=port.direction.out)
		out1 = self.add_port(name='outt2', kind=port.kind.voltage, direction=port.direction.out)
		out2 = self.add_port(name='outt3', kind=port.kind.voltage, direction=port.direction.out)
		
		stored0 = self.add_port(name='storedo0', kind=port.kind.voltage, direction=port.direction.out)
		stored1 = self.add_port(name='storedo1', kind=port.kind.voltage, direction=port.direction.out)
		stored2 = self.add_port(name='storedo2', kind=port.kind.voltage, direction=port.direction.out)
		stored3 = self.add_port(name='storedo3', kind=port.kind.voltage, direction=port.direction.out)
		# clko = self.add_port(name='lastbin', kind=port.kind.voltage, direction=port.direction.out, default = 0)
		clko = self.add_port(name='clko', kind=port.kind.voltage, direction=port.direction.out, default = 0)
		clks = self.add_port(name='clks', kind=port.kind.voltage, direction=port.direction.out, default = 0)
		


		self.add_model(
			FourTimeSpacePlexModel('logic ' + self.name, block=self, 

				))
				
				
