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
    Basic detector block with simple efficiency function. User must choose a model of 
    simulation from: 'monte_carlo', 'linear', 'full_quantum'


    Desired class attributes: efficiency, dead time, dark counts, jitter,
    spectral range, PNR
    """

    reference_prefix="SPD"

    def define(self, Efficiency = 0.7, Bias = 1., Jitter = 50.):
	
        eff = self.add_port(name='efficiency', kind=port.kind.real, direction=port.direction.inp, default = Efficiency)
        bias = self.add_port(name='bias', kind=port.kind.real, direction=port.direction.inp, default = Bias)
        jitter = self.add_port(name='jitter', kind=port.kind.real, direction=port.direction.inp, default = Jitter)
        
        inp = self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
       
        out  = self.add_port(name='out', kind=port.kind.voltage, direction=port.direction.out)

        self.add_model(DetectorModel('detector '+self.name, block=self, out_exprs={out : eff*abs(inp)**2} ))
		
		
		
		
class PhotoDiode(Block):

    """
    Basic detector block with simple efficiency function. User must choose a model of 
    simulation from: 'monte_carlo', 'linear', 'full_quantum'


    Desired class attributes: efficiency, dead time, dark counts, jitter,
    spectral range, PNR
    """

    reference_prefix="PD"

    def define(self, Efficiency = 0.7, Noise = 0.0):
	
        eff = self.add_port(name='efficiency', kind=port.kind.real, direction=port.direction.inp, default = Efficiency)
        noise = self.add_port(name='noise', kind=port.kind.real, direction=port.direction.inp, default = np.random.normal(0,Noise))
        
        inp = self.add_port(name='inp', kind=port.kind.optical, direction=port.direction.inp)
       
        out  = self.add_port(name='out', kind=port.kind.voltage, direction=port.direction.out)

        self.add_model(DetectorModel('detector '+self.name, block=self, out_exprs={out :  cos(6.64534*acos(abs(inp)**2)*(1-acos(abs(inp)**2)))**2} ))

















