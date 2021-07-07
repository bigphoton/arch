from arch.state import State
import numpy as np

"""
Classes of classical states with different 
properties.
"""


class MonochromaticChoherentState(State):
    reference_prefix = "MCCS"

    def define(self, wavelength, power):
        self.data = dict()

        self.data['wavelength'] = wavelength
        self.data['power'] = power


cw = MonochromaticChoherentState(wavelength=2, power=1)
print(cw.data['wavelength'])
