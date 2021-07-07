from arch.state import State
import numpy as np


"""
Classes of quantum state with different 
properties.
"""


class MonochromaticFockState(State):


    reference_prefix='MFS'

    def define(self):

        self.data_dict={
            'Fock_State':{}
            'Wavelength': 
            }