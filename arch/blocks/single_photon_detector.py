"""
Functions and objects describing single photon detectors
"""

from arch.block import Block

class BasicSPD(Block):

    """
    Basic detector block with simple efficiency function. User must choose a model of 
    simulation from: 'monte_carlo', 'linear', 'full_quantum'


    Desired class attributes: efficiency, dead time, dark counts, jitter,
    spectral range, PNR
    """

    reference_prefix="SPD"

    def define(self, model_choice, efficiency=0.7):
        raise NotImplementedError()