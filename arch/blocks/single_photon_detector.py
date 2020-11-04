"""
Functions and objects describing single photon detectors
"""

import numpy as np
from arch.block import base_block
from arch.vis.generic import generic_box, generic_port
from arch.port import port
from arch.models.electro_optic.digital.incoherent import basic_linear_detector, monte_carlo_single_photon_detector, basic_quantum_state_detector

class basic_spd(base_block):

    """
    Basic detector block with simple efficiency function. User must choose a model of 
    simulation from: 'monte_carlo', 'linear', 'full_quantum'


    Desired class attributes: efficiency, dead time, dark counts, jitter,
    spectral range, PNR
    """

    reference_prefix="SPD"

    def define(self, model_choice, efficiency=0.7):
        #setup display box
        w=generic_box.box_width
        h=generic_box.box_height
        l=generic_port.port_length
        x0,y0=self.position
        self.model_choice = model_choice

        #add single optical input
        name="IN"
        self.ports.add(port(name,"optical", True, self, None, 1 ,(x0,h/2+y0),0))
        self.in_port_order.append(name)

        #add a single digital output
        name="OUT"
        self.ports.add(port(name,"digital", False, self, None, 1, (w/2+x0+1,-h/2+(1/2)*h+y0),180))
        self.out_port_order.append(name)

        #setup graphic
        self.graphic=generic_box(self.reference_designator,position=self.position)

        # Model parameter(s)
        self.model_params.update({'efficiency':efficiency})

        #set model
        if model_choice=='monte_carlo':
            self.model=monte_carlo_single_photon_detector(efficiency, self.model_params)
        elif model_choice=='full_quantum':
            self.model=basic_quantum_state_detector(efficiency, self.model_params)
        elif model_choice=='linear':
            self.model=basic_linear_detector(efficiency,self.model_params)