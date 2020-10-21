"""
Functions and objects describing incoherent detection. This class only currently works 
digital input (incident photon or not).
"""


import numpy as np
from arch.models import model
from arch.simulations.monte_carlo import monte_carlo

class detector_basic(model):
    """
	Model for digital single mode optical input, single digital ouput, with given efficiency
	"""

    def __init__(self, efficiency, model_params):
        super(type(self), self).__init__()
		
        self.model_params=model_params
        self.efficiency=efficiency

    def update_params(self,new_params):
        self.model_params=new_params



    def compute(self,input_vector):

        #Get values from input ports
        vin=float(np.abs(input_vector[0].value))

        #Get detector efficiency:
        efficiency=self.efficiency


        if vin>=1:
            vout = np.array(monte_carlo.simulate(self,[0,vin],[1-np.abs(efficiency),np.abs(efficiency)]))
        elif vin==0:
            vout=np.array([0])
        else:
            raise Exception('Detector block can only currently take binary input')
   
        return vout.flat
    
    