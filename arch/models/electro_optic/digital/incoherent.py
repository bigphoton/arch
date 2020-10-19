"""
Functions and objects describing incoherent detection .
"""


import numpy as np
from arch.models import model

class detector_basic(model):
    """
	Model for single mode optical input, single digital ouput, with given efficiency
	"""

    def __init__(self, model_params):
        super(type(self), self).__init__()
		
        self.model_params=model_params

    def update_params(self,new_params):
        self.model_params=new_params


    def compute(self,input_vector):

        #Get values from input ports
        vin=float(np.abs(input_vector[0].value))
        vout=np.array([1])#np.array([np.random.choice([0,1],p=[1-np.abs(vin),np.abs(vin)])])
        
        return vout.flat
    
    