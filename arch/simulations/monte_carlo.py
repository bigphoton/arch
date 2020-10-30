"""
Functions and objects describing monte carlo methods of sampling from probability distributions.

The probability distribution is generated via calculating the permanents of the matrices describing the transition amplitudes
for all possible outputs, given the input state. These probabilities are then fed into a MC method which samples from
the calculated distribution. This is akin to collapsing the wavefunction after each beamsplitter.

This is at the cost of global interference effects (no entanglement?) but can recreate local interefernce efffects
such as the HOM dip. More work to be done on the specific limits of this method.


"""

import numpy as np
import itertools
from arch.simulations import simulations

class monte_carlo(simulations):
    """
    Monte carlo method for simulating basic quantum optical systems by sampling from a probability distribution.
    unitary_matrix_func: function returning n-by-n complex numpy array; should be unitary
    """
    
    def __init__(self, unitary_matrix_func, model_params):
        super(type(self), self).__init__()
        
        self.unitary_matrix_func = unitary_matrix_func
        self.model_matrix = self.unitary_matrix_func(**model_params)
    
    
    def update_params(self, new_params):
        self.model_matrix = self.unitary_matrix_func(**new_params)
    

    def simulate(self, outcomes, probabilities):
        
        if np.sum(probabilities)!=1:
            raise Exception('Sum of probabilities is not 1, invalid distribution.')

        #random choice can only take 1D arrays, so cant pass outcomes directly
        vout_index=np.random.choice(np.arange(start=0,stop=len(outcomes),dtype=int),p=probabilities)
        vout=np.array(outcomes[vout_index])
        
    
        return vout.flat