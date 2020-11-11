import __init__
import unittest
import importlib
import os,sys
import numpy as np


from arch.models.optical.quantum.permanents import permanent


__all__ = ['test_permanents']

class test_permanents(unittest.TestCase):
    '''
    Test the permanents model for calculating transition amplitudes of optical elements.
    Ran into issues testing nested functions so the model isnt fully tested yet!
    '''
    def test_generate_outcomes_zeros(self):
        out=permanent.generate_possible_outcomes(self,[0,0,0,0])
        self.assertEqual(out,[[0,0,0,0,]], 'Incorrect possible outcomes')

    def test_generate_outcomes_negative(self):
        with self.assertRaises(Exception) as context:
            permanent.generate_possible_outcomes(self,[-1])

        self.assertTrue('Negative input is unphysical' in str(context.exception), 'not raising error for negative input')


    def test_transition_matrix_few_photons(self):
        unitary=np.array([['a','b','c'],['d','e','f'],['g','h','i']],dtype=str)
        out=permanent.create_transition_matrix(self,unitary,[1,0,0],[0,1,0],d=str)
        self.assertEqual(out,['b'], 'transition matrix not being created properly for fewer photons than modes')
    

    def test_transition_matrix_many_photons(self):
        unitary=np.array([['a','b','c'],['d','e','f'],['g','h','i']],dtype=str)
        out=permanent.create_transition_matrix(self,unitary,[3,1,3],[1,3,3],d=str)
        np.testing.assert_array_equal(out,[['a','a','a','b','c','c','c'],['d','d','d','e','f','f','f'],['d','d','d','e','f','f','f'],['d','d','d','e','f','f','f'],['g','g','g','h','i','i','i'],['g','g','g','h','i','i','i'],['g','g','g','h','i','i','i']], err_msg='transition matrix failing for more photons than modes')
        
    def test_transition_matrix_equal_photons_modes(self):
        unitary=np.array([['a','b','c'],['d','e','f'],['g','h','i']],dtype=str)
        out=permanent.create_transition_matrix(self,unitary,[1,1,1],[1,1,1],d=str)
        np.testing.assert_array_equal(out,[['a','b','c'],['d','e','f'],['g','h','i']], err_msg='transition matrix failing for equal photons and modes')
        
    def test_transition_matrix_complex(self):
        unitary=np.array([[1,1j],[1j,1]])
        out=permanent.create_transition_matrix(self,unitary,[1,1],[1,1])
        np.testing.assert_array_equal(out,unitary)







if __name__ == "__main__":
    import sys;sys.argv = ['', 'Test.testName']
    unittest.main()