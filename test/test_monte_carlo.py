import unittest
import importlib
import os,sys

"""
TO DO: Add more nuanced tests which check on deviation from the mean etc.
"""


#sys.path.insert(0,"C:\\Users\\mr19164\\OneDrive - University of Bristol\\Documents\\PhD Project\\ArchCore\\arch\\")

from arch.simulations.monte_carlo import monte_carlo

__all__ = ['test_monte_carlo']

class test_monte_carlo(unittest.TestCase):
    '''
    Test the use of the monte carlo class to sample from probability distributions.
    '''

    def test_monte_carlo_definite_outcome(self):
        out=monte_carlo.simulate(self,['a','b','c'],[0,1,0])
        self.assertEqual(out,['b'], 'Monte carlo method failing on certain outcome')

    
    def test_monte_carlo_invalid_dist(self):
        with self.assertRaises(Exception) as context:
            monte_carlo.simulate(self,['a','b','c'],[0,0.5,0])

        self.assertTrue('Sum of probabilities is not 1, invalid distribution.' in str(context.exception), 'not raising error for negative input')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()