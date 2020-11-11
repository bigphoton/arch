import __init__
import unittest
import importlib
import os, sys
import numpy as np

from arch.blocks.single_photon_detector import basic_spd


__all__ = ['test_detectors']

class test_detectors(unittest.TestCase):
    '''
    Test the basic detector model. 
    '''
    def test_perfect_detection(self):
        det=basic_spd(efficiency=1)
        det.ports['IN'].value = 1
        det.compute()
        self.assertEqual(det.ports['OUT'].value,1 )
    
    def test_dead_detector(self):
        det=basic_spd(efficiency=0)
        det.ports['IN'].value = 1
        det.compute()
        self.assertEqual(det.ports['OUT'].value,0 )

    def test_distribution(self):
        det=basic_spd(efficiency=0.5)
        outputs=[]
        mean=500
        std_dev=np.sqrt(0.5*1000*0.5)

        for i in range(1000):
            det.ports['IN'].value = 1
            det.compute()
            outputs.append(det.ports['OUT'].value)
        
        self.assertTrue( (mean-3*std_dev)<= np.sum(outputs) <= (mean+3*std_dev),'Detector distribution > 3 std devs from mean. Run again to check if just the 0.3pc chance, or if there is a problem. ')






if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()