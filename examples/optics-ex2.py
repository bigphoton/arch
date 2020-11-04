"""
Example optics with detectors

THIS MODEL OF DETECTION DOES NOT PRESERVE PHOTON NUMBER, JUST A PLACEHOLDER
"""


import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from arch.blocks import optics
from arch.blocks import single_photon_detector 
import numpy as np
	
print ("Hello world")


from time import sleep

bs0 = optics.beamsplitter(reflectivity=0.5, model_choice='linear')
bs0.position = (-100,+20)

ps = optics.phase_shift(phase=0)
ps.position = (0,-40)

bs1 = optics.beamsplitter(model_choice='linear')
bs1.position = (+100,+20)

spd1=single_photon_detector.basic_spd(model_choice='linear', efficiency=0.7)
spd1.position= (+200,+100)

spd2=single_photon_detector.basic_spd(model_choice='linear', efficiency=0.7)
spd2.position= (+200,-100)

bs0.ports['OUT0'].connect(ps.ports['IN'])
ps.ports['OUT'].connect(bs1.ports['IN0'])
bs0.ports['OUT1'].connect(bs1.ports['IN1'])

bs1.ports['OUT0'].connect(spd1.ports['IN'])
bs1.ports['OUT1'].connect(spd2.ports['IN'])

bs0.ports['IN0'].value = 1
bs0.ports['IN1'].value = 1

def test_beamsplitter():
	for i in range(10):
		bs0.compute()
		print("Beamsplitter1 outputs are Port1:" ,bs0.ports['OUT0'].value,"Port 2:", bs0.ports['OUT1'].value )


test_beamsplitter()
