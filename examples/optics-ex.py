"""
Example optics.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,'..')

from arch.blocks import optics
from arch.blocks import single_photon_detector

import numpy as np
    
print ("Hello world")


from time import sleep

spd1=single_photon_detector.basic_spd(efficiency=0.7)
spd1.position= (+200,+100)

spd2=single_photon_detector.basic_spd(efficiency=0.7)
spd2.position= (+200,-100)


bs0 = optics.beamsplitter(reflectivity=0.5)
bs0.position = (-200,+20)

bs1 = optics.beamsplitter()
bs1.position = (+200,+20)

bs0.ports['OUT0'].connect(bs1.ports['IN0'])
bs0.ports['OUT1'].connect(bs1.ports['IN1'])


bs1.ports['OUT0'].connect(spd1.ports['IN'])
bs1.ports['OUT1'].connect(spd2.ports['IN'])

bs0.ports['IN0'].value = 1.0
bs0.ports['IN1'].value = 0.0

sleep(0.5)


for i in range(10):
    bs0.compute()
    bs1.compute()
    spd1.compute()
    spd2.compute()

    print("\n \n Detection events are: Detector 1:",spd1.ports['OUT'].value, "  Detector 2:",spd2.ports['OUT'].value)
