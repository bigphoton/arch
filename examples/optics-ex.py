"""
Example optics with detectors

THIS MODEL OF DETECTION DOES NOT PRESERVE PHOTON NUMBER, JUST A PLACEHOLDER
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,"C:\\Users\\mr19164\\OneDrive - University of Bristol\\Documents\\PhD Project\\ArchCore\\arch\\")

from arch.blocks import optics
from arch.blocks import single_photon_detector 
import numpy as np
	
print ("Hello world")


from time import sleep

bs0 = optics.beamsplitter(reflectivity=0.5)
bs0.position = (-100,+20)

ps = optics.phase_shift(phase=0)
ps.position = (0,-40)

bs1 = optics.beamsplitter()
bs1.position = (+100,+20)

spd1=single_photon_detector.basic_spd(efficiency=0.7)
spd1.position= (+200,+100)

spd2=single_photon_detector.basic_spd(efficiency=0.7)
spd2.position= (+200,-100)

bs0.ports['OUT0'].connect(ps.ports['IN'])
ps.ports['OUT'].connect(bs1.ports['IN0'])
bs0.ports['OUT1'].connect(bs1.ports['IN1'])

bs1.ports['OUT0'].connect(spd1.ports['IN'])
bs1.ports['OUT1'].connect(spd2.ports['IN'])

bs0.ports['IN0'].value = 1.0
bs0.ports['IN1'].value = 0.0





for phase in np.linspace(0,np.pi,10):
	ps.phase = phase
	bs0.compute()
	ps.compute()
	bs1.compute()
	spd1.compute()
	spd2.compute()

	print("Beamsplitter1 outputs are Port1:" ,bs1.ports['OUT0'].value,"Port 2:", bs1.ports['OUT1'].value )
	print("Detection events are SPD1:",spd1.ports['OUT'].value,"SPD2:" ,spd2.ports['OUT'].value )

	#print("phase={:.3f}, output={:}".format(phase,bs1.out_ports))
sleep(10)