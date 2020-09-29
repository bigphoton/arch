"""
Example optics.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,'..')

from arch.blocks import optics

import numpy as np
	
print ("Hello world")


from time import sleep

bs0 = optics.beamsplitter(reflectivity=0.5)
bs0.position = (-200,+20)

ps = optics.phase_shift(phase=0)
ps.position = (0,-40)

bs1 = optics.beamsplitter()
bs1.position = (+200,+20)

bs0.ports['OUT0'].connect(ps.ports['IN'])
ps.ports['OUT'].connect(bs1.ports['IN0'])
bs0.ports['OUT1'].connect(bs1.ports['IN1'])


bs0.ports['IN0'].value = 1.0
bs0.ports['IN1'].value = 0.0

sleep(0.5)


for phase in np.linspace(0,np.pi,10):
	ps.phase = phase
	bs0.compute()
	ps.compute()
	bs1.compute()
	print("phase={:.3f}, output={:}".format(phase,bs1.out_ports))
