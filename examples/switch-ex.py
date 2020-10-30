"""
Example for switches.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from arch.blocks import electro_optics
from arch.blocks import optics

import numpy as np
	
print ("Switch example start...")



bs = optics.beamsplitter(reflectivity=0.5)
bs.position = (-200,+20)

switch = electro_optics.switch_2x2(extinction_ratio = float('inf'), loss_dB = 1.0)

bs.ports['OUT0'].connect(switch.ports['IN0'])


bs.ports['IN0'].value = 1.0
bs.ports['IN1'].value = 0.0


for switch_state in [0, 1]:
	switch.ports['DIG'].value = switch_state
	bs.compute()
	switch.compute()
	out_ports = [bs.out_ports["OUT1"], switch.ports["OUT0"], switch.ports["OUT1"] ]
	print("switch state={:}, output={:}".format(switch_state, [p.value for p in out_ports]))
