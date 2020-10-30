"""
Example architecture global.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,'..')

from arch.architecture import architecture
from arch.blocks import electro_optics
from arch.blocks import logic
from arch.blocks import optics

from numpy import linspace, pi


g = logic.not_gate()
bs0 = electro_optics.switch_2x2()
ps = optics.phase_shift(phase=0)
bs1 = optics.beamsplitter()
bs2 = optics.beamsplitter()

g.ports['OUT'].connect(bs0.ports['DIG'])
bs0.ports['OUT0'].connect(ps.ports['IN'])
ps.ports['OUT'].connect(bs1.ports['IN0'])
bs0.ports['OUT1'].connect(bs1.ports['IN1'])
bs1.ports['OUT0'].connect(bs2.ports['IN0'])
bs1.ports['OUT1'].connect(bs2.ports['IN1'])

g.ports['IN'].value = 1
bs0.ports['IN0'].value = 1.0
bs0.ports['IN1'].value = 0.0

# We can use the old drawing functionality by calling .draw() manually
if False:
	for b in [g, bs0, ps, bs1, bs2]:
		b.graphic.draw()
		for p in b.ports:
			p.graphic.draw()

arch = architecture(blocks=[g, bs0, ps, bs1, bs2])


for phase in linspace(0,pi,10):
	ps.phase = phase
	arch.compute()
	print("phase={:.3f}, output={:}".format(phase, bs2.out_ports))


arch.draw()