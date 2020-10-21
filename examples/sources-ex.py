"""
Example using on click sources and the architecture class to manage global state evolution.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,"C:\\Users\\mr19164\\OneDrive - University of Bristol\\Documents\\PhD Project\\ArchCore\\arch\\")

from arch.blocks import single_photon_detector
from arch.blocks import electro_optics
from arch.blocks import logic
from arch.blocks import optics
from arch.blocks import sources
from arch.architecture import architecture
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import sleep






s1=sources.black_box_pair_photon_source()
bs1=optics.beamsplitter()
spd1=single_photon_detector.basic_spd(efficiency=1)
spd2=single_photon_detector.basic_spd(efficiency=1)

s1.ports['OUT0'].connect(bs1.ports['IN0'])
s1.ports['OUT1'].connect(bs1.ports['IN1'])
bs1.ports['OUT0'].connect(spd1.ports['IN'])
bs1.ports['OUT1'].connect(spd2.ports['IN'])

s1.ports['IN'].value=1.0


# We can use the old drawing functionality by calling .draw() manually
if False:
	for b in [s1, bs1, spd1, spd2]:
		b.graphic.draw()
		for p in b.ports:
			p.graphic.draw()

arch = architecture(blocks=[s1, bs1, spd1, spd2])


for i in range(10):
	arch.compute()
	print("\n  Detection events are: Detector 1:",spd1.ports['OUT'].value, "  Detector 2:",spd2.ports['OUT'].value)


arch.draw()
