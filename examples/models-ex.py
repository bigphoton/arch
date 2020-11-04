"""
Example using on click sources and the architecture class to simulate a HOM dip via full state evolution, or through 
monte - carlo sim
"""

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

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

#This can be either "full_quantum" or "monte_carlo"
model='full_quantum'

#Block declarations
s2=sources.black_box_single_photon_source(model_choice=model)
s3=sources.black_box_single_photon_source(model_choice=model)
bs1=optics.beamsplitter(model_choice=model)
spd1=single_photon_detector.basic_spd(efficiency=1, model_choice=model)
spd2=single_photon_detector.basic_spd(efficiency=1, model_choice=model)


#define block connections
s2.ports['OUT'].connect(bs1.ports['IN0'])
s3.ports['OUT'].connect(bs1.ports['IN1'])
bs1.ports['OUT0'].connect(spd1.ports['IN'])
bs1.ports['OUT1'].connect(spd2.ports['IN'])


#define input values
s2.ports['IN'].value={}
s3.ports['IN'].value={}
s2.ports['IN'].value['digital_input_signal']=1
s3.ports['IN'].value['digital_input_signal']=1


# We can use the old drawing functionality by calling .draw() manually
if False:
	for b in [s2,s3, bs1, spd1, spd2]:
		b.graphic.draw()
		for p in b.ports:
			p.graphic.draw()

arch = architecture(blocks=[s2, s3,bs1, spd1, spd2], global_model=model)

if model=='monte_carlo':

	for i in range(10):
		arch.compute()
		print("\n  Detection events are: Detector 1:",spd1.ports['OUT'].value, "  Detector 2:",spd2.ports['OUT'].value)
else:
	arch.compute()
	print("\n  Detection events are: Detector 1:",spd1.ports['OUT'].value, "  Detector 2:",spd2.ports['OUT'].value)
	

arch.draw()



