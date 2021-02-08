
# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

try:
	import colored_traceback.auto
except ImportError:
	pass
	
import abc
import arch.port as port
from arch.port import var
from arch.block import Block
from arch.connectivity import Connectivity
from arch.models import Model, SymbolicModel, NumericModel, SourceModel
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder, Interferometer
from arch.blocks.sources import LaserCW, BasicSinglePhotonSource
from arch.architecture import Architecture
from arch.simulations.simulations import InterferometerSimulator
import time
import math
from sympy import Matrix


"""
This is a basic example which (will) simulate a simple HOM dip experiment.
It calculates the beamsplitter by calculating the permanent for each
possible transition and the

"""


if __name__ == '__main__':
	
	#Ideally these would be input ports of the IF class and you wouldnt need to specify them
	#upon instantiation. Could not get to instantiate without explicit declaration.
	bs_mat= Matrix([[1,1j],[1j,1]])*(1/math.sqrt(2))
	identity= Matrix([[1,0],[0,1]])
	bs = Interferometer(unitary=bs_mat)
	bs1=Interferometer(unitary=identity)


	#Define connectivity
	connections = Connectivity( [(bs.out1,bs1.in1),
								(bs.out0, bs1.in0)	])


	#Draw connections graph and then compound the model
	connections.draw(draw_ports=True)
	cm = bs.model.compound("compound name", connections.models, connections)
	compound_unitary=cm.U

	#Hard code in an input state - hopefully generate via source blocks soon
	state={(0, 0): 0j, (0, 1): 0j, (0, 2): 0j, (1, 0): 0j, (1, 1): 1, (1, 2): 0j, (2, 0): 0j, (2, 1): 0j, (2, 2): 0j}
	
	print("Setting up simulator...")
	sim = InterferometerSimulator(
				unitary=compound_unitary,
				input_state=state)

	print("Simulating...")
	sim.run()

	print('output state is', sim.output_state)


	