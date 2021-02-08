
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
import time
from sympy import Matrix


"""
This is a basic example which (will) simulate a simple HOM dip experiment.
It calculates the beamsplitter by calculating the permanent for each
possible transition and the

"""



if __name__ == '__main__':
	
	#Ideally these would be input ports of the IF class and you wouldnt need to specify them
	#upon instantiation. Could not get to instantiate without explicit declaration.
	bs_mat= Matrix([[1,1j],[1j,1]])
	mode_list=[0,1]

	bs = Interferometer()
	bs1=Interferometer()

	#Define connectivity
	connections = Connectivity( [(bs.out1,bs1.in1),
								(bs.out0, bs1.in0)	])

	#Draw connections graph and then compound the model
	connections.draw(draw_ports=True)
	cm = bs.model.compound("compound name", connections.models, connections)
	

	#Generates a defualt input state from the port definitions - this can then be populated
	inputs=cm.default_input_state
	inputs[bs.in0]={(0, 0): 0j, (0, 1): 0j, (0, 2): 0j, (1, 0): 0j, (1, 1): 1, (1, 2): 0j, (2, 0): 0j, (2, 1): 0j, (2, 2): 0j}
	inputs[bs.in1]={(0, 0): 0j, (0, 1): 0j, (0, 2): 0j, (1, 0): 0j, (1, 1): 1, (1, 2): 0j, (2, 0): 0j, (2, 1): 0j, (2, 2): 0j}
	inputs[bs.unitary] = bs_mat
	inputs[bs1.unitary] = bs_mat

	output_state = cm.out_func(inputs)
	print("default output state",output_state)

	