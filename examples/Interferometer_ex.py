
# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

try:
	import colored_traceback.auto
except ImportError:
	pass
	

from arch.connectivity import Connectivity
from arch.blocks.optics import  Interferometer
from arch.simulations.simulations import InterferometerSimulator
import math
from sympy import Matrix


"""
Seb Currie - 24/05/21

This is a basic example which (will) simulate a simple HOM dip experiment.
It calculates the Fock state evolution by calculating the permanents of 
matrices describing each possible transition. These permanents are then 
used to construct the final output state. 

TODO: Fix bug in drawing functionality that stops you drawing standalone blocks.
This is why we connect the beamsplitter to a block that does the identity.
"""


if __name__ == '__main__':
	
	#Ideally these would be input ports of the IF class and you wouldnt need to specify them
	#upon instantiation. 
	bs_mat= Matrix([[1,1j],[1j,1]])*(1/math.sqrt(2))
	identity= Matrix([[1,0],[0,1]])
	bs = Interferometer(unitary=bs_mat)
	identity_block=Interferometer(unitary=identity)		

	
	
	#Define connectivity
	connections = Connectivity( [(bs.out1,identity_block.in1),
								(bs.out0, identity_block.in0)	])		#Connections wont draw without the identity block


	#Draw connections graph and then compound the model
	connections.draw(draw_ports=True)
	cm = bs.model.compound("compound name", connections.models, connections)
	compound_unitary=cm.U


	#Hard code in an input state - TODO: Generate via source block
	state={(0, 0): 0j, (0, 1): 0j, (0, 2): 0j, (1, 0): 0j, (1, 1): 1, (1, 2): 0j, (2, 0): 0j, (2, 1): 0j, (2, 2): 0j}


	print(" \n Input state is: " , state)
	print("\n unitary is: " , bs_mat)
	print(" \n Setting up simulator...")
	sim = InterferometerSimulator(
				unitary=compound_unitary,
				input_state=state)

	print("Simulating...")
	sim.run()

	print(' \n output state is', sim.output_state)


	