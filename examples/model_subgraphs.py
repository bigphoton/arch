
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
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder
from arch.blocks.sources import LaserCW
from arch.blocks.electro_optics import Switch2x2
from arch.blocks.logic import NotGate
from arch.architecture import Architecture


if __name__ == '__main__':
	
	print ("Welcome to the new arch")
	
	laser = LaserCW()
	bs = Beamsplitter()
	sw = Switch2x2(loss_dB=0, extinction_ratio=float('inf'))
	notg = NotGate()
	
	con = Connectivity( [
			(laser.out, bs.in0),
			(bs.out1, sw.in0),
			(notg.out, sw.state),
			] )
	
	con.draw()
		
	import networkx as nx
	