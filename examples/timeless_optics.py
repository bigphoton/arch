
# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

try:
    import colored_traceback.auto
except ImportError:
    pass

from arch.models.model import Linear
from arch.connectivity import Connectivity
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder


if __name__ == '__main__':
	
	ps = PhaseShifter()
	ps1 = PhaseShifter()
	bs = Beamsplitter()
	
	
	con = Connectivity( [
					(bs.out1, ps.inp),
					(ps.out, ps1.inp),
					(ps1.out, bs.in1),
					] )
	
	
	cm = Linear.compound("compound name", [ps.model, bs.model], con)
	
	
	
# 	import IPython
# 	IPython.embed()
# 	quit()