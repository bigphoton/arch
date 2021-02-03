
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
	bs = Beamsplitter()
	bs1 = Beamsplitter()
	
	con_mz = Connectivity( [
					(bs.out1, ps.inp),
					(ps.out, bs1.in1),
					(bs.out0, bs1.in0),
					] )
	
	con_bs_only = Connectivity( [
					(bs.out1, ps.inp),
					(ps.out, bs1.in1),
					(bs1.out1, bs.in1),
					] )
	
	con_ring = Connectivity( [
					(bs.out1, ps.inp),
					(ps.out, bs.in1),
					] )
	
	con_double_ring = Connectivity( [
					(bs.out1, ps.inp),
					(ps.out, bs1.in1),
					(bs1.out1, bs.in1),
					(bs1.out0, bs1.in0),
					] )
	
	con = con_ring
	
# 	con.draw()
	
	cm = Linear.compound("compound name", con.models, con)
	
	
	
# 	import IPython
# 	IPython.embed()
# 	quit()