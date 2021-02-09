
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
	
	ps0 = PhaseShifter()
	ps1 = PhaseShifter()
	ps2 = PhaseShifter()
	bs0  = Beamsplitter()
	bs1 = Beamsplitter()
	
	con_mz = Connectivity( [
					(ps0.out, bs0.in0),
					(bs0.out1, ps1.inp),
					(ps1.out, bs1.in1),
					(bs0.out0, bs1.in0),
					] )
	
	con_bs_only = Connectivity( [
					(ps0.out, bs0.in0),
					(bs0.out1, ps1.inp),
					(ps1.out, bs1.in1),
					(bs1.out1, bs0.in1),
					] )
	
	con_ring = Connectivity( [
					(ps0.out, bs0.in0),
					(bs0.out1, ps1.inp),
					(ps1.out, bs0.in1),
					] )
	
	con_double_ring = Connectivity( [
					(ps0.out, bs0.in0),
					(bs0.out1, ps1.inp),
					(ps1.out, bs1.in1),
					(bs1.out1, bs0.in1),
					(bs1.out0, ps2.inp),
					(ps2.out, bs1.in0),
					] )
	
	cons = [(con_double_ring,'double ring'), (con_mz,'mz'), 
			(con_bs_only,'bs only'), (con_ring,'ring')]
	
	for con, name in cons:
		
		print('|'*80 + f"\n\nConnectivity '{name}'")
# 		con.draw()
		
		print("Compounding")
		cm = Linear.compound("my compound", con.models, con)
		
		if type(cm) == Linear:
			print("Linear: Success!")
		else:
			print(f"{type(cm)}: Failure...")
			quit()
		
		print(f"and it has this unitary matrix:")
		from sympy import pprint
		pprint(cm.U)
	
	
	
	
# 	import IPython
# 	IPython.embed()
# 	quit()