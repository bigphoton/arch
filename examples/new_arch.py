
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
from arch.blocks.optics import Beamsplitter, PhaseShifter, MachZehnder, LaserCW
from arch.architecture import Architecture


if __name__ == '__main__':
	
	"""
	print ("Testing NumericModel composition")
	
	port_a  = var("a",  kind=port.kind.real, direction=port.direction.inp, default=0)
	port_bo = var("bo", kind=port.kind.real, direction=port.direction.out, default=0)
	port_bi = var("bi", kind=port.kind.real, direction=port.direction.inp, default=0)
	port_c  = var("c",  kind=port.kind.real, direction=port.direction.inp, default=0)
	port_d  = var("d",  kind=port.kind.real, direction=port.direction.out, default=0)
	
	# Funcs need to take a dict as input, keyed by port
	#  return a dict as output, keyed by port
	
	ports_bo = {port_a}
	def func_bo(port_dict):
		a = port_dict[port_a]
		return {port_bo: 2*a}
		
	mod_b = NumericModel("mod_b", ports={port_a, port_bo}, out_func=func_bo)
	
	ports_d = {port_bi, port_c}
	def func_d(port_dict):
		b = port_dict[port_bi]
		c = port_dict[port_c]
		return {port_d: b + c}
	
	mod_d = NumericModel("mod_d", ports={port_bi, port_c, port_d}, out_func=func_d)
	
	con = Connectivity( [(port_bo, port_bi)] )
	
	mod_comp = NumericModel.compound(name="comp", models={mod_b, mod_d}, connectivity=con)
	
	state = {port_a:1, port_c:0}
	print(mod_comp.out_func(state))
	print(state)
	"""
	
	
	
	print ("Welcome to the new arch")
	
	laser = LaserCW()
	
	mz0 = MachZehnder()
	mz1 = MachZehnder()#mz0.copy()
# 	
# 	print(mz0.model.out_exprs)
# 	
	ps0 = PhaseShifter()
	ps1 = PhaseShifter()
# 	
	bs = Beamsplitter()
	
	connections = Connectivity( [
					(laser.out, mz0.in0),
					(mz0.out0, ps0.inp),
					(mz0.out1, ps1.inp),
					(ps0.out, bs.in0),
					(ps1.out, bs.in1),
					(bs.out0, mz1.in0),
					(bs.out1, mz1.in1),
					] )
	
	cm = laser.model.compound("compound name", [laser.model, mz0.model], connections)
	
	
	print(cm)
	
	state = cm.default_input_state
	cm.out_func(state)
	print(state)
	
	
# 	from arch.block import AutoCompoundBlock
# 	acb = AutoCompoundBlock(blocks=[laser,mz0], connectivity=connections)
# 	print("ACB outs:",acb.model.out_exprs)
	
# 	acb = AutoCompoundBlock(blocks=[laser,mz0,ps0,ps1,mz1,bs], connectivity=connections)
# 	print("Can use favourite port labels:", mz0.in1 is acb.MZ0_in1)
	
# 	from math import pi
# 	out_dict = acb.model.out_func(in_port_values={mz0.in0:1, mz0.in1:0, mz1.phi:pi/2})
	
# 	acb.connectivity.draw(draw_ports=True)
# 	acb.connectivity.draw(draw_ports=False)
	
	
	
# 	arch = Architecture(connections, None)
	
# 	print([(sym,sym.data) for m in ps.models for sym in m.out_exprs[0].free_symbols])