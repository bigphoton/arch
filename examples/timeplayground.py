#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:50:44 2021

@author: marijaradulovic
"""

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
from arch.blocks.sources import  LaserCW
from arch.architecture import Architecture



if __name__ == '__main__':
	

	print ("Testing SymbolicModel composition")
	
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
		
	mod_b = SymbolicModel("mod_b", ports={port_a, port_bo}, out_func=func_bo)
	
	ports_d = {port_bi, port_c}
	def func_d(port_dict):
		b = port_dict[port_bi]
		c = port_dict[port_c]
		return {port_d: b + c}
	
	mod_d = SymbolicModel("mod_d", ports={port_bi, port_c, port_d}, out_func=func_d)
	
	con = Connectivity( [(port_bo, port_bi)] )
	
	mod_comp = SymbolicModel.compound(name="comp", models={mod_b, mod_d}, connectivity=con)
	
	state = {port_a:2, port_c:4}
	print(mod_comp.out_func(state))
	print(func_d({port_bi:3,port_c:8}))
	print(state)
	
	
	
	print ("Welcome to the new arch")