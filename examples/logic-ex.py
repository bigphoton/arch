"""
Example digital logic.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,'..')

from arch.blocks import electronics



first_gate = electronics.not_gate()
first_gate.position = (-200,+20)

second_gate = electronics.not_gate()
second_gate.position = (+200,+20)


first_gate.ports['OUT'].connect(second_gate.ports['IN'])


for input_val in [0,1]:
	first_gate.ports['IN'].value = input_val
	
	first_gate.compute()
	second_gate.compute()
	
	print("gate1 in={:}, output={:}".format(input_val,first_gate.out_ports))
	print("gate2 in={:}, output={:}".format(input_val,second_gate.out_ports))