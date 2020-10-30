"""
Example digital logic.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys
sys.path.insert(0,'..')

from arch.blocks import logic

# AND - NOT

first_gate = logic.and_gate()
first_gate.position = (-200,+20)

second_gate = logic.not_gate()
second_gate.position = (+200,+20)


first_gate.ports['OUT'].connect(second_gate.ports['IN'])


print("{:6} -> {:}".format("input","!AND"))
for a,b in [(0,0),(0,1),(1,0),(1,1)]:
	first_gate.ports['IN0'].value = a
	first_gate.ports['IN1'].value = b
	
	first_gate.compute()
	second_gate.compute()
	
	print("{:} -> {:}".format((a,b), second_gate.ports['OUT'].value))


# NAND - NOT

first_gate = logic.nand_gate()
second_gate = logic.not_gate()

first_gate.ports['OUT'].connect(second_gate.ports['IN'])

print("{:6} -> {:}".format("input","!NAND"))
for a,b in [(0,0),(0,1),(1,0),(1,1)]:
	first_gate.ports['IN0'].value = a
	first_gate.ports['IN1'].value = b
	
	first_gate.compute()
	second_gate.compute()
	
	print("{:} -> {:}".format((a,b), second_gate.ports['OUT'].value))


# OR - NOT

first_gate = logic.or_gate()
second_gate = logic.not_gate()

first_gate.ports['OUT'].connect(second_gate.ports['IN'])

print("{:6} -> {:}".format("input","!OR"))
for a,b in [(0,0),(0,1),(1,0),(1,1)]:
	first_gate.ports['IN0'].value = a
	first_gate.ports['IN1'].value = b
	
	first_gate.compute()
	second_gate.compute()
	
	print("{:} -> {:}".format((a,b), second_gate.ports['OUT'].value))