"""
Functions and objects describing logic gates.
"""


from arch.block import Block
import arch.port as port
from arch.models.electrical.digital.logic import CombinatorialN

class NotGate(Block):
	"""
	A not gate: 0 -> 1, 1 -> 0
	"""
	
	reference_prefix = "U"
	
	def define(self):
		
		self.add_port(name='inp', kind=port.kind.digital,
						direction=port.direction.inp)
		self.add_port(name='out', kind=port.kind.digital,
						direction=port.direction.out)
		
		self.add_model(CombinatorialN('numerical NOT '+self.name, block=self, 
							truth_table=[1,0], n_output_bits=1) )
		

class AndGate(Block):
	"""
	An and gate: 00 -> 0, 01 -> 0, 10 -> 0, 11 -> 1
	"""
	
	reference_prefix = "U"
	
	def define(self):
		raise NotImplementedError()


class OrGate(Block):
	"""
	An or gate: 00 -> 0, 01 -> 1, 10 -> 1, 11 -> 1
	"""
	
	reference_prefix = "U"
	
	def define(self):
		raise NotImplementedError()


class NandGate(Block):
	"""
	A nand gate: 00 -> 1, 01 -> 1, 10 -> 1, 11 -> 0
	"""
	
	reference_prefix = "U"
	
	def define(self):
		raise NotImplementedError()