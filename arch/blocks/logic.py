"""
Functions and objects describing logic gates.
"""


from arch.block import Block


class NotGate(Block):
	"""
	A not gate: 0 -> 1, 1 -> 0
	"""
	
	reference_prefix = "U"
	
	def define(self):
		raise NotImplementedError()
		

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