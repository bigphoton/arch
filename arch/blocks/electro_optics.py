"""
Functions and objects describing electro-optic components.
"""


from arch.block import Block

class Switch2x2(Block):
	"""
	extinction_ratio: ratio of desired signal to undesired signal from wrong port
	loss_dB: positive number of decibels of loss (0 dB -> 100% tx; 10 dB -> 10% tx)
	"""
	
	reference_prefix = "SW"
	
	def define(self, loss_dB = 3.0, extinction_ratio=1000.0):
		raise NotImplementedError()