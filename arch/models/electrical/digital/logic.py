"""
Functions and objects describing digital logic.
"""

import numpy
from arch.models import model


def _bin_to_int(b_list):
	"""
	Convert b_list, a list of {0,1}, to an integer
	"""
	out = 0
	for b in b_list:
		out = (out << 1) | b
	return out

def _int_to_bin(i, n):
	"""
	Convert integer i to a list of {0,1} of length n
	"""
	return [int(x) for x in list('{0:0{1}b}'.format(i,n))]


class combinatorial(model):
	"""
	Digital input digital output model.
	truth_table: list of ints truth_table[vin] = vout (for binary vin, vout)
	"""
	
	def __init__(self, truth_table, n_output_bits):
		super(type(self), self).__init__()
		
		self.truth_table = truth_table
		self.n_output_bits = n_output_bits
	
	
	def compute(self, input_vector):
		# Get integer input value from port
		in_int = _bin_to_int([e.value for e in input_vector])
		
		# Get truth table output
		out_int = self.truth_table[in_int]
		
		# Get the corresponding binary list
		vout = _int_to_bin(out_int, self.n_output_bits)
		
		return vout