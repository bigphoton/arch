"""
Functions and objects describing digital logic.
"""

import numpy
from arch.models import NumericModel


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


class CombinatorialN(NumericModel):
	"""
	Digital input digital output model.
	
	truth_table: list of int, truth_table[int_in] = int_out
	"""
	
	def define(self, truth_table=[], n_output_bits=0, **kwargs):
		
		self.truth_table = truth_table
		self.n_output_bits = n_output_bits
		
		def out_func(state):
			
			# Convert between dict state, list of bools, and int representation
			vin = [state[p] for p in self.in_ports]
			iin = _bin_to_int(vin)
			iout = self.truth_table[iin]
			vout = _int_to_bin(iout, self.n_output_bits)
			
			return state | {self.out_ports[i]:vout[i] for i in range(len(self.out_ports))}
			
			return state | {p:v for p,v in zip(self.out_ports, vout)}
		
		self.out_func = out_func