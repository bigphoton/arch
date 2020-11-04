"""
Functions and objects describing optical components.
"""

import numpy as np
from arch.models import model
import copy


class on_click_pair_source(model):
	"""
	Simple model for an on demand pair source. If digital input signal recieved, output a pair of photons in two spatial modes.
	"""
	
	def __init__(self,  model_params):
		super(type(self), self).__init__()
		
		self.model_matrix = model_params
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def compute(self, input_vector):
		# Get values from ports
		vin = np.array([e.value for e in input_vector])

		if vin==1:
			vout=np.array([1,1])
		elif vin==0:
			vout=np.array([0,0])
		else:
			raise Exception(' On click source can only currently take binary input')
		
		return vout.flat



class on_click_single_photon_source(model):
	"""
	Simple model for an on demand single photon source. If digital input signal recieved, output a photon in a single spatial mode.
	"""
	
	def __init__(self,  model_params):
		super(type(self), self).__init__()
		
		self.model_matrix = model_params
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def compute(self, input_vector):
		# Get values from ports

		vin = [e.value for e in input_vector]
		amps={}
		vout=[0]

		for i in range(len(vin)):
			if vin[i]['digital_input_signal']==1:
				amps['amp']=1
			elif vin[i]['digital_input_signal']==0:
				amps['amp']=0
			else:
				raise Exception(' On click source can only currently take binary input')
		vout[0]=amps
		return vout



class on_click_single_photon_source_fock(model):
	"""
	Simple model for an on demand single photon source. If digital input signal recieved, output a photon in a single spatial mode.
	"""
	
	def __init__(self,  model_params):
		super(type(self), self).__init__()
		
		self.model_matrix = model_params
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def apply_raising_operator(self, mode, global_input_state, max_occupation=2):
		'''
		Function to apply the lowering operator to the appropriate mode of the input state
		'''
		
		global_output_state=copy.deepcopy(global_input_state)
		#global_input_state={(0, 0): 0, (0, 1): 0j, (0, 2): 0j, (1, 0): 0j, (1, 1): 1, (1, 2): 0j, (2, 0): 0j, (2, 1): 0j, (2, 2): 0j}
		#For every element of the state 
		
		for key in global_input_state:
			
			key_to_change=list(key)
			key_to_change[mode]=key[mode]+1
			key_to_change=tuple(key_to_change)
	
			#Hacky and needs changed in long run. To deal with applying raising 
			#operators to the maximum mode occupation terms.
			
			if key_to_change in global_input_state:
				
				global_output_state[key_to_change]=copy.deepcopy(global_input_state[key])
				
				if key[mode]==0:
					global_output_state[key]=0
					
			else:
				pass


		return global_output_state



	def compute(self, input_vector):
		# Get values from ports, SPS should only have one input port
		vin_list = [e.value for e in input_vector]
		vin = vin_list[0]

		#Copy over relevant parts of input state to output state
		vout={}
		vout['modes']=copy.deepcopy(vin['modes'])
		output_data=[]


		#Apply raising operator to input state 
		post_source_state=self.apply_raising_operator(vin['modes'][0], vin['Global_state'])
		vout['Global_state']=post_source_state

		#Grubby data formatting so that there is an output on each port
		for i in range(len(vin_list)):
			output_data.append(vout)

		return output_data