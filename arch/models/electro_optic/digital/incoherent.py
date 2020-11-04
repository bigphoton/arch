"""
Functions and objects describing incoherent detection. This class only currently works 
digital input (incident photon or not).
"""


import numpy as np
from arch.models import model
from arch.simulations.monte_carlo import monte_carlo
import copy

class detector_basic(model):
	"""
	Model for digital single mode optical input, single digital ouput, with given efficiency.
	This model is compatible with monte carlo simulations.
	"""

	def __init__(self, efficiency, model_params):
		super(type(self), self).__init__()
		
		self.model_params=model_params
		self.efficiency=efficiency

	def update_params(self,new_params):
		self.model_params=new_params



	def compute(self,input_vector):

		#Get values from input ports
		input_data = [e.value for e in input_vector]
		output_data=copy.deepcopy(input_data)
	
		#Get detector efficiency:
		efficiency=self.efficiency


		if input_data[0]['amp']>=1:
			output_data[0]['amp'] = monte_carlo.simulate(self,[0,input_data[0]['amp']],[1-np.abs(efficiency),np.abs(efficiency)])
		elif input_data[0]['amp']==0:
			output_data[0]['amp']=0
		else:
			raise Exception('Detector block can only currently take binary input')
   
		return output_data


class basic_quantum_state_detector(model):
	"""
	Basic model of detection for an input quauntum state vector.
	Implemenets basic detector efficiency.

	Also implements the tracing out of the mode from the global state vector.
	"""

	def __init__(self, efficiency, model_params):
		super(type(self), self).__init__()
		
		self.model_params=model_params
		self.efficiency=efficiency

	def update_params(self,new_params):
		self.model_params=new_params


	def apply_lowering_operator(self, mode, global_input_state):
		'''
		Function to apply the lowering operator to the appropriate mode of the input state
		'''

		#For every element of the state 
		for key in global_input_state:
			
			# if the term being looked at is either the vacuum state for that mode

			if key[mode]==0:
				pass
			
			else:
				key_to_change=list(key)
				key_to_change[mode]=key[mode]-1
				key_to_change=tuple(key_to_change)
				
				global_input_state[key_to_change]=global_input_state[key]


		return global_input_state


	def compute(self,input_vector):

		#Get values from input ports
		input_data = [e.value for e in input_vector]
		vin = input_data[0]
		mode = vin['modes'][0]
		global_state=vin['Global_state']
		output_data={}
		port_output=[]
		
		#Apply the appropriate annihilation operator to the input state
		#post_det_state=self.apply_lowering_operator(mode, global_state)
		#print(post_det_state)
	
		#Get detector efficiency:
		#efficiency=self.efficiency

		# Write tracing out function and digital output
		'''
		if input_data[0]['amp']>=1:

			output_data[0]['amp'] = monte_carlo.simulate(self,[0,input_data[0]['amp']],[1-np.abs(efficiency),np.abs(efficiency)])
		
		elif input_data[0]['amp']==0:
		
			output_data[0]['amp']=0
		
		else:
		
			raise Exception('Detector block can only currently take binary input')

		'''

		#Grotty data restructuring so there's an output on each port
		output_data['modes']=[mode]
		output_data['Global_state']=global_state
		
		for j in range(len(vin['modes'])):
			port_output.append(copy.deepcopy(output_data))
   
		return port_output
	