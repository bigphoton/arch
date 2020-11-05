"""
Functions and objects describing incoherent detection. This class only currently works 
digital input (incident photon or not).
"""


import numpy as np
from arch.models import model
from arch.simulations.monte_carlo import monte_carlo
import copy
import math



class basic_linear_detector(model):
	"""
	model for detecting the amplitude of classical input signals
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

		#Get detector efficiency:
		efficiency=self.efficiency


   
		return input_data



class monte_carlo_single_photon_detector(model):
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


	def measure_no_photons(self, mode, max_occup, state):

		''' A method which makes projective measurements onto each fock basis states
		    and returns the probabilities of seeing, 0, 1 ,2 ... photons in this mode.

			We then monte carlo sample from this probability distribution.
		'''
		amplitude_sum=np.zeros(shape=(max_occup+1), dtype=complex)
		
		#need to sum the amplitude contributions for each possible outcome
		for no_detected in range(max_occup+1):
			
			for state_element in state:

				if state_element[mode]==no_detected:

					amplitude_sum[no_detected]=amplitude_sum[no_detected]+state[state_element]

		
		#turn the amplitude sums into probalities
		probabilities=np.around(np.abs(amplitude_sum)**2, decimals=5)
		
		if np.around(np.sum(np.abs(probabilities)),decimals=5)!=1:
			raise Exception('The sum over probabilities of all detection events is >1 for mode:', mode)

		#Monte carlo simulate an outcome given the calculate probabilities
		outcomes = [i for i in range(max_occup+1)]
		result = monte_carlo.simulate(self, outcomes, probabilities)		
		prob = probabilities[outcomes.index(result)]

		#return the outcome and its associated probability
		return result, prob


	def generate_post_meas_state(self, result, prob, mode, state):

		'''
		Method to update the global quantum state given the result of the projective 
		measurement on the Fock basis of the mode.
		'''
		
		for state_element in state:
			
			# set to 0 state elements with no_photons diff to the meas outcome
			if state_element[mode] != result:
				state[state_element]=0
			
			#Apply renormalisation
			else:
				state[state_element]=np.around(np.divide(state[state_element], math.sqrt(prob)), decimals=6)

		
		return state



	def compute(self,input_vector):

		#Get values from input ports
		input_data = [e.value for e in input_vector]
		vin = input_data[0]
		mode = vin['modes'][0]
		global_state=vin['Global_state']
		max_mode_occupation=np.max(np.array(list(global_state.keys())[-1]))
		output_data={}
		port_output=[]
		
		det_result, prob =self.measure_no_photons(mode, max_mode_occupation , global_state)

		post_meas_state=self.generate_post_meas_state(det_result, prob, mode, global_state)


		#Grotty data restructuring so there's an output on each port
		output_data['modes']=[mode]
		output_data['Global_state']=post_meas_state
		output_data['detection_result']= det_result
		
		for j in range(len(vin['modes'])):
			port_output.append(copy.deepcopy(output_data))
   
		return port_output
	