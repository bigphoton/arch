import numpy as np
import itertools
import copy
try:
	import thewalrus
except:
	print("Unable to import `thewalrus`. How about a backup permanent function?")
import math
import math
from arch.models import model
from arch.simulations.monte_carlo import monte_carlo

class permanent_quantum(model):

	""" A model for calculating the action of an interferometer on a given input state. 
		The model uses the permanent method to calculate the probabilities of all possible outcomes
		of the interferometer. 
		Currently this must then be fed to the monte carlo method which will sample from the calculated
		probability distribution in order to give an output state.
	"""

	def __init__(self, unitary_matrix_func, model_params,model_choice='monte_carlo'):
		super(type(self), self).__init__()
		
		self.unitary_matrix_func = unitary_matrix_func
		self.model_matrix = self.unitary_matrix_func(**model_params)
		self.model_choice=model_choice
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)




	def create_transition_matrix(self,unitary,input_vector,output_vector, d=complex):
		""" Function to make appropriate changes to unitary so that it represents the desired transition
			from this we can then find the permanent representing the probability of this transition.
		"""
		no_photons=int(np.sum(input_vector))
		col_swapped_matrix=np.zeros(shape=(no_photons,no_photons),dtype=d)

		#If there are more or less input photons than output channels we must reshape the matrix slightly for the following to work
		#Definitely exists a more efficient way to do this

		reshaped_unitary=np.zeros(shape=(no_photons,no_photons),dtype=d)
		col_count=0
		row_count=0

		for i in range(len(input_vector)):
			for j in range(len(input_vector)):

				if (no_photons-len(input_vector))>=0:
					reshaped_unitary[i,j]=unitary[i,j]

				elif (no_photons-len(input_vector))<0:
			
					if input_vector[i]!=0 and output_vector[j]!=0:
						reshaped_unitary[row_count,col_count]=unitary[i,j]
						col_count+=1
						row_count+=1

		#Special case of matrix with only 1 photon in and out
		if len(reshaped_unitary)==1:
			return reshaped_unitary[0]


		#Make the column swaps required for the given input vector.
		col_counter=0
		for k in range(len(input_vector)):
			if input_vector[k]==0:
				continue
			else:
				for j in range(input_vector[k]):
					col_swapped_matrix[:,col_counter+j]=copy.deepcopy(reshaped_unitary[:,k])
				col_counter+=1+j


		#Make the row swaps required for a given output vector
		transition_matrix=copy.deepcopy(col_swapped_matrix)
		row_counter=0
		for p in range(len(output_vector)):
			if output_vector[p]==0:
				continue
			else:
				for r in range(output_vector[p]):
					transition_matrix[row_counter+r,:]=copy.deepcopy(col_swapped_matrix[p,:])
				row_counter+=1+r

		
		return transition_matrix




	def calculate_probabilities(self, unitary, state_vector_elements, input_amplitudes):
		"""Using the probability expression in 'Permanents in linear optical networks' Scheel 2004,
		we calculate the probability of each transition and store it in an array.
		"""
		output_amplitudes=np.zeros(shape=(len(input_amplitudes)), dtype=complex)

		#input_amplitudes=[0, 0, 0, 0, 1, 0, 0, 0, 0]


		#in the fully quantum case we need to calculate all possible contributions to the output state
		#that is we need to loop over every element in the input state with a non 0 amplitude
		# and calculate every transition probability
		#Need to sum these in some way

		#For every element of the input state vector
		
		for i in range(len(state_vector_elements)):
			input_element=state_vector_elements[i]
			#Loop over every possible outcome
			for k in range(len(state_vector_elements)):
				element=state_vector_elements[k]

				#If it has a non zero amplitude
				#only consider photon number preserving transitions as these should evaluate to 0 anyway (true?)
				if input_amplitudes[i] != 0 and np.sum(input_element)==np.sum(element): 
				
					#print('The transition being calculated is ', input_element, element )

					trans_matrix=self.create_transition_matrix(unitary, input_element, element)


					if len(trans_matrix)==1:
						output_amplitudes[i]+=(np.abs(trans_matrix[0])**2)*input_amplitudes[i]
						
					else:
						perm=thewalrus.perm(trans_matrix)
						prefactor=1
						for m in range(len(input_element)):
							prefactor=prefactor*(1/math.sqrt(math.factorial(input_element[m])))*(1/math.sqrt(math.factorial(element[m])))
						
						output_amplitudes[k]+=np.around(perm*prefactor, decimals=6)*input_amplitudes[i]
						
		
		return output_amplitudes




	def create_full_state_unitary(self,matrix, Global_state, modes_list ):
		""" 
		The unitary of the interferometer needs to be extended to act with the identity
		on modes which are not input to the interferometer
		"""
		total_mode_number=len(list(Global_state.keys())[0])
		full_state_unitary=np.identity(total_mode_number,dtype=complex)


		for k in range(len(modes_list)):
			for l in range(len(modes_list)):
				full_state_unitary[modes_list[k],modes_list[l]]=matrix[k][l]

		return full_state_unitary



	def compute(self, input_vector):

		input_data = [e.value for e in input_vector]
		vin = input_data[0]
		output_data=copy.deepcopy(vin)
		matrix=self.model_matrix
		modes_list=[]
		port_output=[]
		
		for i in range(len(input_data)):
			modes_list.append(input_data[i]['modes'][0])
		
		#create appropriate unitary to act on the global state from
		full_unitary=self.create_full_state_unitary(matrix, vin['Global_state'], modes_list)
		
		state_vector_elements=[list(key) for key in vin['Global_state']]
		input_amps=list(vin['Global_state'].values() )

	
		output_amplitudes=self.calculate_probabilities(full_unitary, state_vector_elements, input_amps)
		
		
		#print(outcomes, probabilities)
		it=0
		for key in output_data['Global_state']:
			output_data['Global_state'][key]=output_amplitudes[it]
			it+=1


			#Grotty data restructuring so theres an output for every port 
		for j in range(len(modes_list)):
			output_data['modes']=[modes_list[j]]
			port_output.append(copy.deepcopy(output_data))
			
		return port_output
		



