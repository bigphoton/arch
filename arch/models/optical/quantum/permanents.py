import numpy as np
import itertools
import copy

try:
	import thewalrus
except:
	print("Unable to import `thewalrus`. How about a backup permanent function?")
import math
from arch.models import model
from arch.simulations.monte_carlo import monte_carlo

class permanent(model):

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


	def generate_possible_outcomes(self, vin):
		""" Assumes equal number of output and input modes. Given the number of input/output modes,
			and number of incident photons, this calculates all the possible outcomes. 
			This includes outcomes that may have 0 probability.
		"""
		outcomes=[]
		
		for value in vin:
			if value <0:
				raise Exception('Negative input is unphysical')

		#generate possible outcomes, including unphysical outcomes with no number preservation
		no_photons_in=int(np.sum(vin))
		all_outcomes=itertools.product(range(no_photons_in+1), repeat=len(vin))
			
		#get rid of non number conserving outcomes
		for i in all_outcomes:
			if np.sum(i)==no_photons_in:
				outcomes.append(list(i))

		return outcomes





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




	def calculate_probabilities(self, unitary,possible_outcomes, input_vector):
		"""Using the probability expression in 'Permanents in linear optical networks' Scheel 2004,
		we calculate the probability of each transition and store it in an array.
		"""
		probabilities=[]

		for outcome in possible_outcomes:
			trans_matrix=self.create_transition_matrix(unitary, input_vector, outcome)

			if len(trans_matrix)==1:
				probabilities.append(np.abs(trans_matrix[0])**2)
				
			else:
				perm=thewalrus.perm(trans_matrix)
				
				prefactor=1
				for m in range(len(input_vector)):
					prefactor=prefactor*(1/math.sqrt(math.factorial(input_vector[m])))*(1/math.sqrt(math.factorial(outcome[m])))
				
				probabilities.append(np.around(np.abs((perm*prefactor)**2),decimals=6))
	  
		return probabilities



	def compute(self, input_vector):

		input_data = [e.value for e in input_vector]
		output_data=copy.deepcopy(input_data)
		vin=[]
		matrix=self.model_matrix
	

		for inp in input_data:
			vin.append(inp['amp'])

		#if no input photons
		if np.sum(vin)==0:
			vout=np.zeros(len(vin))
		else:
				#Calculate all outcomes and their respective probabilities
			outcomes=self.generate_possible_outcomes(vin)
			probabilities=self.calculate_probabilities(matrix, outcomes, vin)
			vout=monte_carlo.simulate(self,outcomes,probabilities)

			#arrange the output dictionary 
			for i in range(len(output_data)):
				output_data[i]['amp']=vout[i]
			return output_data

