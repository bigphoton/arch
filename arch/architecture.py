



class Architecture(object):
	
	def __init__(self, blocks=None, connections=None, simulation=None):
		
		self.__blocks = blocks
		self.__connections = connections
		self.__simulations = [simulation]
	
	
	@property
	def connections(self):
		return self.__connections
	
	@connections.setter
	def connections(self, new):
		self.__connections = new
	
	
	@property
	def simulations(self):
		return self.__simulations
	
	@connections.setter
	def simulations(self, new):
		self.__simulations = new
	
	
	@property
	def blocks(self):
		return self.__blocks
	
	
	def add_connections(self, connections):
		self.__connections.append(connections)
	
	
	def add_simulation(self, simulation):
		self.__simulations.append(simulation)
