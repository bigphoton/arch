

import networkx as nx
import matplotlib.pyplot as plt
import itertools


class architecture(object):
	"""
	Top-level class containing architectural elements.
	
	Global model - declare either 'linear', 'monte_carlo', 'full_quantum'

	blocks: list of blocks to initialise with

	
	"""
	
	def __init__(self, global_model, blocks=[]):
		
		self.blocks = blocks
		self.global_model=global_model
	
	
	@property
	def graph(self):
		"""
		Update block-port graph if required.
		Return Networkx directed multigraph object.
		"""
		
		G = nx.MultiDiGraph()
		
		def edge(port1, port2=None):
			# Convenience function for getting port owner names in a tuple
			p1 = port1.owner.reference_designator if port1 is not None else port2.name
			p2 = port2.owner.reference_designator if port2 is not None else port1.name
			return (p1, p2)
		
		# Collect all the port connections between blocks
		# Store the blocks and ports in node and edge attribute data
		for b in self.blocks:
			G.add_node(b.reference_designator, block=b)
			for p in b.ports:
				if p.is_output and len(p.connected_ports):
					for cp in p.connected_ports:
						G.add_edge(*edge(p, cp), port=p)
		
		return G
	
	
	def draw(self):
		"""
		Draw all blocks, connections, input and output ports.
		"""
		
		G = self.graph
		
		graph_pos = nx.planar_layout(G)
		graph_pos = nx.spring_layout(G, pos=graph_pos, iterations=10000, threshold=1E-6)
		
		def edge_colour(e):
			d = G.get_edge_data(*e)
			t = d[0]['port'].type
			if t == 'optical':
				return 'blue'
			elif t == 'digital':
				return 'black'
			elif t == 'electrical':
				return 'gray'
			else:
				return 'red'
		
		def node_colour(e):
			d = G.get_edge_data(*e)
			t = d[0]['port'].type
			if d[0]['type'] in ['input', 'output']:
				return 'green'
			elif t == 'optical':
				return 'blue'
			elif t == 'digital':
				return 'black'
			elif t == 'electrical':
				return 'gray'
			else:
				return 'red'
		
		# Drawing properties
		node_size = 1000
		node_color = 'blue'
		node_text_size = 8
		edge_color = [edge_colour(e) for e in G.edges()]
		edge_alpha = 0.5
		edge_thickness = 1
		edge_text_pos = 0.5
		arrow_size = 30
		edge_style = 'arc3,rad=+0.1'
		node_line_colour = "white"
		text_font = 'sans-serif'
		
		# Draw graph elements
		nx.draw_networkx_nodes(G, graph_pos, node_size=node_size, 
								alpha=1, node_color=node_color, linewidths=1, 
								edgecolors=node_line_colour)
							   
		nx.draw_networkx_edges(G, graph_pos, width=edge_thickness, alpha=edge_alpha,
				 edge_color=edge_color, node_size=node_size+arrow_size, arrowstyle="simple", arrowsize=arrow_size, 
				 connectionstyle=edge_style)
							   
		nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size, font_color='white',
								font_family=text_font, font_weight='bold')

# 		labels = range(len(graph))
# 		edge_labels = dict(zip(graph, labels))
# 		nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
# 										label_pos=edge_text_pos,font_size=node_text_size,
# 										font_family=text_font)

		# Show graph
		plt.axis("off")
		plt.show()
	

	#Create a dictionary where keys are the vectors and values are the coefficients
	def create_input_state(self, max_mode_occupation):
		input_state={}
		no_modes=0
		
		for b in self.blocks:
			if b.reference_prefix=='PPS':
				b.ports['IN'].value['modes']=[no_modes,no_modes+1]
				no_modes+=2
	
			elif b.reference_prefix=='SPS':
				b.ports['IN'].value['modes']=[no_modes]
				no_modes+=1
			else:
				continue

		states=itertools.product([i for i in range(max_mode_occupation+1)], repeat=int(no_modes))
		state_list=[tuple(p) for p in states]


		vacuum=True
		for state in state_list:
			if vacuum:
				input_state[state] = 1
			else:
				input_state[state] = 0

			vacuum=False

		return input_state

	
	def compute(self):
		"""
		Compute all blocks and place result on output ports.
		"""
		
		if self.global_model=='linear':

			for b in self.blocks:
				b.compute()


		elif self.global_model=='monte_carlo':

			for b in self.blocks:
				b.compute()

		

		elif self.global_model=='full_quantum':
		
			state=self.create_input_state(max_mode_occupation=2)
			print(' \n Initial Global state is:', state)

			# TODO: Work out the best order to compute blocks in
			for b in self.blocks:
				
				for p in b.ports:
					if p.is_input:
						p.value['Global_state']=state
				

			
				b.compute()

				for p in b.ports:
					if p.is_output:
						state=p.value['Global_state']
				
				print(' \n \n state after block', b.reference_prefix, state )
				
			

				