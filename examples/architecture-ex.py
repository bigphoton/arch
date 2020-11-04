"""
Example architecture global.
"""

# Add the parent folder (outside of `examples`) to the path, so we can find the `arch` module
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

from arch.architecture import architecture
from arch.blocks import electro_optics
from arch.blocks import logic
from arch.blocks import optics

from numpy import linspace, pi



import networkx as nx
import matplotlib.pyplot as plt



class architecture(object):
	"""
	Top-level class containing architectural elements.
	"""
	
	def __init__(self, blocks=[]):
		
		self.blocks = blocks
	
	
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
		
		graph_pos = nx.drawing.layout.planar_layout(G)
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
	
	
	def compute(self):
		"""
		Compute all blocks and place result on output ports.
		"""
		
		# TODO: Work out the best order to compute blocks in
		for b in self.blocks:
			b.compute()
	
	


from time import sleep

g = logic.not_gate()
bs0 = electro_optics.switch_2x2()
ps = optics.phase_shift(phase=0)
bs1 = optics.beamsplitter()
bs2 = optics.beamsplitter()

g.ports['OUT'].connect(bs0.ports['DIG'])
bs0.ports['OUT0'].connect(ps.ports['IN'])
ps.ports['OUT'].connect(bs1.ports['IN0'])
bs0.ports['OUT1'].connect(bs1.ports['IN1'])
bs1.ports['OUT0'].connect(bs2.ports['IN0'])
bs1.ports['OUT1'].connect(bs2.ports['IN1'])

g.ports['IN'].value = 1
bs0.ports['IN0'].value = 1.0
bs0.ports['IN1'].value = 0.0

# We can use the old drawing functionality by calling .draw() manually
if False:
	for b in [g, bs0, ps, bs1, bs2]:
		b.graphic.draw()
		for p in b.ports:
			p.graphic.draw()

arch = architecture(blocks=[g, bs0, ps, bs1, bs2])


for phase in linspace(0,pi,10):
	ps.phase = phase
	arch.compute()
	print("phase={:.3f}, output={:}".format(phase, bs2.ports['OUT0'].value) )


arch.draw()