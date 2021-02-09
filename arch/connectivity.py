"""
Facilities for describing and manipulating the connectivity
between blocks, ports, and models.
"""


import arch.port as port
from arch.port import var
import networkx as nx
import matplotlib.pyplot as plt

class Connectivity:
	
	def __init__(self, connections=[]):
		
		self.__conns = set()
		self.__block_graph = nx.MultiDiGraph()
		self.__port_graph = nx.MultiDiGraph()
		self.__model_graph = nx.MultiDiGraph()
		
		self.update(connections)
	
	
	@classmethod
	def parse_connection_element(cls, connection):
		"""
		Parse and normalise single element of an input connections list.
		
		Element must be an ordered iterable of length 2; the second element may be either
		a singleton or an iterable (for fan-out connections).
		"""
		try:
			if isinstance(connection[1] , str):
				conn = {(connection[0], connection[1])}
			else:
				conn = {(connection[0], c) for c in connection[1]}
		except (IndexError,TypeError) as e:
			conn = {(connection[0], connection[1])}
		
		# TODO: Check conn is valid
		
		return conn
	
	
	def update(self, connections):
		"""
		Add and normalise connections to the set of connections.
		"""
		for conn in connections:
			new_cons = Connectivity.parse_connection_element(conn)
			self.__conns.update(new_cons)
			for c in new_cons:
				self.__block_graph.add_edge(c[0].block, c[1].block, ports=c)
				
				for b in [c[0].block, c[1].block]:
					for p in b.ports:
						if p.direction == port.direction.inp:
							self.__port_graph.add_edge(p, b, ports=(p), weight=0.3)
						else:
							self.__port_graph.add_edge(b, p, ports=(p), weight=0.3)
				
				self.__port_graph.add_edge(c[0], c[1], 
									ports=c, weight=0.6)
				
				try:
					for m0 in c[0].block.models:
						for m1 in c[1].block.models:
							self.__model_graph.add_edge(m0, m1)
				except AttributeError:
					pass
	
	
	def clear(self):
		self.__conns = set()
		self.__block_graph = nx.MultiDiGraph()
		self.__port_graph = nx.MultiDiGraph()
		self.__model_graph = nx.MultiDiGraph()
		
	
	def __iter__(self):
		return iter(self.__conns)
		
	
	def __repr__(self):
		return "Connectivity("+repr(self.__conns)+")"
	
	
	def filtered(self, predicate):
		"""
		Return a Connectivity object with elements filtered by predicate.
		predicate(i,o): function of two ports, returning a boolean
		"""
		
		return Connectivity({(i,o) for i,o in self if predicate(i,o)})
	
	
	def filtered_by_blocks(self, blocks, include_external_ports=True):
		"""
		Return a Connectivity object with connections involving `blocks`.
		blocks: iterable of `Block` objects
		include_external_ports: new connectivity to include ports of blocks not in `blocks`
		"""
		if include_external_ports:
			predicate = lambda i,o: i.block in blocks or o.block in blocks
		else:
			predicate = lambda i,o: i.block in blocks and o.block in blocks
		return self.filtered(predicate)
	
	
	def filtered_by_models(self, models):
		"""
		Return a Connectivity object with connections involving any of `models`.
		models: iterable of `Model` objects
		"""
		model_ports = [p for m in models for p in m.ports]
		predicate = lambda i,o: i in model_ports and o in model_ports
		return self.filtered(predicate)
	
	
	@property
	def ports(self):
		return {p for p in self.__port_graph if type(p) == port.var}
	
	
	def external_ports(self, blocks=None):
		"""
		Get ports on surface of connectivity. If `blocks` is set, get ports on
		surface of those ports within this connectivity.
		
		blocks: collection of blocks or None (default None)
		"""
		if blocks is None:
			return {p for b in self.blocks for p in b.ports if p not in self}
		else:
			subcon = self.filtered_by_blocks(blocks, include_external_ports=False)
			return {p for b in blocks for p in b.ports if p not in subcon}
	
	
	@property
	def external_in_ports(self):
		return {p for p in self.external_ports() if p.direction == port.direction.inp}
	
	
	@property
	def external_out_ports(self):
		print("Getting external out ports")
		return {p for p in self.external_ports() if p.direction == port.direction.out}
	
	
	def test(self, port0, port1=None):
		"""
		Test whether two ports are connected. If only the first argument is provided,
		test whether that port is contained in this connectivity.
		
		port0: var
		port1: var (optional)
		"""
		if port1 is None:
			return (port0 in [p for c in self for p in c])
		else:
			return any([port0 in c and port1 in c for c in self])
	
	
	def __contains__(self, item):
		return (item in self.__conns) or any([item in c for c in self.__conns])
	
	
	@property
	def is_empty(self):
		return True if len(self.__conns) == 0 else False
	
	
	@property
	def has_loops(self):
		"""
		Boolean, indicating whether Connectivity contains loops.
		"""
		ret = not nx.algorithms.dag.is_directed_acyclic_graph(self.__block_graph)
		return ret
	
	
	@property
	def loops(self):
		return nx.algorithms.cycles.simple_cycles(self.__port_graph)
	
	
	@property
	def shortest_loop(self):
		return min(nx.algorithms.cycles.simple_cycles(self.__port_graph), key=lambda c: len(c))
	
	
	@property
	def blocks(self):
		"""
		Return set of all described blocks.
		"""
		return set(self.__block_graph)
	
	
	@property
	def block_graph(self):
		return self.__block_graph
	
	
	@property
	def port_graph(self):
		return self.__port_graph
	
	
	@property
	def models(self):
		return {b.model for b in self.blocks}
	
	
	@property
	def model_graph(self):
		return self.__model_graph
	
	
	def all_blocks_ordered(self):
		"""
		Return list of blocks in causal order.
		"""
		return nx.topological_sort(self.__block_graph)
		
	
	def order_blocks(self, blocks):
		"""
		Return list of blocks in causal order.
		blocks: list of Block objects
		"""
		l = [b for b in self.all_blocks_ordered() if b in blocks]
		for b in blocks:
			if b not in l:
				l.append(b)
		return l
	
	
	def order_models(self, models):
		"""
		Return list of models in causal order.
		models: list of Model objects
		"""
		
		G = self.__model_graph.copy()
		
		"""
		TODO:
		Instead of the below algorithm which handles the loop nodes in an ad hoc way,
		just identify each loop and snip it in the middle by deleting an edge, then order
		the result! -Josh
		"""
		
		for _ in range(100):
			
			if not nx.algorithms.dag.is_directed_acyclic_graph(G):
				loop = nx.algorithms.cycles.find_cycle(G)
				if len(loop) > 1:
					G = nx.algorithms.minors.contracted_nodes(G, loop[0], loop[1],
															self_loops=False, copy=False)
			else:
				
				try:
					l = [m for m in nx.topological_sort(G) if m in models]
				except nx.exception.NetworkXUnfeasible:
					print('nodes:',G.nodes())
					print('cycles:',list(nx.algorithms.cycles.simple_cycles(G)))
					l = []
				l.extend([m for m in self.__model_graph if m in models and m not in G.nodes()])
				l.extend([m for m in models if m not in l])
		
				return l
		
		raise TimeoutError("Too many loops in connectivity to order models")
		
		
	def replace_blocks_with_block(self, blocks_to_replace, with_block):
		"""
		Replace connectivity block nodes with new node retaining the old nodes' connections.
		
		blocks_to_replace: iterable of blocks to replace, all blocks should be in connectivity
		with_block: block to take their place
		"""
		
		# Check all blocks to replace are in self
		assert all([b in self.blocks for b in blocks_to_replace]), (f"All blocks "
								f"({blocks_to_replace}) not in con ({self.blocks}).")
		
		all_block_ports = {p for b in blocks_to_replace for p in b.ports}
		surface_ports = set(self.external_ports(blocks_to_replace))
		internal_ports = all_block_ports - surface_ports
		
		# Check all surface ports are contained in `with_block`
		assert all([p in with_block.ports for p in surface_ports])
		
		
		con = Connectivity()
		
		for op,ip in self:
			if ip in internal_ports and op in internal_ports:
				# Connection is enclosed by surface
				continue
			elif ip in surface_ports:
				# Connection enters surface
				con.update([(op,ip)])
			elif op in surface_ports:
				# Connection exits surface
				con.update([(op,ip)])
			else:
				# Connection does not touch surface
				con.update([(op,ip)])
		
		assert not con.is_empty, "It is not yet possible to replace entire connectivities."
		
		return con
		
	
	
	def draw(self, draw_ports=True):
		"""
		Draw connectivity graph.
		"""
			
		if draw_ports:
			G = self.__port_graph
			for c in self:
				for e in c:
					b = e.block
					for p in b.ports:
						if p.direction == port.direction.inp:
							if (p, b) not in G.edges():
								G.add_edge(p, b, weight=0.3, ports=(p,))
						else:
							if (b, p) not in G.edges():
								G.add_edge(b, p, weight=0.3, ports=(p,))
		else:
			G = self.__block_graph
			
		G_port_nodes = [n for n in G.nodes() if type(n) == var]
		
		
		
		is_block = lambda obj : 'Block' in [t.__name__ for t in type(obj).__mro__]
		
		G_block_nodes = [n for n in G.nodes() if is_block(n)]
		
		
		def rotate_pos(graph_pos, G):
			"""Rotate graph positions such that flow generally goes left to right"""
			centroid = sum([graph_pos[x] for x in G.nodes()])/len(G.nodes())
			total_edge_vector = sum([(graph_pos[y]-graph_pos[x]) for x,y in G.edges()])
			import numpy as np
			v = np.array(+total_edge_vector)
			v = v/np.sqrt(v.dot(v))
			R = np.array([[ v[0],  v[1] ],
						  [-v[1],  v[0] ]])
			graph_pos = {n:(R@(u - centroid)) for n,u in graph_pos.items()}
			return graph_pos
			
		def scale_pos(graph_pos, G, s):
			"""Scale graph positions about the centroid by scale factor s"""
# 			centroid = sum([graph_pos[x] for x in G.nodes()])/len(G.nodes())
			import numpy as np
			R = np.array([[s, 0],
						  [0, s]])
			graph_pos = {n:(R@(u)) for n,u in graph_pos.items()}
			return graph_pos
			
		graph_pos = nx.kamada_kawai_layout(self.__block_graph)
		graph_pos = nx.spring_layout(nx.Graph(self.__block_graph), pos=graph_pos, 
			k=0.2, iterations=100, seed=0)
		graph_pos = rotate_pos(graph_pos, self.__block_graph)
		if draw_ports:
			graph_pos = scale_pos(graph_pos, G, 3.0)
			graph_pos = nx.spring_layout(nx.Graph(G), pos=graph_pos, 
				fixed=G_block_nodes, k=0.15, iterations=100, seed=0)
		
		edge_colours = {
			port.kind.optical:	'blue',
			port.kind.digital:	'black',
			port.kind.real:		'gray',
			None:				'red'
			}
		
		def edge_colour(e):
			if type(e[0]) is var:
				t = e[0].kind
			elif type(e[1]) is var:
				t = e[1].kind
			else:
				t = G.get_edge_data(*e)[0]['ports'][0].kind
			
			try:
				return edge_colours[t]
			except KeyError:
				return edge_colours[None]
		
		def node_colour(e):
			if is_block(e):
				return 'black'
			try:
				t = e.ports[0].kind
				return edge_colours[t]
			except:
				try:
					return edge_colours[e.kind]
				except:
					return edge_colours[None]
		
		fig, ax = plt.subplots()
		
		# Drawing properties
		node_color = {n:'blue' for n in G.nodes}
		edge_color = [edge_colour(e) for e in G.edges()]
		edge_style = 'arc3,rad=+0.2'
		node_line_colour = "white"
		text_font = 'sans-serif'
		
		# Draw graph elements
		if draw_ports:
			nx.draw_networkx_nodes(G_port_nodes, graph_pos, 
						node_size=500, 
						alpha=1, 
						node_color=[node_colour(e) for e in G_port_nodes], 
						linewidths=1, 
						edgecolors=node_line_colour)
			nx.draw_networkx_nodes(G_block_nodes, graph_pos,
						node_size=1000, 
						alpha=1, 
						node_color=[node_colour(e) for e in G_block_nodes], 
						linewidths=1, 
						edgecolors=node_line_colour)
		else:
			nx.draw_networkx_nodes(G_block_nodes, graph_pos, 
						node_size=1000, 
						alpha=1, 
						node_color=[node_colour(e) for e in G_block_nodes], 
						linewidths=1, 
						edgecolors=node_line_colour)
								
		
		el = [e for e in G.edges() if e[1] in G_block_nodes]
		nx.draw_networkx_edges(G, graph_pos, 
					edgelist=el,
					edge_color=[edge_colour(e) for e in el], 
					node_size=1000, 
					connectionstyle=edge_style
					)
		el = [e for e in G.edges() if e[1] in G_port_nodes]
		nx.draw_networkx_edges(G, graph_pos, 
					edgelist=el,
					edge_color=[edge_colour(e) for e in el], 
					node_size=500, 
					connectionstyle=edge_style
					)
		
		if not draw_ports:
			edge_labs = {(n0,n1):G.get_edge_data(n0,n1)[0]['ports'] for n0,n1,m in G.edges}
			nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labs, 
				label_pos=0.5, font_size=8, font_family=text_font)
		
		
		if draw_ports:
			node_labs = dict()
			node_labs.update({n:n.local_name for n in G_port_nodes})
			node_labs.update({n:n.name for n in G_block_nodes})
		else:
			node_labs = {n:n.name for n in G.nodes}
		nx.draw_networkx_labels(G, graph_pos, labels=node_labs, font_size=8, 
			font_color='white', font_family=text_font, font_weight='bold')

		# Show graph
		plt.axis("off")
		
		from time import time
		def on_click(event, ax):
			ax.click_loc = (event.x,event.y)
		
		annotations = []
		MAX_CLICK_LENGTH = 10
		def on_release(event, ax):
			
			# Get event details
			x,y = event.xdata, event.ydata
			canvas = event.canvas			
			
			# Ensure we're on the canvas
			if x is None or y is None:
				return
			
			# Ensure left mouse button
			if event.button != 1:
				return
			
			# Ensure this is not a drag
			if abs(ax.click_loc[0]-event.x)+abs(ax.click_loc[1]-event.y) > MAX_CLICK_LENGTH:
				return
			
			# Get nearest node
			dists = [((pos[0]-x)**2 + (pos[1]-y)**2, n) for n,pos in graph_pos.items()]
			dists.sort()
			node = dists[0][1]
			xn,yn = graph_pos[node]
			
			from matplotlib.offsetbox import TextArea, AnnotationBbox
			
			# Define a 1st position to annotate (display it with a marker)
# 			ax.plot(xn, yn, ".r")

			# Annotate the 1st position with a text box ('Test 1')
			label = ""
			if type(node) == var:
				label += 'Port ' + str(node.name)
				label += '\n' + str(node.direction)
				label += '\n' + str(node.kind)
			else:
				label += 'Block ' + str(node.name)
				label += '\n' + str(node.__class__.__name__)
				label += '\n in ' + str(node.__class__.__module__)
			
			bg_col = 'goldenrod'
			text_col = 'white'
			offsetbox = TextArea(label, textprops={'color':text_col}, minimumdescent=False)
			
			import matplotlib.patheffects as pe
			ab = AnnotationBbox(offsetbox, (xn,yn),
								xybox=(-20, 40),
								xycoords='data',
								boxcoords="offset points",
								frameon=True,
								arrowprops={'shrinkA':0, 'arrowstyle':'-', 
									'facecolor':bg_col, 'edgecolor':bg_col,
									'lw':1.5, 
									'connectionstyle':"arc3,rad=0.4",
									'path_effects':[pe.Stroke(linewidth=4.5, foreground='w'), pe.Normal()]},
								bboxprops={'facecolor':bg_col, 'edgecolor':'w','lw':1.5,  'boxstyle':'round4,pad=0.6'})
			
			# Mark annotation with the node it points to, for later use
			ab.node = node
			
			try:
				annotations[-1].remove()
				# If user clicks same node again, remove annotation
				if annotations[-1].node is not node:
					ax.add_artist(ab)
					annotations.append(ab)
			except:
				ax.add_artist(ab)
				annotations.append(ab)
			
			# Update canvas
			canvas.draw()


		fig.canvas.mpl_connect('button_press_event', lambda event:on_click(event,ax))
		fig.canvas.mpl_connect('button_release_event', lambda event:on_release(event,ax))
		
		plt.show()

