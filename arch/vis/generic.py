"""
Generic graphics for generic objects.
"""

from .graphic import graphic


class generic_box(graphic):
	"""
	Graphics subclass for drawing generic blocks for devices which don't draw their own.
	
	name: label to draw at centre
	n_in: number of input ports (on left side)
	n_out: number of output ports (on right side)
	"""
	
	box_width = 60
	box_height = 120
	box_linewidth = 2
	
	
	def __init__(self, name="", **kwargs):
	
		graphic.__init__(self, **kwargs)
		
		self.name = name
		
		self.graphical_attributes.extend(["name"])
		
		self.update_path()
	
	
	def update_path(self):
		"""
		Update the graphics path with latest parameters (name, n_in, n_out...)
		"""
		
		w = generic_box.box_width
		h = generic_box.box_height
	
		p = []
	
		p.extend(['wi',generic_box.box_linewidth])
	
		# Draw box
		p.extend(['pu',None])
		p.extend(['go',(+w/2,+h/2)])
		p.extend(['pd',None])
	
		p.extend(['go',(+w/2,-h/2)])
		p.extend(['go',(-w/2,-h/2)])
		p.extend(['go',(-w/2,+h/2)])
		p.extend(['go',(+w/2,+h/2)])
		
		# Draw name text
		p.extend(['pu',None])
		p.extend(['go',(0,-6)])
		p.extend(['tx',self.name])
		
		self.path = p


class generic_port(graphic):
	"""
	Graphic for drawing input/output ports.
	
	name: str, label string
	in_port: bool, port direction
	"""
	
	port_length = 20
	
	def __init__(self, name="", is_input=True, **kwargs):
	
		graphic.__init__(self, **kwargs)
		
		self.name = name
		self.is_input = is_input
		
		self.connected_port = None
		
		self.graphical_attributes.extend(["name", "is_input", "port_length"])
		
		self.update_path()
	
	
	def update_path(self):
	
		p = []
	
		p.extend(['pu',None])
		p.extend(['wi',generic_box.box_linewidth])
		
		p.extend(['lt',self.angle+90])
		p.extend(['fd',5])
		p.extend(['tx',self.name])
		p.extend(['bk',5])
		p.extend(['oa',0])
		p.extend(['pd',None])
		
		
		if self.is_input:
			p.extend(['go',(generic_port.port_length,0)])
			p.extend(['arb','stamp'])
			p.extend(['pu',None])

		else:
			p.extend(['rt',180])
			p.extend(['arb','stamp'])
			p.extend(['go',(generic_port.port_length,0)])
			p.extend(['pu',None])
			
		if self.is_input:
			try:
				u = self.transform_to_local(self.position)
				v = self.transform_to_local(self.connected_port.graphic.position)
				p.extend(['wi',1])
				p.extend(['cl','blue'])
				p.extend(['pu',None])
				p.extend(['go',u])
				p.extend(['pd',None])
				p.extend(['go',v])
			except:
				pass
		
		self.path = p

