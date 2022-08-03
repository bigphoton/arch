"""
Input and output ports.
"""

from enum import Enum     # Req. Python >= 3.4
import sympy.core.symbol

class kind(Enum):
	optical = 0
	photonic = 0
	digital = 1
	temperature = 2
	voltage = 3
	current = 4
	quantum = 5
	real = 10
	complex = 11
	integer = 12

# Default symbol values for each kind of port. Put None for no default.
# TODO: Would be nice to integrate this as a property of each kind as kind.default
KIND_DEFAULTS = {
		kind.quantum:[{'modes' : [], 'pos' : [], 'occ' : [], 'amps' :[]}],
		kind.optical:0.0,
		kind.digital:0, 
		kind.temperature:300.0, 
		kind.voltage:0.0,
		kind.current:0.0,
		kind.real:0.0,
		kind.complex:0.0,
		kind.integer:0}

KIND_ASSUMPTIONS = {
		kind.quantum:{'dictionary':True},
		kind.optical:{'complex':True},
		kind.digital:{'integer':True, 'nonnegative':True}, 
		kind.temperature:{'positive':True}, 
		kind.voltage:{'real':True},
		kind.current:{'real':True},
		kind.real:{'real':True},
		kind.complex:{'complex':True},
		kind.integer:{'integer':True}
		}

KIND_NORMALISERS = {
		kind.quantum: (lambda x : np.sum([x['amps'][i]*x['amps'][i].H for i in len(x['amps'])])),
		kind.optical: (lambda x : abs(x)**2),
		kind.digital: (lambda x : x),
		kind.temperature: (lambda x : x),
		kind.voltage: (lambda x : x),
		kind.current: (lambda x : x),
		kind.real: (lambda x : x),
		kind.complex: (lambda x : abs(x)**2),
		kind.integer: (lambda x : x),
		}
	
def norm(port, port_value):
	"""
	Normalise port value, for plotting, etc.
	
	port: port.var instance
	port_value: value of port
	"""
	return KIND_NORMALISERS[port.kind](port_value)

class direction(Enum):
    inp = 0
    out = 1
    buffer = 2


class var(sympy.core.symbol.Symbol):
	"""
	A version of sympy.Symbol with attached attributes.
	
	block: block to which this port was initially attached, Block
	kind: kind of port, port.kind
	direction: sense of port, port.direction
	default: default value if not set, kind-specific
	
	data: dictionary of attached data, for use by models
	"""
	def __new__(self, local_name, block=None, 
					kind=None, direction=None, default=None):
		"""
		We need to intercept __new__ rather than __init__ because Symbol uses it 
			instead of __init__.
		"""
		if block is None:
			my_name = local_name
		else:
			my_name = var.new_name(block.name, local_name)
		obj = sympy.core.symbol.Symbol.__new__(self, my_name, **KIND_ASSUMPTIONS[kind])
		obj.local_name = local_name
		obj.block = block
		obj.kind = kind
		obj.direction = direction
		obj.default = default
		obj.data = dict()
		return obj
	
	@classmethod
	def new_name(cls, block_name, local_name):
		"""
		The name is set based on the *last* block to which we are associated.
		"""
		return block_name + '.' + local_name



def print_state(state):
		
		assert type(state) == dict
		assert all({type(k) == var for k in state})
		
		# Put state kvps in a list, sort appropriately
		l = [{'kind':str(p.kind),'port name':str(p.local_name),'block name':str(p.block.name),'port':p,'value':v} for p,v in state.items()]
		
		l.sort(key=(lambda e : (e['block name'],e['kind'],e['port name'])))
		
		l = ["{:10s}:  {}".format(str(e['port']),e['value']) for e in l]
		s = ",\n ".join(l)
		s = "{"+s+"}"
		
		print(s)