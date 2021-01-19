"""
Functions and objects describing optical components.
"""

import abc
import sympy
from ..connectivity import Connectivity
import arch.port as port


class Model(abc.ABC):
	"""
	Model base class. One of `block` or `ports` must be defined.
	
	name: name of this model for indexing, string
	block: block of which this model is part (optional)
	ports: ports connected to this model (optional)
	kwargs: keyword argument dict passed to subclass Model.define method
	"""
	
	def __init__(self, name, block=None, ports=None, **kwargs):
		self.name = name
		
		if block is not None and ports is None:
			self.ports = list(block.ports)
		elif ports is not None and block is None:
			self.ports = list(ports)
		elif ports is not None and block is not None:
			raise AttributeError("One and only one of either `block` or `ports` "
							"may be set.")
		else:
			self.ports = list()
		
		self.__properties = set()
		
		self.define(**kwargs)
	
	
	@classmethod
	def compound(cls, name, models, connectivity):
		
		print ("Compounding in Model")
		return NumericModel.compound(name, models, connectivity)
	
	
	@property
	def lineage(self):
		"""
		Return list of models in this model's chain of inheritance.
		"""
		def list_bases(c):
			if issubclass(c, Model):
				all_bases = [c]
				for base in c.__bases__:
					all_bases.extend(list_bases(base))
				return all_bases
			else:
				return []
		
		return list_bases(self.__class__)
	
	
	@property
	def properties(self):
		"""
		List of properties of model which change how the model is compounded or simulated.
		Properties (list elements) should normally be strings.
		"""
		return self.__properties
	
	
	@property
	def in_ports(self):
		return [p for p in self.ports if p.direction == port.direction.inp]
	
	@property
	def out_ports(self):
		return [p for p in self.ports if p.direction == port.direction.out]
	
	
	@abc.abstractmethod
	def define(self, **kwargs):
		"""
		Method overridden by subclasses to implement the model. kwargs are
		passed directly from __init__.
		"""
		pass
	
	
	@property
	def port_names(self):
		return {str(e) for e in self.ports}

	@property
	def default_input_state(self):
		"""Dictionary of default values keyed by input port"""
		
		return {p:p.default for p in self.in_ports}


class NumericModel(Model):
	"""
	General numeric model.
	
	out_func: function of dict keyed by input ports, returning dict keyed by output ports
	"""
	
	def define(self, out_func=None, **kwargs):
		super().define(**kwargs)
		
		self.properties.add("numeric")
		
		default_ins = self.default_input_state
		described_out_ports = set(out_func(default_ins).keys())
		
		if not set(self.out_ports).issubset(described_out_ports):
			print(self.out_ports)
			print(described_out_ports)
			raise AttributeError("Model output ports do not match ports"
						" described by out_func. "
						"Ports missing from `out_func` are {:}. ".format(
							[p for p in self.out_ports if p not in described_out_ports]) )
		
		self.out_func = out_func
	
	
	@classmethod
	def compound(cls, name, models=[], connectivity=Connectivity()):
		
		# Filter the connectivity to only cover these models
		connectivity = connectivity.filtered_by_models(models)
		
		# Get ports from models
		ports = [p for m in models for p in m.ports]
		# Filter external ports
		ex_ports = [p for p in ports if p not in connectivity]
		ex_out_ports = [p for p in ex_ports if p.direction == port.direction.out]
	
		def _have_prereqs(model, state):
			"""Does `state` contain all the prerequisite inputs for `model`"""
			return all([p in state for p in model.in_ports])
		
		def out_func(state):
			mods = set(models)
			while mods:
				ready_mods = {mod for mod in mods if _have_prereqs(mod, state)}
				for mod in ready_mods:
					state |= mod.out_func(state)
					state |= {pi:state[po] for po,pi in connectivity if po in state}
				mods -= ready_mods
			return state
		
		return NumericModel(name=name, ports=ex_ports, out_func=out_func)


class SymbolicModel(Model):
	"""
	General symbolic model.
	"""
	
	def define(self, out_exprs=dict(), **kwargs):
		super().define(**kwargs)
		
		self.properties.add("symbolic")
		self.out_exprs = out_exprs
	
	
	def out_func(self, in_state, mode='fast'):
		"""
		Function to compute output port values given input port values. It should return a
		dict keyed by the output ports.
		
		state: dict keyed by ports with target port values as dict values {port0:val0,...}
		"""
		
		assert type(in_state) is dict
		
		if hasattr(self, 'out_exprs'):
			subs = self.default_input_state | in_state
			
			opoes = self.out_exprs.items()
			
			if mode == 'fast':
				return {op:sympy.N(oe.subs(subs)) for op,oe in opoes}
				
			elif mode == 'precise':
				return {op:oe.evalf(subs=subs) for op,oe in opoes}
		
		raise NotImplementedError("Method out_func is not implemented for model of type {:}.".format(type(self)))
	
	
	@classmethod
	def compound(cls, name, models=[], connectivity=Connectivity()):
		
		# Filter the connectivity to only cover these models
		connectivity = connectivity.filtered_by_models(models)
		
		# Get ports from models
		ports = [p for m in models for p in m.ports]
		
		# Filter external ports
		ex_ports = [p for p in ports if p not in connectivity]
		ex_out_ports = [p for p in ex_ports if p.direction == port.direction.out]
		ex_in_ports = [p for p in ex_ports if p.direction == port.direction.inp]
		
		
		def _have_prereqs(model, state):
			"""Does `state` contain all the prerequisite inputs for `model`"""
			return all([p in state for p in model.in_ports])
		
		# Substitute
		state = {p:p for p in ex_in_ports}
		
		mods = set(models)
		while mods:
			ready_mods = {mod for mod in mods if _have_prereqs(mod, state)}
			for mod in ready_mods:
				state |= {op:oe.subs(state) for op,oe in mod.out_exprs.items()}
				state |= {pi:state[po] for po,pi in connectivity if po in state}
			mods -= ready_mods
		
		# Check
		extra_symbols = {s for oe in state.values() 
							for s in oe.free_symbols if s in ex_out_ports}
		if extra_symbols:
			raise AttributeError("Extra symbols found after substitution: {:}. Either "
				"relabel as compound input port, or adjust internal connectivity "
				"accordingly.".format(extra_symbols))
		
		return SymbolicModel(name=name, ports=ex_ports, out_exprs=state)



##############
## UNSORTED ##
##############


import sympy
from sympy import Matrix, sqrt, I, exp
import arch.port as port
from sympy import ImmutableMatrix, Matrix

class Linear(SymbolicModel):
	"""
	Linear optical model for classical and quantum optics.
	unitary_matrix: square sympy Matrix of dimension n; unitary or lossy unitary.
	"""
	
	def define(self, unitary_matrix=None, **kwargs):
		super().define(**kwargs)
		
		self.properties.update({"optical", "time-independent"})
		
		self.in_optical_ports = [p for p in self.in_ports if p.kind == port.kind.optical]
		self.out_optical_ports = [p for p in self.out_ports if p.kind == port.kind.optical]
		
		if unitary_matrix is None:
			unitary_matrix = sympy.eye(len(self.out_optical_ports))
		
		self.U = ImmutableMatrix(unitary_matrix)
		
		if not self.U.is_square:
			raise AttributeError("Linear model matrix (unitary_matrix) must be square.")
		
		self.n_ins = self.U.rows
		self.n_outs = self.U.rows
		
		if len(self.in_optical_ports) != self.n_ins:
			raise AttributeError(f"Number of input optical ports "
				f"{len(self.in_optical_ports)} does not match dimension of model matrix "
				f"({self.n_ins}) of model {self}:{self.name}. Add ports before adding "
				f"model. Input ports were {self.in_ports} ({self.in_optical_ports} "
				f"optical), output ports were {self.out_ports} ({self.out_optical_ports} "
				f"optical).")
		
		if len(self.out_optical_ports) != self.n_outs:
			raise AttributeError("Number of output names {:} does not match dimension of "
					"model matrix {:}. Add ports before adding model.".format(
						len(self.out_optical_ports), self.n_outs))
		
		self.out_exprs = {op:oe for op,oe in 
				zip(self.out_optical_ports, self.U * Matrix(self.in_optical_ports) ) }
	
	
	@classmethod
	def compound(cls, name, block, models, connectivity):
		
		try:
			con = connectivity
		
			if all([isinstance(m,Linear) for m in models]):
		
				if con.has_loops:
					raise NotImplementedError("Unable to hybridise models of type '{:}' "
											"containing loops".format(cls))
			
				# Put models in causal order
				models = con.order_models(models)
			
				# Map modes
				# TODO: This routine is very expensive, possible to optimise?
				modes = dict()
				np = 0
			
				# Pre-populate modes with block port order
				# TODO: Check input (block) port order is respected
				iops = [p for p in block.in_ports if p.kind == port.kind.optical]
				oops = [p for p in block.out_ports if p.kind == port.kind.optical]
				for ip,op in zip(iops, oops):
					modes[np] = {ip, op}
					np += 1
			
				# Map
				for model in models:
					for ip,op in zip(model.in_optical_ports, model.out_optical_ports):
						matched = False
						for mode,mode_ports in modes.items():
							if (ip in mode_ports) or any([con.test(xp,mp) 
									for mp in mode_ports for xp in [ip,op]]):
								# If ports connect to ports of any known mode,
								#  associate them with this mode
								mode_ports |= {ip, op}
								matched = True
								break
						if not matched:
							# If ports match no known mode, add a new mode,
							#  and associate these ports with it
							modes[np] = {ip,op}
							np += 1
			
				# Invert modes[]
				mode_of_port = {p:m for m in modes for p in modes[m]}
			
				# Initial matrix
				U = sympy.eye(np)
			
				# Accumulate model matrix
				for m in models:
					# Map old matrix rows to new ones
					mode_map = [mode_of_port[p] for p in m.in_optical_ports]
			
					U0m = m.U
					n = U0m.rows
					Um = Matrix(Matrix.diag(sympy.eye(np), U0m, sympy.eye(np - n)))
				
					# Orient matrix modes to ports
					for i,j in enumerate(mode_map):
						Um.row_swap(i+np,j)
						Um.col_swap(i+np,j)
			
					# Delete temp row/cols
					for _ in range(np):
						Um.row_del(np)
						Um.col_del(np)
				
	# 				print("Um:")
	# 				sympy.pprint(Um)
				
					# Accumulate
					U = U * Um
			
	# 			print("U:")
	# 			sympy.pprint(U)
	# 			print("ports:")
	# 			print(block.ports)
			
				return Linear(name=name, block=block, unitary_matrix=U)
		
			raise NotImplementedError("Linear unable to compound input models {:}".format(
						[m for m in models]))
		
		except NotImplementedError:
			return super().compound(name=name, block=block, models=models, connectivity=connectivity)


class LinearGroupDelay(Linear):
	"""
	Linear optical model including lumped group delay.
	"""
	
	def define(self, delay=0, **kwargs):
		super().define(**kwargs)
		
		self.properties.add("discrete-time")
		try:
			self.properties.remove("time-independent")
		except KeyError:
			pass
		
		self.delay = delay
		
		for port in self.ports:
			port.data['delay'] = self.delay
	
	
	@classmethod
	def compound(cls, name, block, models, connectivity):
		
		new_mod = Linear.compound(name, block=block,
				models=models, connectivity=connectivity)
	

class SourceModel(SymbolicModel):
	"""
	Model for sources.
	
	out_exprs: output expressions keyed by output port, dict
	"""
	
	def define(self, out_exprs, **kwargs):
		super().define(**kwargs)
		
		assert isinstance(out_exprs, dict)
		
		self.properties.add("source")
		
		self.out_optical_ports = [p for p in self.out_ports if p.kind == port.kind.optical]
		
		for p in self.out_optical_ports:
			self.out_exprs[p] = out_exprs[p]





#########
## OLD ##
#########



class model(abc.ABC):
	"""
	Model base class.
	"""
	
	@abc.abstractmethod
	def update_params(self, new_params):
		"""
		Update compact model (e.g. matrix) with new parameters, such that model.compute() gives
		result based on these new parameters.
		
		Subclasses must implement this method.
		"""
		pass


	@abc.abstractmethod
	def compute(self):
		"""
		Propagate input state to output state.
		
		Subclasses must implement this method.
		"""
		pass
	
	
	@property
	@abc.abstractmethod
	def n_inputs(self):
		"""
		Computed number of model inputs.
		
		Subclasses must implement this method.
		"""
		pass
	
	
	@property
	@abc.abstractmethod
	def n_outputs(self):
		"""
		Computed number of model outputs.
		
		Subclasses must implement this method.
		"""
		pass
		



from collections import deque


class delayed_model(object):
	"""
	Wrapper class for adding fixed delay to each input.
	
	original_model: class of original model to be instantiated
	delays: singleton int (all inputs same), or list of int number of delay time-steps
			with length same as input length.
	args, kwargs: args and kwargs for original model initialisation
	"""
	
	def __init__(self, original_model, delays, *args, **kwargs):
		
		self._model = original_model(*args, **kwargs)
		
		# Start at t=0 
		# Integer
		self.t = 0
		
		self.delays = delays
		for i in range(len(self.delays)):
			self.delays[i] += 1
			if self.delays[i] < 1:
				raise AttributeError("All elements of delays must be > 0.")
		
		# This could be expanded spectrally by making `nt` a tuple
		# Element 0 is the newest input value
		# Element -1 is the oldest input value
		self.input_time_series = [deque(maxlen=d) for d in delays]
		
	
	
	def update_params(self, new_params):
		self.model_matrix = self.unitary_matrix_func(**new_params)
	
	
	def compute(self, input_vector):
		
		# Advance the time by one step
		self.t += 1
		
		
		# Get values from ports' historical values
		# Handle each time series (for each input/output port)
		vin = []
		for i in range(self.n_modes):
			
			# Get the final input vector to multiply
			vin.append(self.time_serieses[i].pop())
		
			# Store the input vector
			self.time_serieses[i].insert(0, input_vector[i].value)
		
		m = self.model_matrix
		
		# Do matrix multiplication
		vout = m @ vin
		
		return vout.flat