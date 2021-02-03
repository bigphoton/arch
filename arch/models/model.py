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
		"""
		Method to be implemented by subclasses. Subclasses should call the `compound` method
		of `super` if they are unable to compound the input models (see snippet below).
		
		try:
			<Subclass compounding code here>
		except NotImplementedError:
			return super().compound(name=name, models=models, connectivity=connectivity)
		"""
		
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
	
	
	def __repr__(self):
		return "<"+self.__class__.__module__+"."+self.__class__.__name__+" '"+self.name+"'>"


class NumericModel(Model):
	"""
	General numeric model.
	
	out_func: function of dict keyed by input ports, returning dict keyed by output ports
	"""
	
	def define(self, out_func=None, **kwargs):
		super().define(**kwargs)
		
		self.properties.add("numeric")
		
		out = out_func(self.default_input_state).keys()
		described_out_ports = set(out_func(self.default_input_state).keys())
		
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
		print("Compounding in NumericalModel")
		
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
			# Initialise ports within loops
			loops = connectivity.loops
			state = {e:e.default for l in loops for e in l if isinstance(e, port.var)} | state
			
			# Substitute ready model values until all models are substituted
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
	
	def define(self, out_exprs=None, **kwargs):
		super().define(**kwargs)
		
		self.properties.add("symbolic")
		if out_exprs is not None:
			self.out_exprs = out_exprs
	
	
	@property
	def out_exprs(self):
		return self.__out_exprs
	
	@out_exprs.setter
	def out_exprs(self, new_out_exprs):
	
		self.__out_exprs = new_out_exprs
		
		# Refresh out_funcs
		try:
			self._out_func_lambda = sympy.lambdify(self.in_ports,
										[self.out_exprs[p] for p in self.out_ports])
		except KeyError as e:
			raise KeyError(f"Output port '{e}' not described by `out_exprs` {self.out_exprs}")
	
	
	def out_func(self, in_state):
		"""
		Compute output state from input state.
		
		in_state: dictionary of port values keyed by port
		return: dictionary of port values (including outputs) keyed by port
		"""
		
		# Since our lambda func (and sympy.lambdify) deals in arg *vectors*, derive them from the
		#  input dict, and derive the output dict from them.
		in_state_vec = [in_state[p] for p in self.in_ports]
		out_state_vec = self._out_func_lambda(*in_state_vec)
		out_state_dict = in_state | {self.out_ports[i]:out_state_vec[i] for i in range(len(out_state_vec))}
		
		return out_state_dict
		
	
	
	@classmethod
	def compound(cls, name, models=[], connectivity=Connectivity(), iter_max=10):
		
		try:
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
			i = 0
			while mods and i < iter_max:
				i += 1
				ready_mods = {mod for mod in mods if _have_prereqs(mod, state)}
				for mod in ready_mods:
					state |= {op:oe.subs(state) for op,oe in mod.out_exprs.items()}
					state |= {pi:state[po] for po,pi in connectivity if po in state}
				mods -= ready_mods
		
			# Check
			if i == iter_max:
				ls = connectivity.loops
				print("Found loops:",list(ls))
				raise NotImplementedError(
							f"Reached max iteration limit ({iter_max}) but all models do not "
							f"yet have their prerequisite inputs. Remaining models are {mods}")
			extra_symbols = {s for oe in state.values() 
								for s in oe.free_symbols if s in ex_out_ports}
			if extra_symbols:
				raise AttributeError("Extra symbols found after substitution: {:}. Either "
					"relabel as compound input port, or adjust internal connectivity "
					"accordingly.".format(extra_symbols))
		
			return SymbolicModel(name=name, ports=ex_ports, out_exprs=state)
			
		except NotImplementedError:
			return super().compound(name=name, models=models, connectivity=connectivity)



##############
## UNSORTED ##
##############


import sympy
from sympy import sqrt, I, exp
from sympy import ImmutableMatrix, Matrix
import arch.port as port

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
		
		# TODO: Should override `out_func` to use matrix multiplication for the optical ports
		self.out_exprs = {op:oe for op,oe in 
				zip(self.out_optical_ports, self.U * Matrix(self.in_optical_ports) ) }
	
	
	@classmethod
	def compound(cls, name, models, connectivity):
		
		try:
			print("Compounding in Linear")
			
			if all([isinstance(m,Linear) for m in models]):
			
				print("starting models =",models)
		
				if connectivity.has_loops:
					print("Connectivity has loops, but trying to compound in Linear anyway")
					
					loops = connectivity.loops
					
					from arch.block import Block
					
					n_loop = 0
					for n_loop,loop in enumerate(loops):
						
						print("Processing loop",' -> '.join([str(e.name) for e in loop]))
						loop_blocks = [b for b in loop if isinstance(b,Block)]
						loop_mods = [b.model for b in loop_blocks]
						
						assert all([m in models for m in loop_mods]), "All loop models not in `models`"
						
						loop_exprs = dict()
						for m in loop_mods:
							loop_exprs |= m.out_exprs
						
						# Keep loop port connections
						con_subs = dict()
						for i in range(len(loop)):
							if type(loop[i]) == type(loop[i-1]) == port.var:
								con_subs |= {loop[i]:loop[i-1]}
						
# 						print("subs=",con_subs)
						
						for o in loop_exprs:
							loop_exprs[o] = loop_exprs[o].subs(con_subs)
						
						# Make homogeneous equations to solve
						loop_vars = list(loop_exprs.keys())
						loop_vars.sort(key=lambda x: str(x))
# 						print(loop_vars)
# 						loop_eqs = [e - o for o,e in loop_exprs.items()]
						loop_eqs = [loop_exprs[o] - o for o in loop_vars]
						
						ex_ports = list(connectivity.external_ports(loop_blocks))
						ex_ports.sort(key=lambda x:str(x))
						ex_ports = [p for p in ex_ports if p.kind == port.kind.optical]
						
						# We need to solve for these
						ex_out_ports = [p for p in ex_ports if p.direction == port.direction.out]
						ex_in_ports = [p for p in ex_ports if p.direction == port.direction.inp]
						ports = list(connectivity.ports)
						ports.sort(key=lambda x:str(x))
						all_out_ports = [p for p in ports if p.direction == port.direction.out]
						
						
						
						if False:
							
							# Get linear system to solve
							from sympy import zeros
							d = len(all_out_ports)
						
							# Work out M.x = b components
							M = zeros(d)
							x = list(all_out_ports)
							b = Matrix([0]*d)
							xvars = all_out_ports
							bvars = ex_in_ports
						
							# Compute elements from eq coefficients
							for i,x in enumerate(xvars):
								for j,y in enumerate(xvars):
									M[i,j] = loop_eqs[i].coeff(y)
								for j,y in enumerate(bvars):
									b[i] -= loop_eqs[i].coeff(y) * y
						
							# Fire linear solver
							sol_set = sympy.linsolve((M, b))
							
						else:
						
							# Fire linear solver
							sol_set = sympy.linsolve(loop_eqs, all_out_ports)
						
						if type(sol_set) != sympy.FiniteSet:
							raise NotImplementedError("Cannot solve loop eqs.")
						
						# Unpack
						sols = next(iter(sol_set))
						sol_exprs = {o:e for o,e in 
											zip(all_out_ports,sols)}
						
						# Now lump loop elements into one matrix
						# Matrix is square, with dimension len(ex_in_ports)
						from sympy import zeros
						U = zeros(len(ex_in_ports))
						
						# Compute matrix elements from expr coefficients
						for no,o in enumerate(ex_out_ports):
							for ni,i in enumerate(ex_in_ports):
								U[no,ni] = sol_exprs[o].expand().coeff(i).simplify()
						
						# We now have ex_out_ports = U @ ex_in_ports for this loop
# 						print('got U =')
# 						sympy.pprint(U)
						
						loop_mod = Linear(name=f'loop {n_loop} mod', ports=ex_ports, unitary_matrix=U)
						
						# Remove loop models from `models`
						for m in loop_mods:
							models.remove(m)
						
						# Add computed `loop_mod` back to `models`
						if type(models) == list:
							models.append(loop_mod)
						elif type(models) == set:
							models.add(loop_mod)
						else:
							models += loop_mod
						
						# Track number of loops handled
						n_loop += 1
						
						print("made loop model",loop_mod)
						print("models =",models)
				
				# Put models in causal order
				models = connectivity.order_models(models)
				
				print("ordered models =",models)
				
				from IPython import embed
				embed()
				return None
				
				
				# Filter the connectivity to only cover these models
				connectivity = connectivity.filtered_by_models(models)
		
				# Get ports from models
				ports = [p for m in models for p in m.ports]
		
				# Filter external ports
				ex_ports = [p for p in ports if p not in connectivity]
				ex_out_ports = [p for p in ex_ports if p.direction == port.direction.out]
				ex_in_ports = [p for p in ex_ports if p.direction == port.direction.inp]
			
				# Map modes
				# TODO: This routine is very expensive, possible to optimise?
				modes = dict()
				np = 0
			
				# Pre-populate modes with port order
				iops = [p for p in ex_in_ports if p.kind == port.kind.optical]
				oops = [p for p in ex_out_ports if p.kind == port.kind.optical]
				
				assert len(iops) == len(oops), ("Numbers of input and output optical ports does "
												"not match")
				
				for ip,op in zip(iops, oops):
					modes[np] = {ip, op}
					np += 1
			
				# Map
				for model in models:
					for ip,op in zip(model.in_optical_ports, model.out_optical_ports):
						matched = False
						for mode,mode_ports in modes.items():
							if (ip in mode_ports) or any([connectivity.test(xp,mp) 
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
			
				return Linear(name=name, ports=ex_ports, unitary_matrix=U)
		
			raise NotImplementedError("Linear unable to compound input models {:}".format(
						[m for m in models]))
		
		except NotImplementedError:
			return super().compound(name=name, models=models, connectivity=connectivity)


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
	def compound(cls, name, models, connectivity):
		
		new_mod = Linear.compound(name,
				models=models, connectivity=connectivity)
	

class SourceModel(SymbolicModel):
	"""
	Model for sources.
	"""
	
	def define(self, **kwargs):
		super().define(**kwargs)
		
		self.properties.add("source")
		
		self.out_optical_ports = [p for p in self.out_ports if p.kind == port.kind.optical]

