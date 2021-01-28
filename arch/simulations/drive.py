"""
Functions for producing time series
"""


def constant(v):
	return lambda t : v
	
def step(v0, v1, t_step):
	return lambda t : v0 if t < t_step else v1
	
def sinusoid(amp, offset, t_period, phase):
	from math import sin, pi
	return lambda t : (amp/2)*sin(2*pi*t/t_period + phase) + offset
	
def ramp(v0, v1, t_period, phase):
	from math import pi
	return lambda t : (v1-v0)*(t%t_period)/t_period + v0