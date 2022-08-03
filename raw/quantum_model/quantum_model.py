"""
Description of time-frequency optical quantum state description
"""

"""
A{n,i}-modes comprise an almost-complete set of quasi-monochromatic temporal
modes used to discretise time (n) and space (i), with frequencies in the
neighbourhood of some omega central carrier frequency. They obey the time-shift property

Bni-modes are the Fourier transforms of the Ani modes, where:
	B{n,i} = B{0,i}*exp(i*2*pi*n/omega)
"""

"""
We choose the temporal probability density of the Ani to be Gaussian, meaning
that the Bni are also Gaussian.
"""

"""
Following the Gaussian function expansion (Gabor decomposition) presented by
de Wolf, in "Gaussian decomposition of beams and other functions", J Appl Phys 
65, 5166–5169 (1989).
"""

import numpy as np


TIME_DOMAIN = 'T'
FREQUENCY_DOMAIN = 'F'


class ChronocyclicState(object):
	
	"""
	func: function of one variable, centred on t=0, to be discretised
	dt: discretisation pitch/period
	sigma: FWHM of discretising Gaussians
	tc: time corresponding to centroid of func: func(0) = f(tc)
	fc: centre frequency (quasi-monochromatic)
	nonzero_width_t: width in z after which to stop searching for nonzero func(z)
	zero_threshold: threshold below which func(z) will be treated as zero
	"""
	
	def __init__(self, func_t=None, func_f=None, disc_t=None, disc_f=None, dt=None, df=None, sigma_t=None, sigma_f=None, tc=0, fc=100E12, nonzero_width_t=1E-9, zero_threshold=1E-6):
		
		self.nonzero_width_t = nonzero_width_t
		self.zero_threshold = zero_threshold
		
		self.dt = dt
		self.sigma_t = dt if sigma_t == None else sigma_t
		
		self.df = df if df != None else 1./nonzero_width_t
		self.sigma_f = sigma_f if sigma_f != None else self.df
		
		self.tc = tc
		self.fc = fc
		self.nc = round(self.fc/self.df)
		
		# Populate the input data. Order matters, last takes precedence
		if disc_f != None: self.disc_f = disc_f
		if disc_t != None: self.disc_t = disc_t
		if func_f != None: self.func_f = func_f
		if func_t != None: self.func_t = func_t
		
		if func_f == func_t == disc_f == disc_t == None:
			raise RuntimeError("At least one of time or frequency data must be provided to __init__.")
	
	
		
	
	
	@classmethod
	def gabor(cls, forigs, dz, sigma, mode=1):
		"""
		Compute Gabor decomposition coefficients to match vector of equally
		spaced samples of the original function, forigs.
		
		Discretise/sample a function using the Gabor decomposition with Gaussian
		basis functions. Based on "Gaussian decomposition of beams and other
		functions", DA de Wolf, J. Appl. Phys. 65 (12), 1989.
		"""
		
		# From eq. 14 of de Wolf 1989
		A = 1/np.sqrt(2*np.pi) * (dz/sigma)
		
		if mode == 0:
			"""Original algorithm, ignoring higher-order terms"""
			disc = A * forigs
		elif mode == 1:
			"""Use second derivative to improve transient response"""
			d2fdz = (1./dz**2) * np.convolve(forigs,[1,-2,1],'same')
			disc = A * (forigs + (-sigma**2/2)* d2fdz)
		else:
			raise AttributeError(f"Unknown mode '{mode}'.")
		
		return disc
	
	
	def discretise_t(self):
		"""
		Discretise continuous time function into Gaussian pulses.
		"""
		
		dt = self.dt
		sigma = self.sigma_t
		self.mc = round(self.tc/self.dt)
	
		half_width_t = self.nonzero_width_t/2
		
		t0s = np.arange(-half_width_t, +half_width_t, dt) # Centred on 0
		t1s = t0s + self.tc # Centred on tc
		self.t0 = t1s[0]
		self.m0 = 0
		self.mc = round(half_width_t/self.dt)
		ms = np.arange(t0s.shape[0])
	
		self.disc_t__ = self.gabor(self.func_t(t0s), dt, sigma, mode=1)
		
		self.ts__ = t0s
		self.ms__ = ms
		
		
		
	def funcapprox_tt(self, t):
		F = np.vectorize(lambda t: np.sum(self.base_t(t,self.ms)), otypes=[complex])
		return F(t)
	
	
	def base_t(self, t,m):
		return self.disc_t[m] * np.exp(-(t - self.ts[m])**2 / (2*self.sigma_t**2))
	
	
	def base_f(self, f,n):
		return self.disc_f[n] * np.exp(-(f - self.fs[n])**2 / (2*self.sigma_f**2))
	
	
    # we appear to be missing a method to go from func_t <-> func_f (a fourier transform duh)
    

	def base_tf(self, f,m):
		"""
		Fourier transorm of time basis function into frequency.
		"""
		
		pi = np.pi
		sigma = self.sigma_t
		dt = self.dt
		
		return self.disc_t[m] * np.sqrt(2*pi) * sigma \
				* np.exp(-((2*pi*sigma*f)**2) / 2) \
				* np.exp(-2*pi*1j*self.ts[m]*f) \
	
	def base_ft(self, t, n):
		"""
		Fourier transorm of frequency basis function into time.
		Based on the transform of a Gaussian (e.g. https://mathworld.wolfram.com/FourierTransformGaussian.html)
		keeping track of the exponential parameters, and keeping the freq shift
		of each component.
		"""
		
		pi = np.pi
		sigma = self.sigma_f
		df = self.df
		
		return self.disc_f[n] * np.sqrt(2*pi) * sigma \
				* np.exp(-((2*pi*sigma*t)**2) / 2) \
				* np.exp(+2*pi*1j*self.fs[n]*t) \
	
	def funcapprox_tf(self, f):
		"""
		Get frequency-domain version of discretised time func
		"""
		# FIXME: This is horrible, but it lets us broadcast
		F = np.vectorize(lambda v: sum(self.base_tf(v, self.ms)), otypes=[complex])
		return F(f)
	
	def funcapprox_ft(self, t):
		"""
		Get time-domain version of discretised freq func
		"""
		# FIXME: This is horrible, but it lets us broadcast
		f = np.vectorize(lambda s: sum(self.base_ft(s, self.ns)), otypes=[complex])
		return f(t)
		
	
	def discretise_f(self):
		"""
		Take time-discretised state, transform it, and re-discretise in frequency.
		"""
		
		df = self.df
		dt = self.dt
		sigma_f = self.sigma_f
	
		half_width_f = 1./(2*dt)
		
		f0s = np.arange(-half_width_f, +half_width_f, df) # Centred on 0
		f1s = f0s + self.fc # Centred on fc
		ns = np.arange(f0s.shape[0])
		
		self.f0 = f0s[0]
		self.n0 = 0
		self.nc = round(half_width_f/self.df)
	
		self.disc_f__ = self.gabor(self.func_f(f0s), df, sigma_f, mode=1)
		
		self.fs__ = f0s
		self.ns__ = ns
		
		
	def funcapprox_ff(self, f):
		F = np.vectorize(lambda f: np.sum(self.base_f(f,self.ns)), otypes=[complex])
		return F(f)
	
	
	def integral_t(self):
		"""
		Integral over all time using Gaussian integral and discretisation.
		"""
		
		return self.sigma_t * np.sqrt(2*np.pi) * sum(self.disc_t)
	
	
	def integral_f(self):
		"""
		Integral over all frequency.
		"""
		
		return self.sigma_f * np.sqrt(2*np.pi) * sum(self.disc_f)
	
	
	def integral_t2(self):
		"""
		Abs-square integral over all time using Gaussian integral and discretisation.
		"""
		
		# TODO: Numpyise this
		s = 0
		for m0 in self.ms:
			for m1 in self.ms:
				s += self.disc_t[m0] * np.conj(self.disc_t[m1]) \
						* np.exp(-self.dt**2 * (m0-m1)**2
											/ (4*self.sigma_t**2) )    # JCA note: why different to time d. expression?
		
# 		return (self.sigma_t*np.sqrt(2*np.pi))**2 * s
		return self.sigma_t*np.sqrt(np.pi) * s
	
	
	def integral_f2(self):
		"""
		Abs-square integral over all frequency using Gaussian integral and discretisation.
		"""
		
		# TODO: Numpyise this
		s = 0
		for n0 in self.ns:
			for n1 in self.ns:
				s += self.disc_f[n0] * np.conj(self.disc_f[n1])
		
		return (self.sigma_f*np.sqrt(2*np.pi))**2 * s
		
		
		
	
	@property
	def ts(self):
		return self.ts__
	
	@property
	def ms(self):
		return self.ms__
		
	
	
	@property
	def fs(self):
		return self.fs__
	
	@property
	def ns(self):
		return self.ns__
	
	@property
	def dominant_domain(self):
		"""
		The domain with the most or most recent input information {'t','f'}.
		"""
		return self.dominant_domain__
	
	
	@property
	def func_t(self):
		return self.func_t__
	
	@func_t.setter
	def func_t(self, new):
		self.func_t__ = np.vectorize(new, otypes=[complex])
		self.dominant_domain__ = TIME_DOMAIN
		self.discretise_t()
		self.func_f__ = self.funcapprox_tf
		self.discretise_f()
	
	
	@property
	def func_f(self):
		return self.func_f__
	
	@func_f.setter
	def func_f(self, new):
		self.func_f__ = np.vectorize(new, otypes=[complex])
		self.dominant_domain__ = FREQUENCY_DOMAIN
		self.discretise_f()
		self.func_t__ = self.funcapprox_ft
		self.discretise_t()
		
	
	
	@property
	def disc_t(self):
		return self.disc_t__
	
	@disc_t.setter
	def disc_t(self, new):
		self.disc_t__ = new
		self.dominant_domain__ = TIME_DOMAIN
		self.func_t__ = self.funcapprox_tt
		self.func_f__ = self.funcapprox_tf
		self.discretise_f()
	
	
	@property
	def disc_f(self):
		return self.disc_f__
	
	@disc_f.setter
	def disc_f(self, new):
		self.disc_f__ = new
		self.dominant_domain__ = FREQUENCY_DOMAIN
		self.func_f__ = self.funcapprox_ff
		self.func_t__ = self.funcapprox_ft
		self.discretise_t()






def test_integrals():
	it2s = []
	it2_true = 1
	W = 1.0
	for dt in np.arange(W/100,W/10+W/100,W/100):
		def forig_test(x):
			if abs(x) < W/2.:
				return np.sqrt(1./W)
			else:
				return 0

		st = ChronocyclicState(func_t=forig_test, tc=0, dt=dt, df=0.2, nonzero_width_t=2*W)

		it2s.append((W,dt,it2_true - np.abs(st.integral_t())))
	
	it2s = np.array(it2s)
	print(it2s)
	return it2s

# test_integrals(); quit()



if __name__=="__main__":

    def forig(x):
        # if x < 0 and -1 <= x:
            # return 1.0
        if -1 <= x and x < 1:
            return 1.0 - x**4   # lets try a complex function and see if everything holds!
        else:
            return 0

    st = ChronocyclicState(func_t=forig, tc=0, dt=0.1, df=0.1, nonzero_width_t=4.0)

    print(f"Initial: t={st.integral_t()}, t2={st.integral_t2()}")
    print(f"Initial: f={st.integral_f()}, f2={st.integral_f2()}")
      # abs. sqaure not equal


    # test transforms & approximations

    from matplotlib import pyplot as plt

    plot_f1 = np.real
    plot_f2 = np.imag

    # in time domain
    plt.subplot(5,1,1)

    xs = np.linspace(min(st.ts), max(st.ts), 2000)
    plt.plot(xs, plot_f1(st.func_t(xs)), 'k-')    # ground truth wave
    # plt.plot(xs, plot_f2(st.func_t(xs)), 'k-')    # ground truth wave
    plt.plot(xs, plot_f1([st.funcapprox_tt(x) for x in xs]), 'b-')     # real part of approximation 
    # plt.plot(xs, plot_f2([st.funcapprox_tt(x) for x in xs]), 'b-')     # imag part of approximation 
    for m in st.ms:
        plt.plot(xs, plot_f1(st.base_t(xs,m))-0.5, '-')    # real part of approximation components
        # plt.plot(xs, plot_f2(st.base_t(xs,m))-1.2, '-')    # imag part of approximation components





    # in freq. domain
    plt.subplot(5,1,2)

    fs = np.linspace(-10,10,500)
    plt.plot(fs, plot_f1(st.funcapprox_tf(fs)) )     # imag part of func
    # plt.plot(fs, plot_f2(st.funcapprox_tf(fs)) )     # imag part of func

    plt.subplot(5,1,3)

    plt.plot(fs, plot_f1(st.funcapprox_ff(fs)), '-')     # real part of approximation
    # plt.plot(fs, plot_f2(st.funcapprox_ff(fs)), '-')     # imag part of approximation 
    for n in st.ns:
        plt.plot(fs, plot_f1(st.base_f(fs,n))-0.7, '-')     # real part of approximation components
        # plt.plot(fs, plot_f2(st.base_f(fs,n))-1.5, '-')     # imag part of approximation components

    # Compare the original function
    # plt.plot(xs, plot_f1(st.func_t(xs)), '-')
    # ...with the Gabor-discretised one
    # plt.plot(xs, plot_f1(st.funcapprox_tt(xs)), '-')
    # ...and with the twice-Fourier-transformed Gabor-discretised one
    # plt.plot(xs, plot_f1(st.funcapprox_ft(xs)), '-')

    plt.subplot(5,1,4)




    ## band limited with heaviside functions
    w = 6
    mask_f = np.heaviside(st.ns-(st.nc-w), 0) * np.heaviside((st.nc+w)-st.ns, 0)
    st.disc_f = st.disc_f * mask_f
    plt.plot(fs, plot_f1(st.func_f(fs)), '-')
    # plt.plot(fs, plot_f2(st.func_f(fs)), '-')
    for n in st.ns:
        plt.plot(fs, plot_f1(st.base_f(fs,n))-0.7, '-')
        # plt.plot(fs, plot_f2(st.base_f(fs,n))-1.5, '-')
        
    #  in time domain
    plt.subplot(5,1,5)
    plt.plot(xs, plot_f1(st.func_t(xs)), '-')     # real part of summed approximation 
    # plt.plot(xs, plot_f2(st.func_t(xs)), '-')     # iamg part of summed approximation
    plt.plot(xs, plot_f1([forig(x) for x in xs]), '-')     # real part of orig 
    # plt.plot(xs, plot_f2([forig(x) for x in xs]), '-')     # imag part of orig 
    for m in st.ms:
        plt.plot(xs, plot_f1(st.base_t(xs,m))-0.5, '-')     # real part of approximation components
        # plt.plot(xs, plot_f2(st.base_t(xs,m))-1.2, '-')     # imag part of approximation components



    plt.show()