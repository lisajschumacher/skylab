from __future__ import print_function

import numpy as np
import numpy.lib.recfunctions
import matplotlib.pyplot as plt
import healpy as H
from scipy import integrate
from memory_profiler import profile

from skylab.psLLH import PointSourceLLH
from skylab.ps_model import EnergyLLH, ExtendedLLH
from skylab.ps_injector import *
from skylab.utils import rotate

from functions import setNewEdges, gauss, load_data
from plot_settings import *

_size=50
_scale=np.radians(6)
_mu_per_source=5
_inj=False
_gamma=2.
_delta_ang=np.radians(10.)

class StackExtendedSources(object):
	r"""
	Set all relevant parameters for calculation of StackingLLH with extended sources,
	using the SkyLab classes for psLLH and psInjection
	
	Attributes:
		mc : 	MonteCarlo sample as record array,
					with fields ('ra', 'dec', 'logE', 'sigma', 'sinDec', 
											 'trueRa', 'trueDec', 'trueE', 'ow')
	
		exp :	Experimental data as record array,
					with fields ('ra', 'dec', 'logE', 'sigma', 'sinDec')
		
		inject :	Signal events as record array in style of exp,
							injected into experimental data in order to
							mimic point sources
	"""
	def __init__(self, basepath="/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample", detector="IC79"):
		r"""
		### This will be updated for being able to use multiple data sets ###
		
		Constructor, filling the class with mc and exp data, 
		as well as default plot settings and livetime
		
		Parameters:
			basepath :	Where to read the data from
			
			detector : Which detector to use
			
		"""
		# Default plot settings
		set_matplotlib_defaults()
		
		# Load data
		self.mc = load_data(basepath, detector)
		self.mc = np.rec.array(self.mc)
		self.exp = load_data(basepath, detector, exp_bool=True)
		self.exp = np.rec.array(self.exp)
		
		# This has to be set automatically in future
		self.livetime=315.506*24*3600
		
		# Initialize injector
		self.inject=None

	def set_random_source_positions(self, size, scale):
		r""" 
		## Preliminary random uniform values, will be exchanged for measured UHECRs 
		and/or sources according to UHECR measurement acceptance ##
		
		Sample random source positions (dec,ra) and extensions (sigma)
		
		Parameters:
			size :	size of the random samples
			
			scale :	shift of neutrino source position in radian,
							with respect to the random UHECR position
		"""
		self.dec=np.random.uniform(-np.pi/2., np.pi/2., size=size)
		self.ra=np.random.uniform(0, np.pi*2., size=size)
		self.sigma=np.radians(np.random.uniform(scale/2., scale, size=size))

		# Find a source position some degree away from assumed source position
		# Needed for injection
		dec3=np.pi/2.-abs(np.random.normal(scale=scale))
		ra3=np.random.uniform(0, np.pi*2.)
		scaling = np.ones_like(self.dec)
		self.ra_rot, self.dec_rot = rotate(0.*scaling, np.pi/2.*scaling, 
														 ra3*scaling, dec3*scaling,
														 self.ra, self.dec)

	def inject_sources(self, gamma, mu_per_source, poisson=True):
		r"""
		Inject point sources
		
		Parameters:
			gamma : Spectral index
			
			mu_per_source : Mean number of neutrinos per source
			
			poisson : Poissonian mu or not
		
		"""
		# Inject the events
		self.injector = InjectorHandler(gamma)
		self.injector.fill(self.dec, self.mc, self.livetime)
		self.inject = self.injector.sample(self.ra_rot, mu_per_source).next()[1]
		print("Sampling done.")

	def fit_llh(self, **kwargs):
		r"""
		Initialize parameters, inject sources, initialize LLH and fit the sources
		kwargs:
			size :	number of sources to be fitted, 
							number of injected sources if inj==True
			
			scale :	deviation from fitted and injected sources
			
			inj :		True: inject sources
			
			gamma :	Spectral index
			
			mu :		Mean number of nu per source
		"""
		# Initialize parameters
		size=kwargs.pop("size", _size)
		scale=kwargs.pop("scale", _scale)
		inj=kwargs.pop("inj", _inj)
		delta_ang=kwargs.pop("delta_ang", _delta_ang)
		llh_model=kwargs.pop("llh_model", ExtendedLLH())
		
		self.set_random_source_positions(size, scale)
		
		if inj==True:
			gamma=kwargs.pop("gamma", _gamma)
			mu_per_source=kwargs.pop("mu", _mu_per_source)
			self.inject_sources(gamma, mu_per_source)
		
		# Fit UHECR sources
		print("LLH setup...")
		ps_llh_ext = PointSourceLLH(self.exp, self.mc, self.livetime, delta_ang=delta_ang, mode="box", llh_model=llh_model)
		print("Fitting sources...")
		fmin, xmin = ps_llh_ext.fit_source(self.ra, self.dec, src_sigma=self.sigma, inject=self.inject)

		print ("fmin: {}".format(fmin))
		print ("xmin: {}".format(xmin))
		
		return fmin, xmin
