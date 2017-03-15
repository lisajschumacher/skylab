#!/bin/env python
from __future__ import print_function

import os
import numpy as np
import numpy.lib.recfunctions
import matplotlib.pyplot as plt
import healpy as H
from scipy import integrate
from memory_profiler import profile

from skylab.psLLH import PointSourceLLH, MultiPointSourceLLH
from skylab.ps_model import EnergyLLH, ExtendedLLH
from skylab.ps_injector import *
from skylab.utils import rotate

from functions import setNewEdges, gauss, load_data, prepareDirectory
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
	def __init__(self, basepath, detector, e_thresh=0., D=np.radians(6.), random_uhecr=False, **kwargs):
		r"""
		Constructor, filling the class with mc and exp data, 
		as well as default settings and livetime
		
		Parameters:
			basepath :	string
									Where to read the data from
			
			detector : 	string
									Which detector to use

			e_thresh : 	int
									energy threshold for UHECR selection

			D : 				float
									Magnetic deflection parameter, usually radian(3 - 9 degrees)
									
			random_uhecr : 	bool
											do or do not use random uhecr positions

		Optional parameters:
			size : 	int, number of random uhecr events

			sets : 	list of strings

			kwargs for PointSourceLLH class:
				llh_model : ExtendedLLh() or EnergyLLH(), if none given, ClassicLLH() will be used
				mode : "all", "box", "band"; used for event selection. Default is "all"
		"""
		# kwargs for random source position, if random_uhecr == True
		size = kwargs.pop("size", _size)

		# kwargs for uhecr
		sets = kwargs.pop("sets", ["auger", "ta"])

		#~ # kwargs for source injection, if inj == True
		#~ gamma=kwargs.pop("gamma", _gamma)
		#~ mu_per_source=kwargs.pop("mu", _mu_per_source)
		
		# Default plot settings
		#~ set_matplotlib_defaults()
		
		# Load data
		self.mc, self.exp, self.livetime = load_data(basepath, detector)
		self.mc = np.rec.array(self.mc)
		self.exp = np.rec.array(self.exp)	
		
		# Initialize injector
		self.inject = None

		# Initialize source positions
		if random_uhecr:
			self.set_random_source_positions(size=size, scale=D)
		else:
			self.set_UHECR_positions(sets, D, e_thresh)

		# Injection		
		#~ if inj==True:			
			#~ self.inject_sources(gamma, mu_per_source)
		
		# Initialize PS LLH
		print("LLH setup...")
		self.ps_llh = PointSourceLLH(self.exp, self.mc, self.livetime, **kwargs)

	def set_random_source_positions(self, size, scale, inj=False):
		r""" 
		## Preliminary random uniform values 
		(in future implementation:
			sources according to UHECR measurement acceptance)##
		
		Sample random source positions (dec,ra) and extensions (sigma)
		
		Parameters:
			size :	size of the random samples
			
			scale :	shift of neutrino source position in radian,
							with respect to the random UHECR position
							- equivalent to source extent in set_UHECR_position()
							(kind of)
		"""
		self.dec=np.random.uniform(-np.pi/2., np.pi/2., size=size)
		self.ra=np.random.uniform(0, np.pi*2., size=size)
		self.sigma=np.radians(np.random.uniform(scale/2., scale, size=size))
		if inj:
			self.set_injection_position()

	def set_injection_position(self):
		"""
		Find a source position some degree away from assumed source position given by self.dec/ra.
		Deviation chosen using self.sigma set in set_UHECR_positions/random_source_positions.
		Needed for injection of sources governed by PointSourceLLH.
		"""
		
		dec3=np.pi/2.-abs(np.random.normal(scale=self.sigma))
		ra3=np.random.uniform(0, np.pi*2.)
		scaling = np.ones_like(self.dec)
		self.ra_rot, self.dec_rot = rotate(0.*scaling, np.pi/2.*scaling, 
														 ra3*scaling, dec3*scaling,
														 self.ra, self.dec)
														 
	def set_UHECR_positions(self, sets, D, e_thresh, inj=False, **kwargs):
		"""
		Choose the data set(s) and read the text file(s)
		Parameters:
			sets : ["auger", "ta"] or one of them
			
			D : float
					parameter for source extent, called "D" in paper
					usually 3 or 6 degree (but plz give in radian k?)

			e_thresh : float
								 threshold value for energy given in EeV
		"""

		if kwargs:
			for attr, value in kwargs.iteritems():
				print(("Unknown attribute '{:s}' (value {:s}, skipping...").format(attr, value)) 
		
		path = "/home/home2/institut_3b/lschumacher/phd_stuff/phd_stuff_git/phd_code/CRdata"
		files_dict = {"auger" : {"f" : "AugerUHECR2014.txt", "data" : None}, "ta" : {"f" : "TelArrayUHECR.txt", "data" : None}}
		dec_temp = []
		ra_temp = []
		e_temp = []
		print("Setting true UHECR positions...")
		for k in sets:
			files_dict[k]["data"] = np.genfromtxt(os.path.join(path, files_dict[k]["f"]), names=True)
			dec_temp.extend(np.radians(files_dict[k]["data"]['dec']))
			ra_temp.extend(np.radians(files_dict[k]["data"]['RA']))
			if k=="ta":
				# Implement energy shift of 13%, see paper
				e_temp.extend(files_dict[k]["data"]['E']*(1.-0.13))
			else:
				e_temp.extend(files_dict[k]["data"]['E'])

		# Check the threshold
		if e_thresh > max(e_temp):
			default = 85.
			print("Energy threshold {:1.2f} too large, max value {:1.2f}! Set threshold to {:1.2f} EeV".format(e_thresh, max(e_temp), default))
			e_thresh=default

		# Select events above energy threshold
		e_mask = np.where(np.array(e_temp)>=e_thresh)[0]
		print("{} events selected via energy threshold".format(len(e_mask)))

		self.dec = np.array(dec_temp)[e_mask]
		self.ra = np.array(ra_temp)[e_mask]
		
		# set source extent by formula sigma_CR = D*100EeV/E_CR
		self.sigma = D*100./np.array(e_temp)[e_mask]
		
		if inj:
			self.set_injection_position()

	def inject_sources(self, gamma, mu_per_source, poisson=True):
		r"""
		Inject point sources
		
		Parameters:
			gamma : Spectral index
			
			mu_per_source : Mean number of neutrinos per source
			
			poisson : Poissonian mu or not
		
		"""
		# Inject the events
		print("Inject signal events by MC sampling ...")
		self.injector = InjectorHandler(gamma)
		self.injector.fill(self.dec, self.mc, self.livetime)
		self.inject = self.injector.sample(self.ra_rot,
																			 mu_per_source).next()[1]
		print("Sampling done.")

	def fit_source(self, inj=False, **kwargs):
		r"""
		Thin wrapper for fitting sources, no kwargs added yet
		"""
		if inj:
			gamma=kwargs.pop("gamma", _gamma)
			mu_per_source=kwargs.pop("mu", _mu_per_source)
			self.inject_sources(gamma, mu_per_source)
		
		print("Fitting sources...")
		fmin, xmin = self.ps_llh.fit_source(src_ra=self.ra, src_dec=self.dec, src_sigma=self.sigma, inject=self.inject)

		print ("fmin: {}".format(fmin))
		print ("xmin: {}".format(xmin))
		
		return fmin, xmin

	def bg_trials(self, n_iter):
		"""
		Wrapper for do_trials function of ps_llh
		This is meant for background trials, no injection of events possible.
		Function ""s_trial"" under construction.

		Parameters:
			n_iter : 	number of iterations
		"""
		self.trials = self.ps_llh.do_trials(src_ra=self.ra, src_dec=self.dec, n_iter=n_iter, src_sigma=self.sigma)

	def s_trials(self):
		r"""
		Under construction
		"""
		print("under construction")

	def set_save_path(self, path):
		self.save_path = path
		print("Savepath set to: {}".format(self.save_path)) 

	def save_trials(self, job, mode=0754):
		header=""
		for i in self.trials.dtype.names:
			header+=i+" "
			   
		savestring = os.path.join(self.save_path, "trials_job{}".format(job))
		prepareDirectory(os.path.join(self.save_path), subs=False)
		np.savetxt(	savestring,
								self.trials, 
								header=header,
								comments=""
							)
		os.chmod(savestring, mode)
		print("{} trials saved to {}".format(len(self.trials), savestring))

class MultiStackExtendedSources(StackExtendedSources):
	r"""
	Handling initialization and preparation for stacking multiple samples
	"""
	def __init__(self, basepath, detector, e_thresh=0., D=np.radians(6.), random_uhecr=False, **kwargs):
		r"""
		Initialize parameters, set UHECR positions
		Initialize MultiPointSourceLLH()
		Fill with initial MC and EXP data
		
		Parameters:
			basepath :	string
									Where to read the data from
			
			detector : 	string
									Which detector to use

			e_thresh : 	int
									energy threshold for UHECR selection

			D : 				float
									Magnetic deflection parameter, usually radian(3 - 9 degrees)
									
			random_uhecr : 	bool
											do or do not use random uhecr positions

		Optional parameters:
			size : 	int, number of random uhecr events

			sets : 	list of strings

			kwargs for PointSourceLLH class:
				llh_model : ExtendedLLh() or EnergyLLH(), if none given, ClassicLLH() will be used
				mode : "all", "box", "band"; used for event selection. Default is "all"
		"""
		# kwargs for random source position, if random_uhecr == True
		size = kwargs.pop("size", _size)

		# kwargs for uhecr
		sets = kwargs.pop("sets", ["auger", "ta"])
		#~ # Initialize injector
		#~ self.inject = None

		# Initialize source positions
		if random_uhecr:
			self.set_random_source_positions(size=size, scale=D)
		else:
			self.set_UHECR_positions(sets, D, e_thresh)

		# Initialize MultiLLH
		self.ps_llh = MultiPointSourceLLH()        
		
		# Add all samples
		print("LLH setup...")
		for det in detector:
			self.add_sample(basepath, det, **kwargs)
				
	def add_sample(self, basepath, detector, **kwargs):
		r"""
		Wrapper for MultiPointSourceLLH.add_sample
		
		Parameters:
		basepath :	string
								Where to read the data from

		detector : 	string
								Which detector to use
		
		kwargs for PointSourceLLH class:
				llh_model : e.g. EnergyLLH() or ExtendedLLH(), 
										if none given, ClassicLLH() is chose
		"""
		mc, exp, livetime = load_data(basepath, detector)
		mc = np.rec.array(mc)
		exp = np.rec.array(exp)
		self.ps_llh.add_sample(name=detector, 
													 obj=PointSourceLLH(copy.copy(exp), 
																							copy.copy(mc),
																							livetime, 
																							**kwargs
																						 )
													)
		del mc
		del exp
