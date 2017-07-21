# -*- coding: utf-8 -*-

r"""This file is part of an experimental extension to SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@file    priorllh.py
@authors Lisa Schumacher
@date    July, 2017
"""
from __future__ import division

import logging
import warnings
import datetime
import itertools
import multiprocessing

import healpy as hp
import numpy as np
import numpy.lib.recfunctions
import scipy.optimize

from astropy.coordinates import UnitSphericalRepresentation
from astropy.coordinates import Angle
from astropy import units as u

from . import utils
from . import psLLH

class PriorLLH(psLLH.PointSourceLLH):

	r"""
	PointSource and Base functionality,
	slightly modified all-sky-scan (and working on event selection) (to do)
	See PointSourceLLH and BaseLLh for more information

	Suggested tests: (To do)
	- Put a flat zero prior map into this function and compare with standard psLLH
	- Put a large prior-sigma into this function (~180 deg?) and compare
	"""
	
	# The log-likelihood function will be taylor-expanded around this treshold
	# value; see llh method.
	_aval = 1e-3
	
	def __init__(self, *args, **kwargs):
		super(PriorLLH, self).__init__(*args, **kwargs)
	'''
	# In Case of wanting to change the init ...
	def __init__(self, exp, mc, livetime, llh_model, scramble=True, mode="box",
				 delta_ang=np.deg2rad(10.), thresh_S=0., **kwargs):
		super(PriorLLH, self).__init__(exp, mc, livetime, llh_model, scramble=scramble,
				mode=mode, delta_ang=delta_ang, thresh_S=thresh_S, **kwargs)
	'''

	def all_sky_scan(self, nside=128, follow_up_factor=2, hemispheres=None,
								 pVal=None, **kwargs):
		r"""Scan the entire sky for single point sources.

		Perform an all-sky scan. First calculation is done on a coarse
		grid with `nside`, follow-up scans are done with a finer
		binning, while conserving the number of scan points by only
		evaluating the most promising grid points.
		
		### New: ###
		Add a spatial prior on the TS values already optimized for ns and gamma
		Fit the hotspot on the prior-TS and return these results
		-> the Prior selects the interesing region and pulls the spatial fit
		### ---- ###

		Parameters
		----------
		nside : int, optional
				NSide value for initial HEALPy map; must be power of 2.
		follow_up_factor : int, optional
				Controls the grid size of following scans,
				``nside *= 2**follow_up_factor``.
		hemispheres : dict(str, tuple(float)), optional
				Declination boundaries in radian of northern and southern
				sky; by default, the horizon is at -5 degrees.
		pVal : callable, optional
				Calculates the p-value given the test statistic and
				optionally sine declination of the source position; by
				default the p-value is equal the test statistic. The p-value
				must be monotonic increasing, because follow-up scans focus
				on high values.
				
		Keyword arguments
		-----------------
		prior : array, length = hp.nside2npix(nside), optional
				prior map to be added onto ts map, already log10 applied
		alternatives :
			pdec, pra, psig : floats, optional
				give position and spread of a Gaussian prior,
				which will be calculated in this function, if 'prior'
				(see above) is not given
				
		Returns
		-------
		iterator
				Structured array describing the scan result and mapping of
				hemispheres to information about the hottest spot.

		Examples
		--------
		In many cases, the test statistic is chi-square distributed.

		>>> def pVal(ts, sindec):
		...     return -numpy.log10(0.5 * scipy.stats.chi2(2.).sf(ts))

		"""
		logger = logging.getLogger(self._logname + ".all_sky_scan")
		logger.info("Parameters for fitting: "+str(self.params))

		if pVal is None:
				def pVal(ts, sindec):
						return ts

		if hemispheres is None:
			hemispheres = dict(
					South=(-np.pi/2., -np.deg2rad(5.)),
					North=(-np.deg2rad(5.), np.pi/2.))

		drange = np.arcsin(self.sinDec_range)

		# NOTE: unique sorts the input list.
		dbound = np.unique(np.hstack([drange] + hemispheres.values()))
		dbound = dbound[(dbound >= drange[0]) & (dbound <= drange[1])]

		npoints = hp.nside2npix(nside)
		ts = np.zeros(npoints, dtype=np.float)
		xmin = np.zeros_like(ts, dtype=[(p, np.float) for p in self.params])
		
		# Pull a prior map, if given
		prior = kwargs.pop("prior", None)
		# If a prior is given, it has to have the same shape as ts
		if prior is not None:
			assert(len(prior) == len(ts))
			calc_prior = False
		# If no prior is given, pull parameters for Gaussian prior
		else:
			prior_dec = kwargs.pop("pdec", 0.)
			prior_ra = kwargs.pop("pra", np.pi)
			prior_sigma = kwargs.pop("psig", np.radians(6.))
			# Calculate prior central direction 
			mean_vec = UnitSphericalRepresentation(Angle(prior_ra, u.radian), Angle(prior_dec, u.radian))
			# Make sure that the prior really is calculated for each iteration
			# Maybe we want to change this for computational reasons
			# TODO
			calc_prior = True
			
			logger.info("Adding Gaussian prior at (dec, ra)=({0},{1}) rad,".format(np.degrees(prior_dec), np.degrees(prior_ra))
					+"sigma is {0:1.2f} deg".format(np.degrees(prior_sigma)))		

		niterations = 1
		while True:
			logger.info("Iteration {0:2d}".format(niterations))
			logger.info("Generating equal distant points on sky map...")
			logger.info("nside = {0:d}, resolution {1:.2f} deg".format(
					nside, np.rad2deg(hp.nside2resol(nside))))

			# Create grid in declination and right ascension.
			# NOTE: HEALPy returns the zenith angle in equatorial coordinates.
			theta, ra = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
			dec = np.pi/2 - theta

			# Interpolate previous scan results on new grid.
			ts = hp.get_interp_val(ts, theta, ra)

			# Calculate the prior on the new grid, or
			# If the prior is given directly into the function, interpolate it now
			if calc_prior:
				map_vec = UnitSphericalRepresentation(Angle(ra, u.radian),
													  Angle(dec, u.radian))
				prior = -1.*np.power((map_vec-mean_vec).norm(), 2) / prior_sigma**2
			else:
				prior = hp.get_interp_val(prior, theta, ra)
				
			pvalue = pVal(ts, np.sin(dec))

			xmin = np.array(zip(
					*[hp.get_interp_val(xmin[p], theta, ra) for p in self.params]),
					dtype=xmin.dtype)

			# Scan only points above the p-value threshold per hemisphere. The
			# thresholds depend on what percentage of the sky is evaluated.
			ppoints = npoints / ts.size
			logger.info("Analyse {0:.2%} of scan...".format(ppoints))

			mask = np.isfinite(ts) & (dec > drange[0]) & (dec < drange[-1])

			for dlow, dup in zip(dbound[:-1], dbound[1:]):
				logger.info("dec = {0:.1f} to {1:.1f} deg".format(
						np.rad2deg(dlow), np.rad2deg(dup)))

				dout = np.logical_or(dec < dlow, dec > dup)

				if np.all(dout):
						logger.warn("No scan points here.")
						continue

				threshold = np.percentile(
						pvalue[~dout], 100.*(1. - ppoints))

				tabove = pvalue >= threshold

				logger.info(
						"{0:.2%} above threshold p-value = {1:.2f}.".format(
								np.sum(tabove & ~dout) / (tabove.size - dout.sum()),
								threshold))

				# Apply threshold mask only to points belonging to the current
				# hemisphere.
				mask &= np.logical_or(dout, tabove)

			nscan = mask.sum()
			area = hp.nside2pixarea(nside) / np.pi

			logger.info(
					"Scan area of {0:4.2f}pi sr ({1:.2%}, {2:d} pix)...".format(
							nscan * area, nscan / mask.size, nscan))

			time = datetime.datetime.now()

			# Here, the actual scan is done.
			ts, xmin = self._scan(ra[mask], dec[mask], ts, xmin, mask)
			## Now we add the prior ##
			ts += prior
			pvalue = pVal(ts, np.sin(dec))

			time = datetime.datetime.now() - time
			logger.info("Finished after {0}.".format(time))

			result = np.array(
					zip(ra, dec, theta, ts, pvalue, prior),
					[(f, np.float) for f in "ra", "dec", "theta", "TS", "pVal", "prior"])

			result = numpy.lib.recfunctions.append_fields(
					result, names=self.params, data=[xmin[p] for p in self.params],
					dtypes=[np.float for p in self.params], usemask=False)

			yield result, self._hotspot(
							result, nside, hemispheres, drange, pVal, logger)

			logger.info(
					"Next follow-up: nside = {0:d} * 2**{1:d} = {2:d}".format(
							nside, follow_up_factor, nside * 2**follow_up_factor))

			nside *= 2**follow_up_factor
			niterations += 1

	def _hotspot(self, scan, nside, hemispheres, drange, pVal, logger):
		r"""Gather information about hottest spots in each hemisphere.

		"""
		result = {}
		for key, dbound in hemispheres.iteritems():
			mask = (
				(scan["dec"] >= dbound[0]) & (scan["dec"] <= dbound[1]) &
				(scan["dec"] > drange[0]) & (scan["dec"] < drange[1])
				)

			if not np.any(mask):
				logger.info("{0:s}: no events here.".format(key))
				continue

			if not np.any(scan[mask]["nsources"] > 0):
				logger.info("{0}: no over-fluctuation.".format(key))
				continue

			hotspot = np.sort(scan[mask], order=["pVal", "TS"])[-1]
			seed = {p: hotspot[p] for p in self.params}

			logger.info(
				"{0}: hot spot at ra = {1:.1f} deg, dec = {2:.1f} deg".format(
					key, np.rad2deg(hotspot["ra"]),
					np.rad2deg(hotspot["dec"])))

			logger.info("p-value = {0:.2f}, t = {1:.2f}".format(
				hotspot["pVal"], hotspot["TS"]))

			logger.info(
				",".join("{0} = {1:.2f}".format(p, seed[p]) for p in seed))

			result[key] = dict(grid=dict(
				ra=hotspot["ra"],
				dec=hotspot["dec"],
				nside=nside,
				pix=hp.ang2pix(nside, np.pi/2 - hotspot["dec"], hotspot["ra"]),
				TS=hotspot["TS"],
				pVal=hotspot["pVal"]))

			result[key]["grid"].update(seed)

			fmin, xmin = self.fit_source_loc(
				hotspot["ra"], hotspot["dec"], size=hp.nside2resol(nside),
				seed=seed, prior=scan["prior"])

			pvalue = np.asscalar(pVal(fmin, np.sin(xmin["dec"])))

			logger.info(
				"Re-fit location: ra = {0:.1f} deg, dec = {1:.1f} deg".format(
					np.rad2deg(xmin["ra"]), np.rad2deg(xmin["dec"])))

			logger.info("p-value = {0:.2f}, t = {1:.2f}".format(
				pvalue, fmin))

			logger.info(
				",".join("{0} = {1:.2f}".format(p, xmin[p]) for p in seed))

			result[key]["fit"] = dict(TS=fmin, pVal=pvalue)
			result[key]["fit"].update(xmin)

			if result[key]["grid"]["pVal"] > result[key]["fit"]["pVal"]:
				result[key]["best"] = result[key]["grid"]
			else:
				result[key]["best"] = result[key]["fit"]

		return result
		
	def fit_source_loc(self, src_ra, src_dec, size, seed, prior, **kwargs):
		r"""Minimize the negative log-likelihood function around source
		position.

		Parameters
		----------
		src_ra : float
			Right ascension of interesting position
		src_dec : float
			Declination of interesting position
		size : float
			Size of box around source position for minimization
		seed : dict(str, float)
			Seeds for remaining parameters; e.g. result from a previous
			`fit_source` call.
		\*\*kwargs
			Parameters passed to the L-BFGS-B minimizer

		Returns
		-------
		fmin : float
			Minimal negative log-likelihood converted into the test
			statistic ``-sign(ns)*llh``
		pbest : dict(str, float)
			Parameters minimizing the negative log-likelihood function

		"""
		def llh(x, *args):
			r"""Wrap log-likelihood to work with arrays and return the
			negative log-likelihood, which will be minimized.

			"""
			# If the minimizer is testing a new position, different events have
			# to be selected; cache position.
			if (np.fabs(x[0] - self._src_ra) > 0. or
					np.fabs(x[1] - self._src_dec) > 0.):
				self._select_events(x[0], x[1])
				self._src_ra = x[0]
				self._src_dec = x[1]

			params = dict(zip(self.params, x[2:]))
			func, grad = self.llh(**params)
			prior_val = hp.get_interp_val(args[0], np.pi/2. - x[1], x[0])

			return -func - prior_val

		dra = size / np.cos(src_dec)

		bounds = [
			(max(0., src_ra - dra), min(2.*np.pi, src_ra + dra)),
			(src_dec - size, src_dec + size)
			]

		bounds = np.vstack([bounds, self.par_bounds])
		params = [src_ra, src_dec] + [seed[p] for p in self.params]

		kwargs.pop("approx_grad", None)
		kwargs.setdefault("pgtol", self._pgtol)

		xmin, fmin, success = scipy.optimize.fmin_l_bfgs_b(
			llh, params, args=(prior,), bounds=bounds, approx_grad=True, **kwargs)

		if self._nevents > 0 and abs(xmin[0]) > self._rho_max*self._nselected:
			warnings.warn(
				"nsources > {0:.2%} * {1:d} selected events, fit-value "
				"nsources = {2:.1f}".format(
					self._rho_max, self._nselected, xmin[0]),
				RuntimeWarning)

		pbest = dict(ra=xmin[0], dec=xmin[1])
		pbest.update(dict(zip(self.params, xmin[2:])))

		# Separate over and under fluctuations.
		fmin *= -np.sign(pbest["nsources"])

		# Clear cache.
		self._src_ra = np.inf
		self._src_dec = np.inf

		return fmin, pbest
