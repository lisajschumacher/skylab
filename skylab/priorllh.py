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

	# The log-likelihood function will be taylor-expanded around this treshold
	# value; see llh method.
	
	def __init__(self, *args, **kwargs):
		super(PriorLLH, self).__init__(*args, **kwargs)
	'''
	# In Case of wanting to change the init ...
	def __init__(self, exp, mc, livetime, llh_model, scramble=True, mode="box",
				 delta_ang=np.deg2rad(10.), thresh_S=0., **kwargs):
		super(PriorLLH, self).__init__(exp, mc, livetime, llh_model, scramble=scramble, mode=mode,
				 delta_ang=delta_ang, thresh_S=thresh_S, **kwargs)
	'''

	def all_sky_scan(self, nside=128, follow_up_factor=2, hemispheres=None,
								 pVal=None, **kwargs):
		r"""Scan the entire sky for single point sources.

		Perform an all-sky scan. First calculation is done on a coarse
		grid with `nside`, follow-up scans are done with a finer
		binning, while conserving the number of scan points by only
		evaluating the most promising grid points.

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

		prior_dec = kwargs.pop("pdec", 0.)
		prior_ra = kwargs.pop("pra", np.pi)
		prior_sigma = kwargs.pop("psig", np.radians(6.))
		logger.info("Adding Gaussian prior at (dec, ra)=({0:1.2f},{1:1.2f}) rad,".format(prior_dec, prior_ra)
					+"sigma is {0:1.2f} deg".format(prior_sigma))
		
		# Calculate prior central direction 
		mean_vec = UnitSphericalRepresentation(Angle(prior_ra, u.radian), Angle(prior_dec, u.radian))

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

			## Now we add our template ##
			map_vec = UnitSphericalRepresentation(Angle(ra, u.radian), Angle(dec, u.radian))
			prior = -1.*np.power((map_vec-mean_vec).norm(), 2) / prior_sigma**2 
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

