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

@file    stacking_priorllh.py
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

class PriorLLHMixin(object):
    r""" 
    Mixin for several analysis functions, 
    which need to be overridden over baseLLH.
    In particular:
        all_sky_scan
        _hotspots
        fit_source_loc
        do_trials    
    """
    _logname = "MixIn"
    
    def _add_injection(self, inject):
	if isinstance(inject, dict):
	    """
            if inject is still a dict, we are working with a single sample
	    Then, just take the content of the first (and only) key
	    """
	    inject = inject[inject.keys()[0]]
	inject = numpy.lib.recfunctions.append_fields(
	    inject, names="B", data=self.llh_model.background(inject),
	    usemask=False)
	
	self.exp = np.append(self.exp, inject)

    def _remove_injection(self):
	self.exp = self.exp[:self._nbase_events]
    
    #~ @profile
    def all_sky_scan(self, prior, nside=128, follow_up_factor=2,
                        hemispheres=None, pVal=None, **kwargs):
        r"""Scan the entire sky for single point sources.

	Perform an all-sky scan. First calculation is done on a coarse
	grid with `nside`, follow-up scans are done with a finer
	binning, while conserving the number of scan points by only
	evaluating the most promising grid points.
	
	### New: ###
	Add a spatial prior on the TS values already optimized for ns and gamma
	Fit the hotspot on the prior-TS and return these results
	-> the Prior selects the interesing region and pulls the spatial fit
	### Stacking ###
	Ability to repeat the calculation for different priors and add them all up
	### --- ###

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

        # If a prior is given, it has to have the same shape as ts
        if prior is not None:
            assert(np.shape(prior)[1] == len(ts))
        # If no prior is given, assume uniform prior (equivalent to no prior)
        else:
            # Initialize the prior with arrays and assert the lengths
            prior = np.zeros_like(ts)		

        niterations = 1
	tm = np.exp(prior)
	tm = tm/tm.sum(axis=1)[np.newaxis].T # should be normalized, but let's be sure ...
	tm = tm.sum(axis=0)
        while niterations <= 2:
            logger.warn("Iteration {0:2d}".format(niterations))
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
		
	    # make a mask based on the priors, because we don't want to scan useless areas
	    tm = hp.get_interp_val(tm, theta, ra)
            tm[tm<=0] = min(tm[tm>0])
            # logger.warn("min tm {}".format( min(tm)))
	    # this basically corresponds to 7 sigma range of prior size
	    mask &= np.log(tm)>-50
	    logger.info("percentage covered:", np.count_nonzero(mask)*100./len(mask))
	    
            nscan = mask.sum()
            area = hp.nside2pixarea(nside) / np.pi

            logger.info(
                    "Scan area of {0:4.2f}pi sr ({1:.2%}, {2:d} pix)...".format(
                            nscan * area, nscan / mask.size, nscan))

            time = datetime.datetime.now()

            # Here, the actual scan is done.
            # This scan is needed only once, prior and hotspot fit
            # are done individually for each UHECR position
            ts, xmin = self._scan(ra[mask], dec[mask], ts, xmin, mask)

            time = datetime.datetime.now() - time
            logger.info("Finished after {0}.".format(time))
	    if follow_up_factor==0: break
	    logger.info(
	        "Next follow-up: nside = {0:d} * 2**{1:d} = {2:d}".format(
		nside, follow_up_factor, nside * 2**follow_up_factor))

            nside *= 2**follow_up_factor
            niterations += 1
	    
	result = np.array(zip(ra, dec, theta, ts, np.zeros_like(ts)),
			  [(f, np.float) for f in "ra", "dec", "theta", "preTS", "allPrior"]
			 )
	self.postTS = np.zeros((len(prior),len(ts)))
	result = numpy.lib.recfunctions.append_fields(
		 result, names=self.params, 
		 data=[xmin[p] for p in self.params],
		 dtypes=[np.float for p in self.params], usemask=False
		 )
	hotspots = []

	for i,prior_i in enumerate(prior):
	    # Calculate the prior on the new grid
	    current_prior = hp.get_interp_val(prior_i, theta, ra)

	    ## Now we add the prior ##
	    p_ts = ts + 2.*current_prior 
	    pvalue = pVal(p_ts, np.sin(dec))
	    names = ["TS", "pVal", "prior"]
	    ## ... and find the hotspot
	    hotspots.append(self._hotspot(numpy.lib.recfunctions.append_fields(
				    result, names=names,
				    data=[p_ts, pvalue, current_prior],
				    dtypes=[np.float for n in names], usemask=False),
				nside, hemispheres, drange, pVal)
			    )
	    result["allPrior"] += np.exp(current_prior)
	    self.postTS[i] += np.where(p_ts>0.,p_ts,np.zeros_like(p_ts))

	return result, np.array(hotspots)


    
    def _hotspot(self, scan, nside, hemispheres, drange, pVal):
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
        logger = logging.getLogger(self._logname + ".hotspot")
        result = {}
        for key, dbound in hemispheres.iteritems():
            mask = (
                (scan["dec"] >= dbound[0]) & (scan["dec"] <= dbound[1]) &
                (scan["dec"] > drange[0]) & (scan["dec"] < drange[1])
                )

            if not np.any(mask):
                logger.info("{0:s}: no events here.".format(key))
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
                ns=hotspot["nsources"],
                nside=nside,
                pix=hp.ang2pix(nside, np.pi/2 - hotspot["dec"], hotspot["ra"]),
                TS=hotspot["TS"],
                pVal=hotspot["pVal"]))

            result[key]["grid"].update(seed)

            if not np.any(scan[mask]["TS"] > -10):
                logger.info("{0}: no over-fluctuation.".format(key))
                result[key]["fit"] = result[key]["grid"]
                result[key]["best"] = result[key]["grid"]
                continue

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
            func, grad = super(PriorLLHMixin,self).llh(**params)
            prior_val = hp.get_interp_val(args[0], np.pi/2. - x[1], x[0])

            return -func - 2.*prior_val

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
        #fmin *= -np.sign(pbest["nsources"])

        # Clear cache.
        self._src_ra = np.inf
        self._src_dec = np.inf

        return fmin, pbest
    
    def do_trials(self, prior, n_iter=2, mu=None, do_scrambles=True, **kwargs):
        r"""Create trials of scrambled event maps to estimate the test
		statistic distribution.

		Parameters
		----------
		src_ra : float
			Right ascension of source position
		src_dec : float
			Declination of source position
		n_iter : int, optional
			Number of trials to create
		mu : Injector, optional
			Inject additional events into the scrambled map.
		\*\*kwargs
			Parameters passed to `fit_source`

		Returns
		-------
		ndarray
			Structured array of information about hotspots

		"""
        logger = logging.getLogger(self._logname+".do_trials")
        if do_scrambles==False and n_iter>1:
            logger.warn("More than one trial and NO scrambling!!")
	
        if mu is None:
            mu = itertools.repeat((0, None))

        inject = [mu.next() for i in range(n_iter)]

        follow_up_factor = kwargs.pop("follow_up_factor", 2)
        hemispheres = kwargs.pop("hemispheres",
                                    dict(
                                    South=(-np.pi/2., -np.deg2rad(5.)),
                                    North=(-np.deg2rad(5.), np.pi/2.)))
        h_keys = hemispheres.keys()
        res_keys = ["ra", "dec"] + self.params #super(PriorLLHMixin, self).params
        best_hotspots = np.zeros(len(prior),
                                  dtype=[(p, np.float) for p in h_keys] 
                                  +[(p, np.float) for p in res_keys]
                                  +[("best", np.float)]
				  +[("ra_inj", np.float)]
				  +[("dec_inj", np.float)]
                                  +[("n_inj", np.float)])

        # all_sky_scan for every trial
        #for i in range(n_iter):
        i = 0
        while True:
	    if mu is not None and inject[i][1] is not None:
		self._add_injection(inject[i][1])
		best_hotspots["n_inj"] = inject[i][0]
		best_hotspots["ra_inj"] = inject[i][2]
		best_hotspots["dec_inj"] = inject[i][3]
		
            result, hotspots = self.all_sky_scan(prior,
                                                 hemispheres=hemispheres,
                                                 follow_up_factor = follow_up_factor,
                                                 **kwargs)
	    #~ if follow_up_factor == 0: break # In case you don't want a follow-up
	    #~ if scan_i > 0:
	    #~ # break after first follow up
	    #~ break
            
            for h_i,hspots in enumerate(hotspots):
                for hk in h_keys:
                    best_hotspots[hk][h_i] = hspots[hk]["best"]["TS"]
                if len(h_keys)==2 and (best_hotspots[h_keys[1]][h_i] >= best_hotspots[h_keys[0]][h_i]):
                    best_hotspots["best"][h_i] = best_hotspots[h_keys[1]][h_i]
                    for p in res_keys:
                        best_hotspots[p][h_i] = hspots[h_keys[1]]["best"][p]
                else:
                    best_hotspots["best"][h_i] = best_hotspots[h_keys[0]][h_i]
                    for p in res_keys:
                        best_hotspots[p][h_i] = hspots[h_keys[0]]["best"][p]

	    # Remove injected events and then
            # scramble these events after scan
            # Call the corresponding scramble method of super class
	    if mu is not None and inject[i][1] is not None: 
		self._remove_injection()
            if do_scrambles: super(PriorLLHMixin, self)._scramble_exp()
            yield best_hotspots, result
            i += 1
            if i>=n_iter: return

        #return best_hotspots, result


class StackingPriorLLH(PriorLLHMixin, psLLH.PointSourceLLH):

    r"""
    PointSource and MixIn functionality,
    slightly modified all-sky-scan that can add multiple priors
    and find each hotspot afterwards.
    Results are finally added up.

    See PointSourceLLH and PriorLLHMixin for more information
    """

    # The log-likelihood function will be taylor-expanded around this treshold
    # value; see llh method.
    _aval = 1e-3


class MultiStackingPriorLLH(PriorLLHMixin, psLLH.MultiPointSourceLLH):
    r"""
    PointSource and Base functionality,
    slightly modified all-sky-scan (and working on event selection) (to do)
    See PointSourceLLH and BaseLLh for more information

    Handles multiple event samples that are distinct of each other.
    Different samples have different effective areas that have to be
    taken into account for parting the number of expected neutrinos in
    between the different samples. Each sample is represented as an
    instance of `PointSourceLLH`.
    """
    def add_sample(self, name, llh):
	r"""Add log-likelihood function object.

	Parameters
	-----------
	name : str
		Name of event sample
	llh : StackingPriorLLH
		Log-likelihood function using single event sample

	"""
	if not isinstance(llh, StackingPriorLLH):
		raise ValueError("'{0}' is not correct LLH-style".format(llh))

	names = self._enums.values()

	if name in names:
		enum = self._enums.keys()[names.index(name)]
		logger = logging.getLogger(self._logname)
		logger.warn("Overwrite {0:d} - {1}".format(enum, name))
	else:
		if len(names) > 0:
			enum = max(self._enums) + 1
		else:
			enum = 0

	self._enums[enum] = name
	self._samples[enum] = llh

    def _add_injection(self, inject):
	for enum in self._samples:
            if isinstance(inject, dict):
                events = inject.pop(enum, None)
            else:
                events = inject

            self._samples[enum]._add_injection(inject=events)

    def _remove_injection(self):
	for enum in self._samples:
            self._samples[enum]._remove_injection()
