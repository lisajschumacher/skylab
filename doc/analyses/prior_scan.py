# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import numpy.lib.recfunctions
import healpy as hp
from scipy.stats import percentileofscore

class ScrambledPriorScan(object):
    
    def __init__(self, scan, spatial_prior, rs=np.random.RandomState(0), hotspots=None):
        r"""
        scan: healpy array with fields 
             'ra',
             'dec',
             'theta',
             'TS',
             'pVal'
        spatial_priors: instance of SpatialPrior class
        """
        self.original_scan = scan
        self.spatial_prior = spatial_prior
        self.nprior = len(spatial_prior.p)
        assert(len(scan)==len(spatial_prior.p[0]))
        self.nside = int(np.sqrt(len(scan)//12))
        self.rs = rs
        if hotspots is None:
            self.original_hotspots = self.all_sky_scan_standalone(scan, self.nside, spatial_prior)[1]
        else:
            self.original_hotspots = hotspots
        
    def scramble_scan(self, scan):
        scan['ra'] = self.rs.uniform(0, np.pi*2, size=len(scan))
        ipix = hp.ang2pix(self.nside, scan['theta'], scan['ra'])
        return scan[ipix]
        
    def do_scrambled_scan(self, niter=1):
        for ii in range(niter):
            scan = np.copy(self.original_scan)
            scan = self.scramble_scan(scan)
            result, hotspots = self.all_sky_scan_standalone(scan, self.nside, spatial_prior=self.spatial_prior)
            
            yield result, hotspots
    
    def all_sky_scan_standalone(self, scan, nside, spatial_prior):
        r"""Scan the entire sky for single point sources.

        Perform an all-sky scan. First calculation is done on a coarse
        grid with `nside`, follow-up scans are done with a finer
        binning, while conserving the number of scan points by only
        evaluating the most promising grid points.

        Parameters
        ----------
        nside : int
            NSide value for initial HEALPy map; must be power of 2.
        scan: skymap on HEALPY grid with fields ra, dec, theta, TS, pVal
        spatial_prior:
        Returns
        -------
        results : np.ndarray
            Structured array describing the likelihood result at every location
            checked during the all-sky scan.
        hotspots : dict
            Dictionary with information about hottest spot(s).


        """

        #if pVal is None:
        def pVal(ts, sindec):
            return ts

        # search for hotspots
        drange = [-np.pi/2., np.pi/2.]

        hotspots = {}
        nprior = spatial_prior.nprior
        ts_names = ["TS_spatial_prior_%d" % i for i in range(nprior)]

        # append space for total TS with each spatial prior
        result = numpy.lib.recfunctions.append_fields(
            scan, names=ts_names, dtypes=nprior*[np.float],
            data=np.zeros((nprior,len(scan))), usemask=False)

        # calculate TS with spatial priors
        for i,ts_name in enumerate(ts_names):
            spatial_prior_ts_i = spatial_prior.ts(i)
            try: 
                result[ts_name] = spatial_prior_ts_i(scan["TS"], scan["ra"], scan["dec"])
            except Exception as e:
                print result.dtype.names
                print scan.dtype.names
                print ts_name
                print i
                raise(Exception)

            # search for hotspot with prior
            hotspots.update(self.hotspot_grid_standalone(
                            scan=result, nside=nside,
                            hemispheres={ts_name.lstrip("TS_"):np.radians([-90., 90.])},
                            drange=drange, pVal=pVal, spatial_prior_ts=spatial_prior_ts_i, 
                            order=ts_name, ts_name=ts_name) 
                           )

        return result, hotspots

    def hotspot_grid_standalone(self, scan, nside, hemispheres, drange, pVal,
        spatial_prior_ts=None, order=["pVal","TS"], ts_name="TS"):
        r"""
        Gather information about hottest spots in each hemisphere.
        Based on a grid scan without prior

        -- stand-alone version --
        """
        result = {}
        for key, dbound in hemispheres.items():
            mask = (
                (scan["dec"] >= dbound[0]) & (scan["dec"] <= dbound[1]) &
                (scan["dec"] > drange[0]) & (scan["dec"] < drange[1])
                )

            if not np.any(mask):
                print("{0:s}: no events here.".format(key))
                continue

            """if not np.any(scan[mask]["TS"] > 0):
                print("{0}: no over-fluctuation.".format(key))
                continue"""

            hotspot = np.sort(scan[mask], order=order)[-1]

            # re-compute p-value here to account for spatial priors which
            # modify the original TS value and yield different pVal
            hotspot["pVal"] = pVal(hotspot[ts_name], np.sin(hotspot["dec"]))

            result[key] = dict(grid=dict(
                ra=hotspot["ra"],
                dec=hotspot["dec"],
                nside=nside,
                pix=hp.ang2pix(nside, np.pi/2 - hotspot["dec"], hotspot["ra"]),
                TS=hotspot[ts_name],
                pVal=hotspot["pVal"]))

        return result
    
    def calculate_post_trial_pvalue(self, niter=1000, verbose=False):
        TS = np.zeros((niter,))
        parter = max(niter//10, 1) 
        for i,(result, hotspot) in enumerate(self.do_scrambled_scan(niter)):
            if verbose and (i%parter)==0: print i
            TS[i] = np.array([hotspot[hs]["grid"]["TS"] for hs in hotspot]).sum()
            
        original_TS_sum = sum([self.original_hotspots[hs]["grid"]["TS"] for hs in self.original_hotspots])
        #result = percentileofscore(TS, original_TS_sum)
        
        return TS, original_TS_sum