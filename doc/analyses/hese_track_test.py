###############################################################################
# @file   hese_track_test.py
# @author Lisa Schumacher, Josh Wood
# @date   May 15 2018
# @brief  Example showing how to build a spatial prior from HESE error contour
#         and perform a search for a point source consisent with the contour.
###############################################################################

import os
import numpy as np
import healpy as hp

from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.ps_injector import PriorInjector
from skylab.llh_models import EnergyLLH
from skylab.datasets import Datasets
from skylab.utils import dist
from skylab.priors import SpatialPrior

def gaussian2D(sigma, dist):
    norm = 0.5/np.pi/sigma**2
    return norm * np.exp(-0.5*(dist / sigma)**2) 

GeV = 1
TeV = 1000 * GeV

###############################################################################

##################
# HESE PRIOR MAP #
##################

nside = 512
npix = hp.nside2npix(nside)
pixels = range(npix)

# HESE event 58
hese_ra = np.radians(102.1)
hese_dec = np.radians(-32.4)
hese_error = np.radians(1.3)

print("HESE event at (ra %.2f rad, dec %.2f rad)" % (hese_ra, hese_dec))

theta, phi = hp.pix2ang(nside, pixels)
ra  = phi
dec = np.pi/2. - theta

# angular distance between healpix bin centers and HESE event
hese_dist = dist(hese_ra, hese_dec, ra, dec)

# create prior map using 2D Gaussian spatial PDF
hese_prior_map = gaussian2D(hese_error, hese_dist)

# create spatial prior class from the map
spatial_prior = SpatialPrior(hese_prior_map)

##################
# HESE PRIOR MAP #
##################

###############################################################################

#######
# LLH #
#######

# NOTE: This is exactly the same as normal likelihood class setup

llh = [] # store individual llh as lists to prevent pointer over-writing
multillh = MultiPointSourceLLH(ncpu=20)

print("\n seasons:")
for season in ["IC86, 2012-2014"]:

    exp, mc, livetime = Datasets['PointSourceTracks'].season(season)
    sinDec_bins = Datasets['PointSourceTracks'].sinDec_bins(season)
    energy_bins = Datasets['PointSourceTracks'].energy_bins(season)

    msg = "   - % 15s (" % season
    msg += "livetime %7.2f days, %6d events" % (livetime, exp.size)
    msg += ", mjd0 %.2f" % min(exp['time'])
    msg += ", mjd1 %.2f)" % max(exp['time'])
    print(msg)

    llh_model = EnergyLLH(twodim_bins=[energy_bins, sinDec_bins],
                          allow_empty=True, bounds=[1., 4.],
                          seed=2., kernel=1, ncpu=10)

    llh.append(PointSourceLLH(exp, mc, livetime, mode="box",
                              scramble=True, llh_model=llh_model,
                              nsource_bounds=(0., 1e3), nsource=15.))

    multillh.add_sample(season, llh[-1])

    # save a little RAM by removing items copied into LLHs
    del exp, mc

# END for (season)

#######
# LLH #
#######

###############################################################################

############
# INJECTOR #
############

# Prior injector is similar to PointSourceInjector, but takes a prior map
# rather than src_ra, src_dec arguments. It injects point sources at random
# locations based on the prior map.

inj = PriorInjector(spatial_prior, gamma=2.0, E0=1 * TeV)
inj.fill(multillh.exp, multillh.mc, multillh.livetime)

print("\n injected spectrum:")
print("   - %s" % str(inj.spectrum))

############
# INJECTOR #
############

###############################################################################

################
# ALL-SKY SCAN #
################

import logging
logging.getLogger("skylab.ps_llh.PointSourceLLH").setLevel(logging.INFO)

# probability distribution of test-statistic (rough guess)
from scipy.stats import chi2
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(multillh.params)).sf(TS)
                                             + chi2(len(multillh.params)).cdf(-TS)))
# regular all-sky scan
result, hotspot = multillh.all_sky_scan(nside=2**4, follow_up_factor=1,
                                   pVal=pVal_func, iterations=2,
                                   hemispheres=dict(No_prior=np.radians([-90., 90.])))
for k in hotspot.keys():
    print("\n %s:" % k)
    print(hotspot[k])

# all-sky scan with prior
result, hotspot = multillh.all_sky_scan(nside=2**4, follow_up_factor=1,
                                   spatial_prior=spatial_prior,
                                   pVal=pVal_func, iterations=2)
for k in hotspot.keys():
    print("\n %s:" % k)
    print(hotspot[k])

# run three all-sky scans with different seeds to yield different trials.
# also inject a source at the same time to see if we can find it.
for i in range(3):
    print("\n all-sky scan with seed %d" % (i + 1000))

    ni, inject = inj.sample(mean_signal=200)
    print(" injected source with %d events" % ni[0])
    # NOTE: mean_signal is the number of signal events you expect at maximum
    # detector acceptance. The actual number of injected events will scale
    # according to the injection location to account for the declination 
    # dependence of IceCube's acceptance.

    result, hotspot = multillh.all_sky_scan(nside=2**4, follow_up_factor=1,
                                   spatial_prior=spatial_prior, rng_seed=i+1000,
                                   pVal=pVal_func, iterations=2, inject=inject,
                                   hemispheres=dict(No_prior=np.radians([-90., 90.])))
    print(" hotspot with prior %s" % hotspot["spatial_prior_0"])

################
# ALL-SKY SCAN #
################

###############################################################################

