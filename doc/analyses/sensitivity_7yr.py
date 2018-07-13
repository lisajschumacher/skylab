
from __future__ import print_function

import argparse
import os
import time
import numpy as np

from config import config, fit_background, GeV
import ic_utils as utils
from skylab.sensitivity_utils import estimate_sensitivity

###############################################################################

#############
# ARGUMENTS #
#############

p = argparse.ArgumentParser(description="Calculates Sensitivity and Discovery"
                            " Potential Fluxes for Point Source Analysis.",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--nscramble", default=1000, type=int,
               help="Number of background only scrambles used "
               "to measure TS distribution (default=1000)")
p.add_argument("--nsample", default=100, type=int,
               help="Number of signal samples used to compute "
               "discovery potential (default=100)")
p.add_argument("--ni-bounds", default=[0, 25], nargs=2,
               type=float, dest="ni_bounds",
               help="Range in signal events to search for "
               "discovery potential flux (default= 0 25)")
p.add_argument("--nstep", default=11, type=int,
               help="Number of signal injection steps (default=11)")
p.add_argument("--seed", default=1, type=int,
               help="Seed for RNG (default=1)")
p.add_argument("--ra", default=180., type=float,
               help="Right ascension of source in degrees (default=180)")
p.add_argument("--dec", default=0., type=float,
               help="Declination of source in degrees (default=0)")
p.add_argument("--discovery_thresh", default=5, type=float,
               help="Discovery threshold in sigma (default=5)")

args = p.parse_args()

src = np.empty(1, dtype=[('name', np.unicode_, 16),
                         ('ra', float), ('dec', float)])

src['name'] = "test"
src['ra'] = np.radians(args.ra)
src['dec'] = np.radians(args.dec)

tag = "dec%+06.2f" % args.dec

msg = "# output from \\`python sensitivity_multiyear.py"
msg += " --nscramble %d --nsample %d --discovery_thresh %.1f" % \
       (args.nscramble, args.nsample, args.discovery_thresh)
msg += " --ni-bounds %d %d --nstep %d --seed %d --dec %.3f\\`" % \
       (args.ni_bounds[0], args.ni_bounds[1], args.nstep, args.seed, args.dec)

os.system("echo \"" + msg + " on $HOSTNAME " + time.strftime("(%b %d, %Y)\""))
os.system("mkdir -p figures")

#############
# ARGUMENTS #
#############

###############################################################################

##########
# SKYLAB #
##########

# see config.py for more details

seasons = ["IC40", "IC59", "IC79", "ANTARES"]

startup_dict = dict(sample_names = seasons,
                    seed = args.seed,
                    src_gamma = 2.0,
                    src_dec = src["dec"],
                    rescale=6.,
                    Nsrc=1,
                    ncpu=25
                   )

llh, inj = utils.startup(**startup_dict)

#llh, inj = config(seasons, srcs=src, gamma=2.0, rng_seed=args.seed)

##########
# SKYLAB #
##########

###############################################################################

###################################
# BACKGROUND ONLY TS DISTRIBUTION #
###################################

print("\nRunning background only trials ...")

t0 = time.time()
trials = llh.do_trials(args.nscramble, src_ra=src['ra'], src_dec=src['dec'])
dt = time.time() - t0
print(" Completed %d trials in %.2f sec (%.2f trials/sec)" %
      (trials.size, dt, trials.size / dt))

ts_parameters = fit_background(
    trials, "figures/background_trials_%s.png" % tag)
median_ts, eta, ndf, scale = ts_parameters

print("\nBackground only TS > 0 described by:")
print(" Median TS: %6.4f" % median_ts)
print(" PDF(TS>0): %6.4f * chi2(ts / %.4f, ndf = %.4f)" % (eta, scale, ndf))

###################################
# BACKGROUND ONLY TS DISTRIBUTION #
###################################

###############################################################################

########################
# ESTIMATE SENSITIVITY #
########################

results = estimate_sensitivity(llh, inj, src_ra=src['ra'], src_dec=src['dec'],
                               nstep=args.nstep, nsample=args.nsample,
                               ni_bounds=args.ni_bounds,
                               ts_parameters=ts_parameters,  # see BACKGROUND ONLY TS DISTRIBUTION
                               disc_thresh=args.discovery_thresh,
                               path="figures/sensitivity_%s_" % tag)

sensitivity_str = (("| Sensitivity flux @ %.1f GeV" % (inj.E0 / GeV)) +
                   (" is %.2e" % (results['sensitivity_flux'] * GeV)) +
                   (" +/- %.2e" % (results['sensitivity_flux_error'] * GeV)) +
                   " GeV^-1cm^-2s^-1 |")

discovery_str = (("| Discovery flux   @ %.1f GeV" % (inj.E0 / GeV)) +
                 (" is %.2e" % (results['discovery_flux'] * GeV)) +
                 (" +/- %.2e" % (results['discovery_flux_error'] * GeV)) +
                 " GeV^-1cm^-2s^-1 |")

print("\n*" + "-" * (len(sensitivity_str) - 2) + "*")
print(sensitivity_str)
print(discovery_str)
print("*" + "-" * (len(sensitivity_str) - 2) + "*")

########################
# ESTIMATE SENSITIVITY #
########################

###############################################################################
