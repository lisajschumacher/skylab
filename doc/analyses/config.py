###############################################################################
# @file   config.py
# @author Josh Wood
# @date   Oct 13 2017
# @brief  Sets up skylab likelihood and injector for 7yr point source anlaysis
#         done by Stefan Coenders.
###############################################################################

import os
import numpy as np

from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.ps_injector import PointSourceInjector
from skylab.llh_models import ClassicLLH, EnergyLLH
from skylab.datasets import Datasets
from skylab.sensitivity_utils import DeltaChiSquare

GeV = 1
TeV = 1000 * GeV

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

if int(mpl.__version__[0]) > 1:
    mpl.style.use('classic')

###############################################################################


def config(seasons, srcs=None, rng_seed=1, scramble=True,
           gamma=2.0, E0=1 * TeV, e_range=(0, np.inf), verbose=True):
    r""" Configure multi season point source likelihood and injector.

    Parameters
    ----------
    seasons : list
      list of seasons to use in likelihood
    srcs : numpy.ndarray
      source list as structured numpy array with name, ra, dec columns
    rng_seed : int
      seed for random number generator
    scramble : bool
      scrambles data events in right ascension when True
    gamma : float
      spectral index for point source injection
    E0 : float
      normalization energy of power law spectrum (E / E0)^-gamma
    e_range : tuple
      min & max of true neutrino energy for point source injector
    verbose : bool
      shows print statements when True

    Returns
    -------
    multillh : MultiPointSourceLLH
      Multi year point source likelihood object
    inj : PointSourceInjector
     Point source injector object
    """

    # store individual llh as lists to prevent pointer over-writing
    llh = []

    multillh = MultiPointSourceLLH(seed=rng_seed, ncpu=25)

    # setup likelihoods
    if verbose:
        print("\n seasons:")
    for season in np.atleast_1d(seasons):

        exp, mc, livetime = Datasets['PointSourceTracks'].season(season)
        sinDec_bins = Datasets['PointSourceTracks'].sinDec_bins(season)
        energy_bins = Datasets['PointSourceTracks'].energy_bins(season)

        if verbose:
            msg = "   - % 15s (" % season
            msg += "livetime %7.2f days, %6d events" % (livetime, exp.size)
            msg += ", mjd0 %.2f" % min(exp['time'])
            msg += ", mjd1 %.2f)" % max(exp['time'])
            print(msg)

        llh_model = EnergyLLH(twodim_bins=[energy_bins, sinDec_bins],
                              allow_empty=True, bounds=[1., 4.],
                              seed=2., kernel=1)

        llh.append(PointSourceLLH(exp, mc, livetime, mode="box",
                                  scramble=scramble, llh_model=llh_model,
                                  nsource_bounds=(0., 1e3), nsource=15.))

        multillh.add_sample(season, llh[-1])

        # save a little RAM by removing items copied into LLHs
        del exp, mc

    # END for (season)

    if verbose:
        print("\n fitted spectrum:")
        print("   - dN/dE = A (E / %.1f GeV)^-index GeV^-1cm^-2s^-1" % E0)
        print("   - index is *fit*")

    # return with just multillh object if no src specified for injection
    if srcs is None:
        return multillh

    #######
    # LLH #
    #######

    ###########################################################################

    ############
    # INJECTOR #
    ############

    inj = PointSourceInjector(gamma=gamma, E0=1 * TeV,
                              seed=rng_seed, e_range=e_range)
    inj.fill(srcs['dec'], multillh.exp, multillh.mc, multillh.livetime)

    if verbose:
        print("\n injected spectrum:")
        print("   - %s" % str(inj.spectrum))

    ############
    # INJECTOR #
    ############

    ###########################################################################

    return (multillh, inj)

# END config()

###############################################################################


def fit_background(trials, image):
    r""" Fit background only TS distribution using delta function
         at ts = 0 and chi squared function for ts > 0.

    Parameters
    ----------
    trials : numpy.ndarray
      trials array from BaseLLH.do_trials()
    image : str
      path to output imate

    Return
    ------
    median_ts : float
      median of TS distribution
    eta : float
      fraction TS > 0
    ndf : float
      effective chi squared degrees of freedom ts > 0
    scale : float
      effective scaling of TS > 0
    """

    ################################
    # HISTOGRAM BACKGROUND ONLY TS #
    ################################

    # fix rounding errors
    trials['TS'][trials['TS'] < 0] = 0.

    # histogram
    y, bins = np.histogram(trials['TS'], np.arange(0., 20.01, .25))
    x = (bins[1:] + bins[:-1]) / 2
    w = (bins[1:] - bins[:-1])

    # see sensitivity_utils.py for definition of DeltaChiSquare()
    func = DeltaChiSquare()
    func.fit(x, y, w, verbose=True)

    # parameters describing TS > 0
    ndf = func.ndf
    eta = func.eta
    scale = func.scale

    # median TS for background only
    median_ts = np.percentile(trials['TS'], 50)

    ###########################
    # PLOT BACKGROUND ONLY TS #
    ###########################

    plt.bar(x - 0.5 * w, y, w, color='b', alpha=0.5, linewidth=0,
            label=("%d scrambles" % trials.size))
    plt.plot(x, trials.size * func.binval(x, w), color='r',
             linestyle='--', label="$\chi^{2}$ fit")

    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xlabel("TS", horizontalalignment='right', x=1.0)
    ax.set_ylabel("Number of Scrambles", horizontalalignment='right', y=1.0)
    ax.set_xlim(0, 20)
    ax.set_ylim(0.5, 1.1 * trials.size)

    plt.legend(loc=2)
    plt.savefig(image)
    plt.clf()

    print ("\nSaved " + image)

    return [median_ts, eta, ndf, scale]

# END fit_background()
