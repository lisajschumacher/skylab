# -*-coding:utf8-*-

r"""Example script to create data in the right format and load it correctly
to the LLH classes.

"""

from __future__ import print_function

# Python

# SciPy
import numpy as np

# skylab
from skylab.psLLH import PointSourceLLH, MultiPointSourceLLH
from skylab.ps_model import UniformLLH, EnergyLLH, PowerLawLLH
from skylab.ps_injector import PointSourceInjector
from skylab.priorllh import PriorLLH

mrs = np.radians(1.)
mrs_min = np.radians(0.05)
log_sig = 0.2
logE_res = 0.1

# fix seed to reproduce same results
np.random.seed(1)

def exp(N=100):
    r"""Create uniformly distributed data on sphere. """
    g = 3.7

    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float)])

    arr["ra"] = np.random.uniform(0., 2.*np.pi, N)
    arr["sinDec"] = np.random.uniform(-1., 1., N)

    E = np.log10(np.random.pareto(g, size=N) + 1)
    arr["sigma"] = np.random.lognormal(mean=np.log((mrs - mrs_min) * np.exp(-np.log(10)*E) + mrs_min),
                                       sigma=log_sig)
    arr["logE"] = E + logE_res * np.random.normal(size=N)

    return arr

def MC(N=1000):
    r"""Create uniformly distributed MC data on sphere. """
    g = 2.

    arr = np.empty((N, ), dtype=[("ra", np.float), ("sinDec", np.float),
                                 ("sigma", np.float), ("logE", np.float),
                                 ("trueRa", np.float), ("trueDec", np.float),
                                 ("trueE", np.float), ("ow", np.float)])

    # true information

    arr["trueRa"] = np.random.uniform(0., 2.*np.pi, N)
    arr["trueDec"] = np.arcsin(np.random.uniform(-1., 1., N))
    arr["trueE"] = np.random.pareto(g, size=N) + 1
    arr["ow"] = arr["trueE"]**(g)
    arr["ow"] /= arr["ow"].sum()

    eta = np.random.uniform(0., 2.*np.pi, len(arr))
    arr["sigma"] = np.random.lognormal(mean=np.log((mrs - mrs_min) * np.exp(-np.log(10)*np.log10(arr["trueE"])) + mrs_min),
                                       sigma=log_sig)
    arr["ra"] = arr["trueRa"] + np.cos(eta) * arr["sigma"] / np.cos(arr["trueDec"])
    arr["sinDec"] = np.sin(arr["trueDec"] + np.sin(eta) * arr["sigma"])
    arr["logE"] = np.log10(arr["trueE"]) + logE_res * np.random.normal(size=len(arr))

    return arr

def init(Nexp, NMC, energy=True, **kwargs):
    Nsrc = kwargs.pop("Nsrc", 0)
    fixed_gamma = kwargs.pop("fixed_gamma", False)
    fit_gamma = kwargs.pop("fit_gamma", 2.)

    arr_exp = exp(Nexp - Nsrc)
    arr_mc = MC(NMC)

    if Nsrc > 0:
        src_dec = kwargs.pop("src_dec", 0.)
        src_ra = kwargs.pop("src_ra", np.pi)
        gamma_inj = kwargs.pop("gamma_inj", 2.)
        print("Injecting Point source with {0} events, at (dec, ra)=({1:1.2f},{2:1.2f}) rad".format(Nsrc, src_dec, src_ra))
        print("Spectral index of the Source is {0:1.2f}".format(gamma_inj))
        print("This will be fitted with a fixed gamma of {0:1.2f}".format(fit_gamma))
        inj = PointSourceInjector(gamma_inj, sinDec_bandwidth=1, seed=0)
        inj.fill(src_dec, arr_mc, 333.)

        source = inj.sample(src_ra, Nsrc, poisson=False).next()[1]

        arr_exp = np.append(arr_exp, source)

    if energy and not fixed_gamma:
        """
        llh_model = PowerLawLLH(["logE"], min(50, Nexp // 50),
                                range=[[0.9 * arr_mc["logE"].min(),
                                        1.1 * arr_mc["logE"].max()]],
                                sinDec_bins=min(50, Nexp // 50),
                                sinDec_range=[-1., 1.],
                                bounds=(0, 5))
                                #"""
        llh_model = EnergyLLH(sinDec_bins=min(50, Nexp // 50),
                                sinDec_range=[-1., 1.],
                                bounds=(0, 5))
    elif fixed_gamma:
        llh_model = EnergyLLH(sinDec_bins=min(50, Nexp // 50),
                                sinDec_range=[-1., 1.],
                                bounds=(fit_gamma, fit_gamma))
    else:
        llh_model = UniformLLH(sinDec_bins=max(3, Nexp // 200),
                               sinDec_range=[-1., 1.])
    if fixed_gamma:
        llh = PriorLLH(arr_exp, arr_mc, 365., llh_model=llh_model,
                             mode="all", nsource=25, scramble=False,
                             nsource_bounds=(-Nexp / 2., Nexp / 2.)
                                            if not energy else (0., Nexp / 2.),
                             seed=np.random.randint(2**32),
                             **kwargs)
    else:
        llh = PointSourceLLH(arr_exp, arr_mc, 365., llh_model=llh_model,
                             mode="all", nsource=25, scramble=False,
                             nsource_bounds=(-Nexp / 2., Nexp / 2.)
                                            if not energy else (0., Nexp / 2.),
                             seed=np.random.randint(2**32),
                             **kwargs)

    return llh

def multi_init(n, Nexp, NMC, **kwargs):
    energy = kwargs.pop("energy", False)

    llh = MultiPointSourceLLH(nsource=25,
                              nsource_bounds=(-Nexp / 2., Nexp / 2.)
                                             if not energy else (0., Nexp / 2.),
                              seed=np.random.randint(2**32),
                              **kwargs)

    for i in xrange(n):
        llh.add_sample(str(i), init(Nexp, NMC, energy=energy))

    return llh


