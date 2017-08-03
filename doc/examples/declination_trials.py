# -*-coding:utf8-*-

import os
import logging

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np

# skylab
from skylab.psLLH import MultiPointSourceLLH

# local
import utils
from skylab.utils import FitDeltaChi2

logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(logging.INFO)
logging.getLogger("skylab.psLLH.MultiPointSourceLLH").setLevel(logging.INFO)

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))

label = dict(TS=r"$\mathcal{TS}$",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
             )

if __name__=="__main__":
    
    backend = "svg"
    extension = "_d_trials.png"
    
    plt = utils.plotting(backend=backend)
    nside = 2**7
        
    llh, mc = utils.startup(multi=True, NN=1, n=2) 
    #~ print(llh)
    n_iter=1000
    trials = llh.do_trials(src_ra=np.pi/2., src_dec=np.pi/4., mu=None, n_iter=n_iter)
    dcf = FitDeltaChi2()
    dc = dcf.fit(trials["TS"])
    n, bins, _ = plt.hist(trials["TS"], bins=20, normed=True)
    x = np.linspace(bins[0], bins[-1], 100)
    plt.plot(x, dc.pdf(x)/n_iter)
    plt.savefig("figures/TS"+extension)
    plt.semilogy(nonposy="clip")

    plt.savefig("figures/logTS"+extension)
    plt.close("all")
    
