# -*-coding:utf8-*-

import os
import logging
import time
import cPickle as pickle
from socket import gethostname

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np

# skylab
from skylab.psLLH import MultiPointSourceLLH

# local
icdata=True # read real data or use toy data
if icdata:
    import ic_utils as utils
    basepath, inipath, savepath, crpath = utils.get_paths(gethostname())
else:
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
    extension = "_test_trials.png"
    
    plt = utils.plotting(backend=backend)
    nside = 2**7

    if icdata:
        llh, injector = utils.startup(basepath, inipath,
                                        multi=True, burn=False,
                                        fixed_gamma=True)
    else:
        llh, mc = utils.startup(multi=True, NN=1, n=2)
    num=30
    declinations = np.arcsin(np.linspace(-1., 1., num=num+1))
    declinations = (declinations[:-1]+declinations[1:])/2.
    print(llh)
    if not os.path.exists("figures/TS_trials"):
        os.makedirs("figures/TS_trials")
    n_iter=100
    start1 = time.time()
    for i,src_dec in enumerate(declinations):
        if i%5==1: print "Iteration", i, "of", num
        trials = llh.do_trials(src_ra=np.pi/2., src_dec=src_dec, mu=None, n_iter=n_iter)
        dcf = FitDeltaChi2()
        dc = dcf.fit(trials["TS"])
        n, bins, _ = plt.hist(trials["TS"], bins=20, normed=True)
        x = np.linspace(bins[0], bins[-1], 100)
        #~ plt.figure(i)
        plt.plot(x, dc.pdf(x))
        plt.semilogy(nonposy="clip")

        plt.savefig(("figures/TS_trials/logTS_{}"+extension).format(i))
        pickle.dump(dc, open("figures/TS_trials/logTS_{}_func.pickle".format(i), "wb"))
        plt.close("all")
    stop1 = time.time()
    mins, secs = divmod(stop1 - start1, 60)
    print("Trials finished after {0:2d}' {1:4.2f}''".format(int(mins), int(secs)))
    pickle.dump(declinations, open("figures/TS_trials/declinations.pickle", "wb"))
