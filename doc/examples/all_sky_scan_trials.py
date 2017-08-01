# -*-coding:utf8-*-

import os
import logging
import time
from socket import gethostname

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np

# skylab
from skylab.psLLH import MultiPointSourceLLH
from skylab.prior_generator import UhecrPriorGenerator
from skylab.prior_injector import PriorInjector

# local
# sorry but I had to change the names
# this is too confusing for me and the interpreter :D
import ic_utils as utils

level=logging.WARNING
logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(level)
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.StackingPriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.MultiStackingPriorLLH").setLevel(level)
logging.getLogger("skylab.prior_injector.PriorInjector").setLevel(level)

# convert test statistic to a p-value for a given point
ndof=1
#~ ndof=len(llh.params)
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(ndof).sf(TS)
                                             + chi2(ndof).cdf(-TS)))


basepath, inipath, savepath, crpath = utils.get_paths(gethostname())

if __name__=="__main__":
    
    hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
    hcolor = dict(North = "cyan", South = "red")

    nside_param = 4
    nside = 2**nside_param
    
    multi = True # work with multiple different samples
    # LLH and Fitting parameters
    fixed_gamma = True
    add_prior = True
    src_gamma = 2.
    fit_gamma = 2.

    # Other stuff
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 4
    else:
        ncpu = 1
    burn = True
    inj_seed = 666
    save_res = True
    identifier = "test_"

    # "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data"
    # "/home/lschumacher/git_repos/general_code_repo/data"
    pg = UhecrPriorGenerator(nside_param, np.radians(6), 100, crpath)
    tm = np.exp(pg.template)
    tm = tm/tm.sum(axis=1)[np.newaxis].T


    llh, injector = utils.startup(basepath,
                                    inipath,
                                    Nsrc=15,
                                    fixed_gamma=fixed_gamma,
                                    add_prior=add_prior,
                                    src_gamma=src_gamma,
                                    multi=multi,
                                    n_uhecr=pg.n_uhecr,
                                    prior=tm,
                                    nside_param=nside_param,
                                    burn=burn,
                                    inj_seed=inj_seed,
                                    ncpu=ncpu
                                    )
    #~ print(llh)
    # iterator of all-sky scan with follow up scans of most interesting points
    start1 = time.time()
    for i, (scan, hotspots) in enumerate(llh.all_sky_scan(
                                            nside=nside,
                                            follow_up_factor=1,
                                            pVal=pVal_func,
                                            hemispheres=hemispheres,
                                            prior=pg.template,
                                            fit_gamma=fit_gamma)
                                            ):

        if i > 0:
            # break after first follow up
            break

    stop1 = time.time()

    mins, secs = divmod(stop1 - start1, 60)
    print("Full scan finished after {0:2d}' {1:4.2f}''".format(int(mins), int(secs)))
    # Looking at the hotspots and separating them into North and South
    # There can only be one true source position, so we choose the larges TS
    # Trials will account for that, hopefully
    hk = hemispheres.keys()
    print "Hemisphere keys:", hk
    best_hotspots = np.zeros(pg.n_uhecr, dtype=[(p, np.float) for p in hk]
                                                +[("best", np.float)]
                                                +[("dec", np.float)]
                                                +[("ra", np.float)])

    for i,hi in enumerate(hotspots):
        for h in hk:
            best_hotspots[h][i] = hi[h]["best"]["TS"]
        if best_hotspots[hk[0]][i] >= best_hotspots[hk[1]][i]:
            best_hotspots["best"][i] = best_hotspots[hk[0]][i]
            best_hotspots["ra"][i] = hi[hk[0]]["best"]["ra"]
            best_hotspots["dec"][i] = hi[hk[0]]["best"]["dec"]
        else:
            best_hotspots["best"][i] = best_hotspots[hk[1]][i]
            best_hotspots["ra"][i] = hi[hk[1]]["best"]["ra"]
            best_hotspots["dec"][i] = hi[hk[1]]["best"]["dec"]
        
    print "Best:"
    print best_hotspots

    if save_res:
        import cPickle as pickle
        pickle.dump(hotspots, open(os.path.join(savepath, identifier+"hotspots.pickle"), "wb"))
    
