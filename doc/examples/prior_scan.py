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

logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(logging.INFO)
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(logging.WARNING)

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))

label = dict(TS=r"$\mathcal{TS}$",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
             )


if __name__=="__main__":

    plt = utils.plotting(backend="pdf")
    # This sets whether or not we choose the template fit with fixed gamma
    fixed_gamma=True
    fit_gamma = 2.
    
    # Source parameters for injection
    src_sigma = np.radians(6.)
    src_dec = 0.
    src_gamma = 2.
    
    # We want to test several separations of source and prior
    # Up to 5 sigma for now
    N = 6
    src_ra_arr = np.arange(N) * src_sigma + np.pi

    for j, src_ra in enumerate(src_ra_arr):
        info_string = "deflection"+str(j)+"_"
        
        # The source moves now in R.A.
        llh, mc = utils.startup(Nsrc=10, fixed_gamma=fixed_gamma,
                            gamma_inj=src_gamma,
                            src_dec=src_dec, src_ra=src_ra
                            )

        # ... but we fit always the same prior
        # iterator of all-sky scan with follow-up scans of most interesting points
        for i, (scan, hotspot) in enumerate(llh.all_sky_scan(
                                    nside=2**6, follow_up_factor=1,
                                    pVal=pVal_func,
                                    hemispheres=dict(Full=np.radians([-90., 90.])),
                                    pdec=src_dec,
                                    pra=src_ra_arr[0],
                                    psig=src_sigma,
                                    fit_gamma=fit_gamma)
                                    ):

            if i > 0:
                # break after first follow up
                break
        print("Found smallest P-Value: {:.2f}".format(hotspot["Full"]["best"]["pVal"]))
        print("at (ra, dec) = ({:.2f}, {:.2f}) deg".format(np.degrees(hotspot["Full"]["best"]["ra"]),
                                                    np.degrees(hotspot["Full"]["best"]["dec"])))
        for k in scan.dtype.names:
            scan[k] = hp.sphtfunc.smoothing(scan[k], sigma=np.radians(0.5))

        eps = 1.

        # Custom colormap using cubehelix from seaborn, see utils
        cmap = utils.cmap

        fig, ax = utils.skymap(plt, scan["pVal"], cmap=cmap,
                               vmin=0., vmax=np.ceil(hotspot["Full"]["best"]["pVal"]),
                               colorbar=dict(title=r"$-\log_{10}\rm p$"),
                               rasterized=True)
        
        if not os.path.exists("figures"):
            os.makedirs("figures")

        fig.savefig("figures/"+info_string+"skymap_pVal.pdf", dpi=256)

        if fixed_gamma:
            fig, ax = utils.skymap(plt, scan['prior'], cmap=cmap,
                                   rasterized=True)

            fig.savefig("figures/"+info_string+"prior.pdf", dpi=256)

        for key in ["TS"] + llh.params:
            if fixed_gamma and key=="gamma": continue
            eps = 0.1 if key != "TS" else 0.0
            vmin, vmax = np.percentile(scan[key], [eps, 100. - eps])
            vmin = np.floor(max(0, vmin))
            vmax = min(8, np.ceil(vmax))
            q = np.ma.masked_array(scan[key])
            q.mask = ~(scan["nsources"] > 0.5) if key != "TS" else np.zeros_like(q, dtype=np.bool)
            fig, ax = utils.skymap(plt, q, cmap=cmap,
                                   vmin=vmin, vmax=vmax,
                                   colorbar=dict(title=label[key]),
                                   rasterized=True)

            fig.savefig("figures/"+info_string+"skymap_" + key +".pdf", dpi=256)
