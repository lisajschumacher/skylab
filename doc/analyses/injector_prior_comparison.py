import matplotlib
matplotlib.use('Agg')

from os.path import join
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
    
import getpass
username = getpass.getuser()

import time
import numpy as np
import healpy as hp
from scipy.stats import describe

from skylab.datasets import Datasets
from skylab.ps_injector import PriorInjector, PointSourceInjector
from skylab.priors import SpatialPrior

# vector calculations
from astropy.coordinates import UnitSphericalRepresentation
from astropy.coordinates import Angle
from astropy import units as u

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("nside", type=int, 
                    help="nside parameter for healpy -> 2**nside")
parser.add_argument("nmap", type=int, 
                    help="number of maps for averaging")
parser.add_argument("psize", type=float, 
                    help="prior size in degree")
parser.add_argument('--savefigs', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--flux', action="store_true")

args = parser.parse_args()

savefigs = bool(args.savefigs)

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except:
    print("Could not load seaborn")
    
if username=="lschumacher":
    savefigs = True
    figurepath = "/home/lschumacher/public_html/uhecr_stacking/hotspot_fit/test_plots/"
    savepath = "/data/user/lschumacher/projects/stacking/hotspot_fitting/"

def make_array_1d(array):
    array = np.atleast_1d(array)
    array = array.flatten()
    return array

def generate_prior_maps(ra, dec, sigma, nside):
    """
    Make simple 2D priors
    """
    assert(ra.shape==dec.shape)
    assert(ra.shape==sigma.shape)
    ra = make_array_1d(ra)
    dec = make_array_1d(dec)
    sigma = make_array_1d(sigma)
    
    theta_map, ra_map = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    dec_map = np.pi/2. - theta_map

    prior_maps = np.empty((len(ra), hp.nside2npix(nside)), dtype=np.float)

    for i in range(len(ra)):
        mean_vec = UnitSphericalRepresentation(Angle(ra[i], u.radian), 
                                               Angle(dec[i], u.radian))
        map_vec = UnitSphericalRepresentation(Angle(ra_map, u.radian),
                                              Angle(dec_map, u.radian))

        prior_maps[i] = np.exp(-1.*np.power((map_vec-mean_vec).norm(), 2) / sigma[i]**2 / 2.)
    return prior_maps

def skymap(plt, vals, **kwargs):
    fig, ax = plt.subplots(subplot_kw=dict(projection="aitoff"))
    gridsize = 1000

    x = np.linspace(np.pi, -np.pi, 2 * gridsize)
    y = np.linspace(np.pi, 0., gridsize)

    X, Y = np.meshgrid(x, y)

    r = hp.rotator.Rotator(rot=(-180., 0., 0.))

    YY, XX = r(Y.ravel(), X.ravel())

    pix = hp.ang2pix(hp.npix2nside(len(vals)), YY, XX)

    Z = np.reshape(vals[pix], X.shape)

    lon = x[::-1]
    lat = np.pi /2.  - y

    cb = kwargs.pop("colorbar", dict())
    plot_cb = kwargs.pop("plot_cb", True)
    cb.setdefault("orientation", "horizontal")
    cb.setdefault("fraction", 0.075)

    title = cb.pop("title", None)

    p = ax.pcolormesh(lon, lat, Z, **kwargs)
    plt.hlines(np.radians(-5.), -np.pi, np.pi, 
               color="gray", alpha=0.75, linestyle="--", lw=1)

    cbar = fig.colorbar(p, **cb)
    cbar.solids.set_edgecolor("face")
    cbar.update_ticks()
    if title is not None:
        cbar.set_label(title)
    if not plot_cb: cbar.remove()
    ax.xaxis.set_ticks([])
    plt.text(0,0, r"$180^\circ$", horizontalalignment='center')
    plt.text(np.pi+0.1, 0, r"$0^\circ$", horizontalalignment='left')

    return fig, ax

def generate_signal_map(prinj, mean_signal, nside, 
                        flux=False, poisson=True, 
                        verbose=False):
    """
    Generate healpy map with neutrino sources from injector sampler
    
    Parameters:
    prinj : instance of PriorInjector
    mean_signal: int, max number of signal events, NOT per source but TOTAL
    nside : int, parameter for healpy map size, should match sampler's
    in order to not introduce any bias
    flux: bool, do or do not return map with flux instead of bare event counts
    
    Returns:
    hres: healpy map with nside**2*12 pixels, 
    bins +=1 where events where injected (flux=False)
    or flux per event (flux=True)
    """
    
    num, sam, src_ra, src_dec = prinj.sample(mean_signal, poisson=poisson, debug=True)
    #print src_ra, src_dec
    num_sum = num.sum()
    num_per_source = num.sum(axis=-1)
    num_per_sample = num.sum(axis=0)
    #print num
    #print num_per_source
    hres = np.zeros(hp.nside2npix(nside))
    pix = hp.ang2pix(nside, np.pi/2. - src_dec, src_ra)
    # calculating flux per source
    flux_val = prinj.mu2flux(mean_signal*1./len(num))
    for i,p in enumerate(pix):
        if flux:
            # normalize by event number
            # such that the histogram shows
            # flux contribution per event in each bin
            # 8-)
            hres[p] += flux_val #/num_per_source[src_num_idx[i]]
        else:
            hres[p] += num_per_source[i]
    return hres

def generate_multiple_signal_maps(nsig, **kwargs):
    """
    Generate multiple healpy maps from function generate_signal_map()
    
    Parameters:
    nsig : number of signal maps to be generated
    
    kwargs:
    passed to generate_signal_map(**kwargs)
    
    returns:
    hres: combined healpy map of all signal injections
    """
    nside = kwargs["nside"]
    hres = np.zeros(hp.nside2npix(nside))
    for n in xrange(nsig):
        hres += generate_signal_map(**kwargs)
    return hres*1./nsig

exp, mc, livetime = Datasets['PointSourceTracks'].season("IC86, 2012")
# choose a 10% subset to reduce computing timei
if args.test:
    rs = np.random.RandomState(1)
    exp = rs.choice(exp, len(exp)/10)
    mc = rs.choice(mc, len(mc)/10)
# use dicts if you want to test multi-year
exps = {1:exp, 0:exp}
mcs = {1:mc, 0:mc}
livetimes = {1:livetime, 0:livetime}

nside = 2**args.nside
# uniform prior map
# p = np.ones(hp.nside2npix(nside), float)
pix = np.array([50, 250, 500, 750])*((nside*1./8)**2)
pix = np.array(pix, dtype=np.int)
theta, ra = hp.pix2ang(nside, pix)
theta = np.concatenate([theta, [np.radians(94)]])
ra = np.concatenate([ra, [np.pi]])

dec = np.pi/2. - theta
#print ra, dec
sigma = np.full_like(ra, fill_value=np.radians(args.psize))

p = generate_prior_maps(ra, dec, sigma, nside)
prior = SpatialPrior(p)


print("\n prior injector setup:")
t0 = time.time()
prinj = PriorInjector(prior, gamma=2., seed=2) # this gamma is what I use
prinj.fill(exps, mcs, livetimes)
print("  - took %.2f sec" % (time.time()-t0))
print("raw_flux conversion factor: {:1.2e}".format(prinj._raw_flux))
### single injection and plotting
mean_signal = 1000
single_maps_raw = generate_signal_map(prinj, 
                                        mean_signal*len(prior.p), 
                                        nside, 
                                        flux=args.flux, 
                                        poisson=False, 
                                        verbose=True
                                       )
### Mask all zero values to see where actually events have been injected
fig, ax = skymap(plt, 
                 np.where(single_maps_raw>0, 
                          single_maps_raw, 
                          np.ones_like(single_maps_raw)*np.nan), 
                 cmap=plt.cm.viridis)

if savefigs: 
    plt.savefig(join(figurepath, "test_single_injection.png"))
    plt.close("all")

### multiple injections and plotting
### smooth with prior to compare to pure prior map
multi_maps = generate_multiple_signal_maps(args.nmap, 
                                         prinj=prinj, 
                                         mean_signal=mean_signal*len(prior.p), 
                                         nside=nside, 
                                         flux=args.flux
                                        )
if username=="lschumacher":
    np.savetxt(join(savepath, "multi_maps.npy"), multi_maps)
    
fig, ax = skymap(plt, 
                 np.where(multi_maps>0, 
                          multi_maps, 
                          np.ones_like(single_maps_raw)*np.nan), 
                 cmap=plt.cm.viridis)

if savefigs: 
    plt.savefig(join(figurepath, "test_multiple_injection.png"))
    plt.close("all")
"""
# prior map and plotting
all_prior = prior.p.sum(axis=0)
fig, ax = skymap(plt, all_prior, cmap=plt.cm.viridis)
if savefigs: 
    plt.savefig(join(figurepath, "test_multiple_prior.png"))
    plt.close("all")

# difference of normalized prior map and injection maps
# divided by std of injection maps to get a handle on order-of-magnitude deviation
mask = multi_maps<=0 #min(all_prior)
m1 = np.ma.array(all_prior, mask=mask)
m1 = m1*1./m1.sum()
m2 = np.ma.array(multi_maps, mask=mask)
m2 = m2*1./m2.sum()
diff_map = (m1-m2)*1./m2.std()
fig, ax = skymap(plt, diff_map, cmap=plt.cm.viridis)
if savefigs: 
    plt.savefig(join(figurepath, "test_injector_prior_residual.png"))
    plt.close("all")

    
try: 
    import seaborn as sns
    sea = True
except:
    print("could not load seaborn, use pyplot instead")
    sea = False
if sea:
    sns.set_style("whitegrid")
    sns.distplot(diff_map, kde=False)
    plt.semilogy(nonposy="clip")
else:    
    ax = plt.hist(diff_map, bins=25)
    plt.semilogy(nonposy="clip")

if savefigs: 
    plt.savefig(join(figurepath, "injector_prior_residual_hist.png"))
    plt.close("all")

res = describe(diff_map[abs(diff_map)>0.085])
print("statistics describing difference between prior map and injection map:")
print(res)"""