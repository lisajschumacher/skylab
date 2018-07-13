
import time
import numpy as np
import healpy as hp

from skylab.datasets import Datasets
from skylab.ps_injector import PriorInjector, PointSourceInjector
from skylab.priors import SpatialPrior

# uniform prior map
nside = 8
p = np.ones(hp.nside2npix(nside), float)
prior = SpatialPrior(p)

exp, mc, livetime = Datasets['PointSourceTracks'].season("IC86, 2012")

# use dicts if you want to test multi-year
exps = {1:exp, 2:exp}
mcs = {1:mc, 2:mc}
livetimes = {1:livetime, 2:livetime}

print("\n prior injector setup:")
t0 = time.time()
prinj = PriorInjector(prior, seed=2)
prinj.fill(exp, mc, livetime)
print("  - took %.2f sec" % (time.time()-t0))

for pix in [0, 250, 500, 750]:

    print("\n" + 80*"=")
    mean_signal = 10.

    # update prior map with single, non-zero pixel
    p = np.zeros(hp.nside2npix(nside), float)
    p[pix] = 1.
    prinj.prior = SpatialPrior(p)

    theta, phi = hp.pix2ang(nside, pix)
    ra  = phi
    dec = np.pi/2. - theta
    
    print("\n spatial prior:")
    print("  - single location (ra %.2f deg, dec %.2f deg)" %
          tuple(np.degrees([ra, dec])))

    nis = []
    for i in range(1000):
        ni, sample = prinj.sample(mean_signal)
        nis.append(ni)

    avg_ni = sum(nis) / float(len(nis))
    print("\n prior injector:")
    print("  - mean_signal %.2f" % mean_signal)
    print("  - avg n_inj   %.2f" % avg_ni)
    print("  - mu2flux     %.2e [GeV-1cm-2s-1]" % prinj.mu2flux(mean_signal))
    print("  - E0          %.2f GeV" % prinj.E0)

    # comparison to standard injector. Note that the avg_ni should be
    # used for mean_signal in the case of standard injector because
    # the standard case doesn't need to account for acceptance factors.
    inj = PointSourceInjector()
    inj.fill(dec, exp, mc, livetime)
    print("\n standard injector:")
    print("  - mean_signal %.2f" % avg_ni)
    print("  - mu2flux     %.2e [GeV-1cm-2s-1]" % inj.mu2flux(avg_ni))
    print("  - E0          %.2f GeV" % inj.E0)

    rel_diff = 100*(prinj.mu2flux(mean_signal) - inj.mu2flux(avg_ni)) / inj.mu2flux(avg_ni)
    print("\n relative difference: %.2f%%\n" % rel_diff)

