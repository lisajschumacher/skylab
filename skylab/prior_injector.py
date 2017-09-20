# -*-coding:utf8-*-

r"""This file is part of SkyLab

Skylab is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

@file    prior_injector.py
@authors Lisa Schumacher
@date    July, 2017
"""
from __future__ import division

import abc
import logging
from functools import partial
from collections import defaultdict

import numpy as np
import numpy.lib.recfunctions
import healpy as hp
import scipy.interpolate
# profiling
#~ from memory_profiler import profile

from skylab import ps_injector

class PriorInjector(ps_injector.PointSourceInjector):
    r""" Multiple point source injector with prior template

    The source's energy spectrum follows a power law.

    .. math::

        \frac{\mathrm{d}\Phi}{\mathrm{d}E} =
            \Phi_{0} E_{0}^{2 - \gamma}
            \left(\frac{E}{E_{0}}\right)^{-\gamma}

    By this definiton, the flux is equivalent to a power law with a
    spectral index of two at the normalization energy ``E0*GeV``GeV.
    The flux is given in units GeV^{\gamma - 1} s^{-1} cm^{-2}.

    Parameters
    ----------
    nside_param : int, required
        For Healpy map, such that it fits the template
        healpy.NSIDE = 2**nside_param
    seed : int, optional
        Random seed initializing the pseudo-random number generator.

    Attributes
    -----------
    gamma : float
        Spectral index; use positive values for falling spectrum.
    template : numpy.array
        Healpy map with self._nside;
        Used as prior for selecting source positions
    sinDec_range : tuple(float)
        Shrink allowed declination range.
    sinDec_bandwith : float
        Set to large value, all events are pre-selected for injection.
    e_range : tuple(float)
        Select events only in a certain energy range.
    random : RandomState
        Pseudo-random number generator

    """
    def __init__(self, gamma, template, nside_param, n_uhecr, **kwargs):
        assert(nside_param >= 0)
        assert(n_uhecr >= 1)
        self._n_uhecr = n_uhecr
        self._nside = 2**np.int(nside_param)
        super(PriorInjector, self).__init__(gamma, **kwargs)
        self.template = template
        
        
    @property
    def template(self):
        r""" Prior template of all selected UHECR events
        Translated to HealPy map scheme with certain nside parameter
        Sum of all entries normed to one
        """
        return self._template
    
    @template.setter
    def template(self, templ):
        r""" Prior template of all selected UHECR events
        Translated to HealPy map scheme with certain nside parameter
        Sum of all entries normed to one
        
        Make sure that basic prior template requirements are fulfilled
        
        Calculate additionally the sole declination dependence
        """
        # length should fit
        assert(np.shape(templ)[-1] == hp.nside2npix(self._nside))
        assert(len(templ) == self._n_uhecr)
        self._template = templ
        
    def _setup(self):
        r""" Reset solid angle.
        Declination range only determined by sinDec_range,
        no additional event selection
        """
        min_sinDec, max_sinDec = self.sinDec_range

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # Solid angle of selected events
        self._omega = 4. * np.pi

    #~ @profile
    def fill(self, mc, livetime):
        r""" Fill injector with Monte Carlo events, selecting events
        around the source position.

        Parameters
        -----------
        mc : ndarray, dict(enum, ndarray)
            Either structured array describing Monte Carlo events or a
            mapping of `enum` to such arrays
        livetime : float, dict(enum, float)
            Livetime per sample

        Raises
        ------
        TypeError
            If `mc` and `livetime` are not of the same type.

        See Also
        --------
        psLLH.MultiPointSourceLLH

        """
        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise TypeError("mc and livetime are not compatible.")

        # Reset solid angle and declination band.
        self._setup()

        dtype = [
            ("idx", np.int),
            ("enum", np.int),
            ("trueDec", np.float), 
            ("ow", np.float)
            ]
            
        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}
            
        mc_numbers = dict()
        N_tot = 0
        for key, mc_i in mc.iteritems():
            mc_numbers[key]=len(mc_i)
            N_tot += mc_numbers[key]
            
        self.mc_arr = np.empty(N_tot, dtype=dtype)

        self.mc = mc
            
        cur_index = 0
        for key in mc:
            
            N = mc_numbers[key]
            self.mc_arr["idx"][cur_index:cur_index+N] = np.arange(N)
            self.mc_arr["enum"][cur_index:cur_index+N] = key * np.ones(N)
            self.mc_arr["ow"][cur_index:cur_index+N] = self.mc[key]["ow"] * self.mc[key]["trueE"]**(-self.gamma) / self._omega
            self.mc_arr["trueDec"][cur_index:cur_index+N] = self.mc[key]["trueDec"]
            cur_index += N
            self._logging.info(
                "Sample {0}: selected {1:d} events".format(
                    key, N))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        self._logging.info("Selected {0:d} events in total.".format(
            len(self.mc_arr)))

        self._weights()
        
    def _weights(self):
        r""" Setup weights for assuming a power-law flux.
        """
        # Weights given in days; weighted to the point source flux
        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)
        self._norm_w = self.mc_arr["ow"] / self._raw_flux
        
        n, bins = np.histogram(np.sin(self.mc_arr["trueDec"]),
                               bins=100, 
                               weights=self._norm_w, 
                               range=(-1., 1.)
                              )
        x=(bins[1:] + bins[:-1]) / 2.
        y=n*1./np.max(n)
        #~ print(y)
        self._signal_acceptance = scipy.interpolate.InterpolatedUnivariateSpline(x, y, k=1)
        
        # Double-check if no weight is dominating the sample.
        if self._norm_w.max() > 0.1:
            self._logging.warn("Maximal weight exceeds 10%: {0:.2%}".format(
                self._norm_w.max()))

    #~ @profile
    def sample(self, mean_mu, poisson=True, position=False):
        r""" Sample events for given source location.

        Parameters
        -----------
        mean_mu : float
            Total mean number of events to sample
        poisson : bool, optional
            Use Poisson fluctuations, otherwise sample `mean_mu`.

        Returns
        --------
        num : array
            Number of events per source
        sam_ev : iterator
            Sampled events for each loop iteration; either as simple
            array or as dictionary for each sample

        """
        while True:
                            
            src_dec, src_ra = self._get_source_positions(self._n_uhecr)
            self._src_dec = src_dec
            self._src_ra = src_ra
            self._logging.info("Injecting sources at ra = {} deg".format(np.degrees(src_ra))
                                +" and dec = {} deg".format(np.degrees(src_dec)))
            acceptance_weighting = self._signal_acceptance(np.sin(src_dec))
            
            # Generate event numbers using Poisson events.
            if poisson:
                num = self.random.poisson(mean_mu * acceptance_weighting,
                        size=self._n_uhecr)
            else:
                num = np.array(np.around(mean_mu * acceptance_weighting),
                        dtype=np.int)
                
            num_sum = np.sum(num, dtype=int)
            self._logging.info("Mean number of events {0:.1f}".format(mean_mu))
            self._logging.info("Generated number of events {}".format(num))

            if num_sum < 1:
                # No events will be sampled.
                yield num_sum, None
                continue

            sam_ev = defaultdict(partial(np.ndarray, 
                                         0, 
                                         dtype=[('ra', '<f8'), 
                                                ('logE', '<f8'), 
                                                ('sigma', '<f8'), 
                                                ('sinDec', '<f8')
                                               ]
                                        )
                                )
            for i,(s_dec,s_ra) in enumerate(zip(src_dec,src_ra)):
                min_dec, max_dec = self._get_dec_band(s_dec)
                band = np.logical_and(
                            np.sin(self.mc_arr["trueDec"]) > np.sin(min_dec),
                            np.sin(self.mc_arr["trueDec"]) < np.sin(max_dec))

                sam_idx = self.random.choice(self.mc_arr[band], 
                                             size=num[i], 
                                             p=self._norm_w[band]/np.sum(self._norm_w[band])
                                            )

                # Get the events that were sampled.
                enums = np.unique(sam_idx["enum"])

                if len(enums) == 1 and enums[0] < 0:
                    # Only one event will be sampled.
                    sam_ev_i = np.copy(self.mc[enums[0]][sam_idx["idx"]])
                    sam_ev[enums[0]] = np.append(sam_ev[enums[0]],
                                        ps_injector.rotate_struct(sam_ev_i, s_ra, s_dec))
                    continue

                for enum in enums:
                    idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                    sam_ev_i = np.copy(self.mc[enum][idx])
                    sam_ev[enum] = np.append(sam_ev[enum],
                                    ps_injector.rotate_struct(sam_ev_i, s_ra, s_dec))

            if position:
                yield num, sam_ev, src_ra, src_dec
            else:
                yield num, sam_ev

    #~ @profile
    def _get_source_positions(self, n):
        r""" Draw n source positions with (dec,ra) from the template map
        """
        dec = np.empty(n)
        ra = np.empty(n) 
        for i in xrange(n):
            pix = np.random.choice(np.arange(hp.nside2npix(self._nside)),  
                                   p=self._template[i])
            theta, phi = hp.pix2ang(self._nside, pix)
            dec[i] = np.pi/2. - theta
            ra[i] = phi
        return dec, ra
    
    def _get_dec_band(self, src_dec):
        r""" Get declination band aroudn src_dec position
        """
        A, B = self.sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(src_dec) + b

        min_sinDec = max(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = min(B, sinDec + self.sinDec_bandwidth)
        return np.arcsin(min_sinDec), np.arcsin(max_sinDec)

# Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skylab.prior_generator import UhecrPriorGenerator
    from test_utils import startup, cmap
    
    fixed_gamma = True
    add_prior = True
    llh, mc = startup(n=3, NN=4, multi=True, fixed_gamma=fixed_gamma, add_prior=add_prior)
    lt = dict([(i, 350.+np.random.uniform(-10, +10)) for i in range(len(llh))])
    nside_param = 6
    # "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data"
    # "/home/lschumacher/git_repos/general_code_repo/data"
    pg = UhecrPriorGenerator(nside_param,
                            np.radians(6),
                            120,
                            "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data")
    tm = np.exp(pg.template)
    tm = tm/tm.sum(axis=1)[np.newaxis].T
    injector = PriorInjector(2.,
                            tm,
                            n_uhecr=pg.n_uhecr,
                            nside_param=nside_param)
    injector.fill(mc, lt)
    sampler = injector.sample(20, poisson=False)

    num, sam = sampler.next()

    sindec_d = np.empty(np.sum(num))
    ra_d = np.empty(np.sum(num))
    c = 0
    for k,s in sam.iteritems():
        dc = len(s['sinDec'])
        sindec_d[c:c+dc] = s['sinDec']
        ra_d[c:c+dc] = s['ra']
        c+=dc

    hres = np.zeros(hp.nside2npix(2**nside_param))
    pix = hp.ang2pix(2**nside_param, np.pi/2. - np.arcsin(sindec_d), ra_d)
    for p in pix:
        hres[p]+=1
    plt.figure(1)
    cmap.set_under("w")
    hp.mollview(hp.smoothing(hres, sigma=np.radians(0.5)), cmap=cmap, fig=1, rot=[180,0,0])
    hp.projtext(np.pi/2, 0.01, r"$0^\circ$", color="w", ha="right")
    hp.projtext(np.pi/2, -0.01, r"$360^\circ$", color="w")
    hp.projtext(np.pi/2, np.pi, r"$180^\circ$", color="w")
    path = "/home/home2/institut_3b/lschumacher/phd_stuff/skylab_git/"
    plt.savefig(path + "figures/test_injection.png")

    fig = plt.figure(2)
    tm = tm.sum(axis=0)
    hp.mollview(tm, fig=-1, cmap=cmap, rot=[180,0,0])
    hp.projtext(np.pi/2, 0.01, r"$0^\circ$", color="w", ha="right")
    hp.projtext(np.pi/2, -0.01, r"$360^\circ$", color="w")
    hp.projtext(np.pi/2, np.pi, r"$180^\circ$", color="w")
    hp.projscatter(np.pi/2. - injector._src_dec, injector._src_ra, 20,
                       marker="x",
                       color="magenta",
                       alpha=0.5)
    plt.savefig(path+"figures/test_injection_template.png")
    
