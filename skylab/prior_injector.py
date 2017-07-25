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

from . import utils
from . import ps_injector

class PriorInjector(ps_injector.PointSourceInjector):
    r"""Multiple point source injector with prior template

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
        #self.sinDec_bandwidth = 10
        self.template = template
        
        
    @property
    def template(self):
        r"""
        Prior template of all selected UHECR events
        Translated to HealPy map scheme with certain nside parameter
        Sum of all entries normed to one
        """
        return self._template
    
    @template.setter
    def template(self, templ):
        r"""
        Prior template of all selected UHECR events
        Translated to HealPy map scheme with certain nside parameter
        Sum of all entries normed to one
        
        Make sure that basic prior template requirements are fulfilled
        
        Calculate additionally the sole declination dependence
        """
        # length should fit
        assert(len(templ) == hp.nside2npix(self._nside))
        # and the values should be between 0 and 1
        assert(min(templ)>=0)
        assert(max(templ)<=1)
        self._template = templ
        
    def _setup(self):
        r"""Reset solid angle.
        Declination range only determined by sinDec_range,
        no additional event selection
        """
        min_sinDec, max_sinDec = self.sinDec_range

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # Solid angle of selected events
        self._omega = 4. * np.pi
        
    def fill(self, mc, livetime):
        r"""Fill injector with Monte Carlo events, selecting events
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
            ("idx", np.int), ("enum", np.int),
            ("trueE", np.float), ("trueDec", np.float), 
            ("ow", np.float)
            ]

        self.mc_arr = np.empty(0, dtype=dtype)

        self.mc = dict()

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # Get MC events in the selected energy and sine declination range.

            band_mask = np.logical_and(
                mc_i["trueE"] / self.GeV > self.e_range[0],
                mc_i["trueE"] / self.GeV < self.e_range[1])

            if not np.any(band_mask):
                self._logging.warn(
                    "Sample {0:d}: no events were selected.".format(key))

                self.mc[key] = mc_i[band_mask]

                continue

            self.mc[key] = mc_i[band_mask]

            N = np.count_nonzero(band_mask)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            mc_arr["ow"] = self.mc[key]["ow"] * livetime[key] * 86400.
            mc_arr["trueE"] = self.mc[key]["trueE"]
            mc_arr["trueDec"] = self.mc[key]["trueDec"]

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            self._logging.info(
                "Sample {0}: selected {1:d} events".format(
                    key, N))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        self._logging.info("Selected {0:d} events in total.".format(
            len(self.mc_arr)))

        self._weights()
        
    def _weights(self):
        r"""Setup weights for assuming a power-law flux.

        """
        # Weights given in days; weighted to the point source flux
        self.mc_arr["ow"] *= self.mc_arr["trueE"]**(-self.gamma) / self._omega
        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)
        self._norm_w = self.mc_arr["ow"] / self._raw_flux
        
        n, bins = np.histogram(self.mc_arr["trueDec"],
                               bins=100, 
                               weights=self._norm_w, 
                               range=(-np.pi/2., np.pi/2.)
                              )
        self._signal_acceptance = scipy.interpolate.InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2.,
                                    n/np.max(n))
        
        # Double-check if no weight is dominating the sample.
        if self._norm_w.max() > 0.1:
            self._logging.warn("Maximal weight exceeds 10%: {0:.2%}".format(
                self._norm_w.max()))

    def sample(self, mean_mu, poisson=True):
        r"""Sample events for given source location.

        Parameters
        -----------
        mean_mu : float
            Total mean number of events to sample
        poisson : bool, optional
            Use Poisson fluctuations, otherwise sample `mean_mu`.

        Returns
        --------
        num : int
            Total number of events
        sam_ev : iterator
            Sampled events for each loop iteration; either as simple
            array or as dictionary for each sample

        """
        while True:
                            
            src_dec, src_ra = self._get_source_positions(self._n_uhecr)
            acceptance_weighting = self._signal_acceptance(src_dec)
            
            # Generate event numbers using Poisson events.
            if poisson:
                num = self.random.poisson(mean_mu * acceptance_weighting,
                        size=self._n_uhecr)
            else:
                num = np.array(np.around(mean_mu * acceptance_weighting),
                        dtype=np.int)
                
            num_sum = np.sum(num, dtype=int)
            self._logging.info("Mean number of events {0:.1f}".format(mean_mu))
            self._logging.info("Generated number of events {0:d}".format(num_sum))

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

            yield num, sam_ev
                        
    def _get_source_positions(self, n):
        r""" Draw n source positions with (dec,ra) from the template map
        """
        pix = np.random.choice(np.arange(hp.nside2npix(self._nside)), 
                               size=n, 
                               p=self._template, 
                               replace=True)
        theta, ra = hp.pix2ang(self._nside, pix)
        return np.pi/2.-theta, ra
    
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
