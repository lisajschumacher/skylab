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
        Healpy map with nside = 2**self._nside_param;
        Used as prior for selecting source positions
    sinDec_range : tuple(float)
        Shrink allowed declination range.
    sinDec_bandwith : float
        Set to large value, all events are pre-selected for injection.
    src_dec : float
        Declination of source position
    e_range : tuple(float)
        Select events only in a certain energy range.
    random : RandomState
        Pseudo-random number generator

    """
    def __init__(self, gamma, template, nside_param, **kwargs):
        self._nside_param = np.int(nside_param)
        super(PriorInjector, self).__init__(gamma, **kwargs)
        self.sinDec_bandwidth = 10
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
        """
        # length should fit
        assert(len(templ) == hp.nside2npix(2**self._nside_param))
        # and the values should be between 0 and 1
        assert(min(templ)>=0)
        assert(max(templ)<=1)
        self._template = templ
