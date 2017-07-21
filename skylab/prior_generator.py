# -*-coding:utf8-*-

from __future__ import print_function

"""
This file is part of SkyLab

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

@file    prior_generator.py
@authors Lisa Schumacher
@date    July, 2017
"""

# python packages
import sys
import os
import abc

# General calculations
import healpy as hp
import numpy as np

# Vector calculations
from astropy.coordinates import UnitSphericalRepresentation
from astropy.coordinates import Angle
from astropy import units as u


class PriorGenerator(object):
    r"""
    PriorGenerator builds sky templates based on HealPy maps.
    The templates can be used as priors for 
    - PriorLLH
    - StackingPriorLLH
    
    and can be used for event generation, like
    - PriorInjector
    - ...
    """
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, nside_param):
        self.nside = 2**nside_param
        theta, ra = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        self.dec = np.pi/2. - theta
        self.ra = ra
        
    @abc.abstractproperty
    def template(self):
        r"""
        Return the healpy array representing the prior template
        Normed and shifted such that the maximum is 1 and the minimum is larger than zero
        """
        return np.zeros(hp.nside2npix(self.nside))

class UhecrPriorGenerator(PriorGenerator):
    r"""
    UhecrPriorGenerator builds a prior template from UHECR data
    
    Parameters:
        nside:
        data_path:
        deflection: in radian
        energy_threshold: in EeV
    """
    def __init__(self, nside_param, deflection, energy_threshold, data_path, multi=False):
        r"""
        Initialize and calculate the Prior template
        
        Parameters:
            nside_param : int
            Sets Healpy parameter nside = 2**nside_param
            
            deflection : float
            Magnetic deflection parameter in radian
            
            energy_threshold : float
            Energy threshold for UHECR selection, should be between min and max of expected values
            
            data_path : string
            Path name where to find UHECR data
        """
        super(UhecrPriorGenerator, self).__init__(nside_param)
        self._template = self.calc_template(self._get_UHECR_positions(deflection, energy_threshold, data_path), multi)        
        
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
        assert(len(templ) == len(self._template))
        # and the values should be between 0 and 1
        assert(min(templ)>=0)
        assert(max(templ)<=1)
        self._template = templ

    
    def calc_template(self, params, multi):
        r"""
        Calculate the Prior template from ra, dec and sigma parameters.
        The single templates are constructed as 2D-symmetric Gaussians 
        on position (ra, dec) with width of sigma. All contributions are
        added up to build the complete Prior template or
        TO DO: LEAVE THEM AS SINGLE PRIORS
        
        Using 2D-Vectors on a sphere
        
        Parameters:
            params: combined array of ra, dec and sigma
                Can be single float values each or 1D-arrays themselves
                Should have the same length
            multi: bool
                whether or not this metho generates one combined prior or
                multiple single priors
                
        returns:
            Prior Template in HealPy map format, normalized to 1
        """
        t_ra, t_dec, t_sigma = params
        
        t_ra = np.atleast_1d(t_ra)
        t_dec = np.atleast_1d(t_dec)
        t_sigma = np.atleast_1d(t_sigma)
        assert(len(t_ra) == len(t_dec))
        assert(len(t_ra) == len(t_sigma))

        if multi:
            _template = np.empty((len(t_ra), hp.nside2npix(self.nside)), dtype=np.float)
        else:
            _template = super(UhecrPriorGenerator, self).template
            
        for i in range(len(t_ra)):
            mean_vec = UnitSphericalRepresentation(Angle(t_ra[i], u.radian), 
                                                   Angle(t_dec[i], u.radian))
            map_vec = UnitSphericalRepresentation(Angle(self.ra, u.radian),
                                                  Angle(self.dec, u.radian))
            if multi:
                _template[i] = np.exp(-1.*np.power((map_vec-mean_vec).norm(), 2) / t_sigma[i]**2)
            else:
                _template += np.array(np.exp(-1.*np.power((map_vec-mean_vec).norm(), 2) / t_sigma[i]**2)
                            / 4. / np.pi / t_sigma[i]**2 )
                # make it normalized
                _template /= np.sum(_template)
        return _template

    def _get_UHECR_positions(self, deflection, energy_threshold, data_path,
                             files = ["AugerUHECR2014.txt", "TelArrayUHECR.txt"]):
        """ 
        Read the UHECR text file(s)
        Parameters:
            deflection : float
                        parameter for source extent, called "D" in paper
                        usually 3 or 6 degree (but plz give in radian k?)
            energy_threshold : float
                         threshold value for energy given in EeV

            data_path : Where to find the CR data
                    the data sets can be downloaded from
                    https://wiki.icecube.wisc.edu/auger/index.php/Auger_Event_List
                    https://wiki.icecube.wisc.edu/auger/index.php/TA_Event_List

        Optional:
            files : list of strings
                    filenames to be found in data_path that hold CR data
                    These should be readable with numpy.genfromtext
                    and have names ["dec", "RA", "E"]

        returns : three arrays
        ra, dec, sigma (i.e. assumed magnetic deflection)
        """

        # This could be done probably better with standard file reading ...
        dec_temp = []
        ra_temp = []
        e_temp = []
        sigma_reco = []

        for f in files:
            data = np.genfromtxt(os.path.join(data_path, f), names=True)
            dec_temp.extend(np.radians(data['dec']))
            ra_temp.extend(np.radians(data['RA']))
            if f=="TelArrayUHECR.txt":
                # Implement energy shift of 13%, see paper
                e_temp.extend(data['E']*(1.-0.13))
                sigma_reco.extend(len(data)*[np.radians(1.5)])
            else:
                e_temp.extend(data['E'])
                sigma_reco.extend(len(data)*[np.radians(0.9)])

        # Check the threshold
        if energy_threshold > max(e_temp):
            default = 85.
            print("Energy threshold {:1.2f} too large, max value {:1.2f}! Set threshold to {:1.2f} EeV".format(energy_threshold, max(e_temp), default))
            energy_threshold=default

        # Select events above energy threshold
        e_mask = np.where(np.array(e_temp)>=energy_threshold)[0]

        dec = np.array(dec_temp)[e_mask]
        ra = np.array(ra_temp)[e_mask]

        # set source extent by formula sigma_CR = D*100EeV/E_CR
        mag_deflection = deflection*100./np.array(e_temp)[e_mask]
        sigma_reco = np.array(sigma_reco)[e_mask]
        sigma = np.sqrt(mag_deflection**2 + sigma_reco**2)
        return ra, dec, sigma

# Testing
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from seaborn import cubehelix_palette
    cmap = cubehelix_palette(as_cmap=True, start=0.2, rot=0.9, dark=0., light=0.9, reverse=True, hue=1)
    t = UhecrPriorGenerator(6, np.radians(6), 125, multi=True, data_path="/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data")


    for i,tm in enumerate(t.template):
        fig = plt.figure(i)
        hp.mollview(tm, fig=i, cmap=cmap)
        plt.savefig(str(i)+"_test_template.png")
