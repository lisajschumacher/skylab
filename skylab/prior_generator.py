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
    r""" PriorGenerator builds sky templates based on HealPy maps.
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
        r""" Return the healpy array representing the prior template
        Normed and shifted such that the maximum is 1 and the minimum is larger than zero
        """
        return np.zeros((1,hp.nside2npix(self.nside)))

class UhecrPriorGenerator(PriorGenerator):
    r""" UhecrPriorGenerator builds a prior template from UHECR data
    
    Parameters:
        nside_param : int
            used to generate healpy map with nside=2**nside_param
            
        data_path: string
            where to find the UHECR data
            
        deflection: float
            Magnetic deflection parameter in radian, usually np.radians(3 or 6)
            will be scaled with energy of UHECR to calculate average deflection
            
        energy_threshold: float
            in EeV, between 0 and max(energy of UHECR), used to
            select events according to their energy

    Attributes:
        ra, dec : arrays of floats
            Coordinates of the generated healpy template
        
        nside : int
            Healpy nside of the generated template
        
        template_s/m : array
            Healpy map representing the prior template, with
            parameters nside/ra,dec. Generated from UHECR data
            s : single combined map of all events
            m : multiple maps for one event each, logarithmic
        
        n_uhecr : int > 0
            Number of UHECR events selected for generating the template
    """
    def __init__(self, nside_param):
        r""" Initialize and calculate the Prior templates
        
        Parameters:
            nside_param : int
            Sets Healpy parameter nside = 2**nside_param
            
            deflection : float
            Magnetic deflection parameter in radian
            
            energy_threshold : float
            Energy threshold for UHECR selection, should be between
            min and max of expected values
            
            data_path : string
            Path name where to find UHECR data
        """
        super(UhecrPriorGenerator, self).__init__(nside_param)

    @property
    def template(self):
        r""" Multiple logarithmic Prior templates,
        for all selected UHECR events individually.
        Translated to HealPy map scheme with certain nside parameter
        Max of all entries is 0 
        """
        return self._template
    
    @template.setter
    def template(self, templ):
        r""" Multiple logarithmic Prior templates,
        for all selected UHECR events individually.
        Translated to HealPy map scheme with certain nside parameter
        Max of all entries is 0 
        
        Make sure that basic prior template requirements are fulfilled
        """
        # shape should fit
        assert(np.shape(templ) == np.shape(self._template))
        # and the values should be below zero, i.e. exp() below 1
        assert(max(templ)<=0)
        self._template = templ
        
    @property
    def n_uhecr(self):
        r""" Get the number of uhecr events that passed the energy threshold
        """
        return self._n_uhecr

    @n_uhecr.setter
    def n_uhecr(self, n):
        r""" Set the number of uhecr events that passed the energy threshold
        """
        n = int(n)
        if n<=0:
            print("Invalid number of UHECR events, will be set to standard value of 1")
            self._n_uhecr = 1
        else:
            self._n_uhecr = n

    
    def calc_template(self, deflection, t_params):
        r""" Calculate the Prior template from ra, dec and sigma parameters.
        The single templates are constructed as 2D-symmetric Gaussians 
        on position (ra, dec) with width of sigma. All contributions are
        added up to build the complete Prior template or
        TO DO: LEAVE THEM AS SINGLE PRIORS
        
        Using 2D-Vectors on a sphere
        
        Parameters:
            deflection : float
                        parameter for source extent, called "D" in paper
                        usually 3 or 6 degree (but plz give in radian k?)
            t_params: arrays of ra, dec, energy and reco error
                Can be single float values each or 1D-arrays themselves
                Should have the same length
                
        returns:
            Prior Template in HealPy map format
        """
        t_ra, t_dec, t_energy, t_reco = t_params
        t_ra = np.atleast_1d(t_ra)
        t_dec = np.atleast_1d(t_dec)
        t_energy = np.atleast_1d(t_energy)
        t_reco = np.atleast_1d(t_reco)
        assert(len(t_ra) == len(t_dec))
        assert(len(t_ra) == len(t_energy))
        assert(len(t_ra) == len(t_reco))

        # set source extent by formula sigma_CR = D*100EeV/E_CR
        # plus reco error
        t_sigma = np.sqrt((deflection*100./t_energy)**2 + t_reco**2)

        _template = np.empty((len(t_ra), hp.nside2npix(self.nside)), dtype=np.float)
            
        for i in range(len(t_ra)):
            mean_vec = UnitSphericalRepresentation(Angle(t_ra[i], u.radian), 
                                                   Angle(t_dec[i], u.radian))
            map_vec = UnitSphericalRepresentation(Angle(self.ra, u.radian),
                                                  Angle(self.dec, u.radian))
            
            _template[i] = -1.*np.power((map_vec-mean_vec).norm(), 2) / t_sigma[i]**2 / 2.
        return _template

    def _get_UHECR_positions(self, energy_threshold, data_path,
                             files = ["AugerUHECR2014.txt", "TelArrayUHECR.txt"]):
        """ Read the UHECR text file(s)
        Parameters:
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
        ra, dec, energy, sigma_reco
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
                # Maybe this has to be updated with new results from TA and Auger
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
        energy = np.array(e_temp)[e_mask]
        sigma_reco = np.array(sigma_reco)[e_mask]

        # set attributes which we want to have access to
        self.n_uhecr = len(ra)
        self.energy = energy
        return ra, dec, energy, sigma_reco

# Testing
if __name__=="__main__":
    import matplotlib.pyplot as plt
    from seaborn import cubehelix_palette
    from ic_utils import cmap
    path = "/home/home2/institut_3b/lschumacher/phd_stuff/skylab_git/"
    t = UhecrPriorGenerator(5)
    template = t.calc_template(np.radians(6),
                    t._get_UHECR_positions(85,
                    data_path="/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data"))
    print("Selected {} CRs".format(t.n_uhecr))
    print("Above energies of {} EeV".format(min(t.energy)))
    
    fig = plt.figure(-1)
    tm = np.exp(template)
    tm = tm/tm.sum(axis=1)[np.newaxis].T
    tm = tm.sum(axis=0)
    hp.mollview(tm, fig=-1, cmap=cmap, rot=[180,0,0])
    hp.projtext(np.pi/2, 0.01, r"$0^\circ$", color="w", ha="right")
    hp.projtext(np.pi/2, -0.01, r"$360^\circ$", color="w")
    hp.projtext(np.pi/2, np.pi, r"$180^\circ$", color="w")
    plt.savefig(path+"figures/full_test_template.png")
    fig = plt.figure(7)
    tm = np.exp(template)
    tm = tm.sum(axis=0)
    hp.mollview(np.log10(tm), fig=7, cmap=cmap, rot=[180,0,0], min=-20)
    hp.projtext(np.pi/2, 0.01, r"$0^\circ$", color="w", ha="right")
    hp.projtext(np.pi/2, -0.01, r"$360^\circ$", color="w")
    hp.projtext(np.pi/2, np.pi, r"$180^\circ$", color="w")
    plt.savefig(path+"figures/full_log_test_template.png")
    
   
    for i,tm in enumerate(template[:3]):
        fig = plt.figure(i)
        hp.mollview(tm, fig=i, cmap=cmap, rot=[180,0,0], min=-20)
        hp.projtext(np.pi/2, 0.01, r"$0^\circ$", color="w", ha="right")
        hp.projtext(np.pi/2, -0.01, r"$360^\circ$", color="w")
        hp.projtext(np.pi/2, np.pi, r"$180^\circ$", color="w")
        plt.savefig(path+"figures/"+str(i)+"_test_template.png")

