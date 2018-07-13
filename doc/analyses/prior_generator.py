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


from skylab.priors import SpatialPrior

class UhecrSpatialPrior(SpatialPrior):
    r""" Class for building a spatial prior from UHECR information
    using healpix map(s). Same usage as UhecrSpatialPrior,
    just that the prior maps are directly generated from
    Uhecr information that is loaded from disk.
    
    Attributes
    ----------
    p : np.ndarray, list of np.ndarray
        Healpix map(s) in the form of a numpy array(s) where each entry
        represents the probability for finding a point source at each
        pixel location.
    log_p : np.ndarray, list of np.ndarray
        Natural logarithm of 'p' attribute
    nprior : int
        Number of prior maps. One source will be injected for each
        prior map during sample()
    nside : int
        Nside of the prior map
    npix : int
        Number of pixels in the prior map
    pixels : list
        List of pixel id numbers in map
    ts_norm : list
        Normalization of the TS term for the prior(s). This equals
        the log of the max prior value and is subtracted from
        log(prior) under the ts() function to ensure the prior
        always acts as a penalty term <= 0.
    """
    def __init__(self, nside_param, deflection, energy_threshold, data_path,
                 scan_thresh=0, allow_neg=False, 
                 **kwargs):
        r""" Constructor

        Parameters
        ----------
        p : np.ndarray, list of np.ndarray
            Healpix map(s) in the form of a numpy array(s) where each entry
            represents the probability for finding a point source at each
            pixel location.
        scan_thresh : float
            Threshold for checking pixel during all-sky scan
        \*\*kwargs :
            Given to self._get_UHECR_position
        """    
        
        self.nside = 2**nside_param
        # generate the prior maps        
        self.calc_template(deflection, energy_threshold, data_path, **kwargs)
        
        self.scan_thresh = scan_thresh
        self.allow_neg = allow_neg

        # ensure shape is at least 2d
        if len(self.p.shape) == 1:
            self.p = np.array([p])

        self.nprior = self.p.shape[0]
        self.npix = len(self.p[0])
        self.pixel_area = 4 * np.pi / self.npix

        # ensure all prior maps have same pixel number
        # and integrate to 1 over the surface of (ra, dec)
        for i in range(self.nprior):
            assert(len(self.p[i]) == self.npix)

            integral = np.sum(self.p[i], dtype=float) * self.pixel_area
            self.p[i] /= integral

            delta = np.sum(self.p[i], dtype=float) * self.pixel_area - 1.0
            assert(np.fabs(delta) < 1e-6)

        self.pixels = range(self.npix)
        self.nside = hp.npix2nside(self.npix)

        # compute log of the prior map
        mask = self.p > 0.
        self.log_p = -1e6 * np.ones_like(self.p)
        self.log_p[mask]  = np.log( self.p[mask] )

        self.ts_norm = np.log([max(prior) for prior in self.p])
    
    def calc_template(self, deflection, energy_threshold, data_path, **kwargs):
        r""" Calculate the Prior template from ra, dec and sigma parameters.
        The single templates are constructed as 2D-symmetric Gaussians 
        on position (ra, dec) with width of sigma. All contributions are
        added up to build the Prior template 
        
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
        t_ra, t_dec, t_energy, t_reco = self._get_UHECR_positions(energy_threshold,
                                                                  data_path, **kwargs)
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

        self.p = np.empty((len(t_ra), hp.nside2npix(self.nside)), dtype=np.float)
        theta, ra = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        dec = np.pi/2. - theta
        ra = ra
            
        for i in range(len(t_ra)):
            mean_vec = UnitSphericalRepresentation(Angle(t_ra[i], u.radian), 
                                                   Angle(t_dec[i], u.radian))
            map_vec = UnitSphericalRepresentation(Angle(ra, u.radian),
                                                  Angle(dec, u.radian))
            
            self.p[i] = np.exp(-1.*np.power((map_vec-mean_vec).norm(), 2)\
                               / t_sigma[i]**2 / 2.)

    def _get_UHECR_positions(self, 
                             energy_threshold, 
                             data_path,
                             shift = 0.,
                             files = ["AugerUHECR2014.txt", "TelArrayUHECR-2017.txt"]
                            ):
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
            data = np.genfromtxt(os.path.join(data_path, f), names=True, comments="#")
            dec_temp.extend(np.radians(data['dec']))
            ra_temp.extend(np.radians(data['RA']))
            if "TelArrayUHECR" in f:
                # Implement energy shift of 13%, see paper
                # Maybe this has to be updated with new results from TA and Auger
                e_temp.extend(data['E']*(1.-0.13))
                sigma_reco.extend(len(data)*[np.radians(1.5)])
            elif "test" in f.lower():
                e_temp.extend(data['E'])
                sigma_reco.extend(len(data)*[0.001])
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
        ra += shift * sigma_reco
        ra = ra%(np.pi*2.)
        # set attributes which we want to have access to
        self.energy = energy
        return ra, dec, energy, sigma_reco
    
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

    
    def calc_template(self, deflection, energy_threshold, data_path, **kwargs):
        r""" Calculate the Prior template from ra, dec and sigma parameters.
        The single templates are constructed as 2D-symmetric Gaussians 
        on position (ra, dec) with width of sigma. All contributions are
        added up to build the Prior template 
        
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
        t_ra, t_dec, t_energy, t_reco = self._get_UHECR_positions(energy_threshold,
                                                                  data_path,
                                                                  **kwargs
                                                                 )
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
            
            _template[i] = np.exp(-1.*np.power((map_vec-mean_vec).norm(), 2) / t_sigma[i]**2 / 2.)
        return _template

    def _get_UHECR_positions(self, 
                             energy_threshold, 
                             data_path,
                             shift = 0.,
                             files = ["AugerUHECR2014.txt", "TelArrayUHECR-2017.txt"],
                             declination_range = [-np.pi/2., np.pi/2.]
                            ):
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
            data = np.genfromtxt(os.path.join(data_path, f), names=True, comments="#")
            dec_temp.extend(np.radians(data['dec']))
            ra_temp.extend(np.radians(data['RA']))
            if "TelArrayUHECR" in f:
                # Implement energy shift of 13%, see paper
                # Maybe this has to be updated with new results from TA and Auger
                e_temp.extend(data['E']*(1.-0.13))
                sigma_reco.extend(len(data)*[np.radians(1.5)])
            elif "test" in f.lower():
                e_temp.extend(data['E'])
                sigma_reco.extend(len(data)*[0.001])
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
        ra += shift * sigma_reco
        ra = ra%(np.pi*2.)
        # set attributes which we want to have access to
        self.n_uhecr = len(ra)
        self.energy = energy
        return ra, dec, energy, sigma_reco

# Testing
if __name__=="__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ic_utils import cmap, get_paths, skymap
    from socket import gethostname
    import getpass
    username = getpass.getuser()
    from os.path import join
    from skylab.priors import SpatialPrior
    
    if username=="lschumacher":
        basepath, inipath, savepath, crpath, figurepath = get_paths(gethostname())
        figurepath = join(figurepath, "test_plots")
    else:
        raise("You probably have to set the crpath and figurepath for this script")
    
    pgen = UhecrPriorGenerator(5)
    p = pgen.calc_template(np.radians(6), 100, crpath)
    prior = SpatialPrior(p)
    print("Selected {} CRs".format(pgen.n_uhecr))
    print("Above energies of {} EeV".format(min(pgen.energy)))
    
    fig, ax = skymap(plt, prior.p.sum(axis=0), cmap=cmap)
    plt.savefig(join(figurepath, "test_prior_generator.png"))
    plt.close("all")

