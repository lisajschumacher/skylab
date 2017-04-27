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

ps_injector
===========

Point Source Injection classes. The interface with the core
PointSourceLikelihood - Class requires the methods

    fill - Filling the class with Monte Carlo events

    sample - get a weighted sample with mean number `mu`

    flux2mu - convert from a flux to mean number of expected events

    mu2flux - convert from a mean number of expected events to a flux

"""

# python packages
import logging
from collections import OrderedDict, defaultdict
import copy
import os
import time
from functools import partial

# scipy-project imports
import numpy as np
from numpy.lib.recfunctions import drop_fields
from scipy.interpolate import InterpolatedUnivariateSpline
#from memory_profiler import profile
#from profilehooks import profile

# local package imports
from . import set_pars
from .utils import rotate, rotate_around

# get module logger
def trace(self, message, *args, **kwargs):
    r""" Add trace to logger with output level beyond debug

    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

_deg = 4
_ext = 3

def rotate_struct(ev, ra, dec, rand=False):
    r"""Wrapper around the rotate-method in skylab.utils for structured
    arrays.

    Parameters
    ----------
    ev : structured array
        Event information with ra, sinDec, plus true information

    ra, dec : float
        Coordinates to rotate the true direction onto

    Returns
    --------
    ev : structured array
        Array with rotated value, true information is deleted

    """
    names = ev.dtype.names

    rot = np.copy(ev)

    # Function call
    rot["ra"], rot_dec = rotate(ev["trueRa"], ev["trueDec"],
                                ra * np.ones(len(ev)), dec * np.ones(len(ev)),
                                ev["ra"], np.arcsin(ev["sinDec"]))
    if rand:
        rot["ra"], rot_dec = rotate_around(rot["ra"], rot_dec, ra * np.ones(len(ev)), dec * np.ones(len(ev)))

    if "dec" in names:
        rot["dec"] = rot_dec
    rot["sinDec"] = np.sin(rot_dec)

    # "delete" Monte Carlo information from sampled events
    mc = ["trueRa", "trueDec", "trueE", "ow"]

    return drop_fields(rot, mc)


class Injector(object):
    r"""Base class for Signal Injectors defining the essential classes needed
    for the LLH evaluation.

    """

    def __init__(self, *args, **kwargs):
        r"""Constructor: Define general point source features here...

        """
        self.__raise__()

    def __raise__(self):
        raise NotImplementedError("Implemented as abstract in {0:s}...".format(
                                    self.__repr__()))

    def fill(self, *args, **kwargs):
        r"""Filling the injector with the sample to draw from, work only on
        data samples known by the LLH class here.

        """
        self.__raise__()

    def flux2mu(self, *args, **kwargs):
        r"""Internal conversion from fluxes to event numbers.

        """
        self.__raise__()

    def mu2flux(self, *args, **kwargs):
        r"""Internal conversion from mean number of expected neutrinos to
        point source flux.

        """
        self.__raise__()

    def sample(self, *args, **kwargs):
        r"""Generator method that returns sampled events. Best would be an
        infinite loop.

        """
        self.__raise__()


class PointSourceInjector(Injector):
    r"""Class to inject a point source into an event sample.

    """
    _src_dec = np.nan
    _sinDec_bandwidth = 0.1 # Corresponds to approx. 5.7deg
    _sinDec_range = [-1., 1.]

    _E0 = 1.
    _GeV = 1.e3
    _e_range = [0., np.inf]

    _random = np.random.RandomState()
    _seed = None

    def __init__(self, gamma, **kwargs):
        r"""Constructor. Initialize the Injector class with basic
        characteristics regarding a point source.

        Parameters
        -----------
        gamma : float
            Spectral index, positive values for falling spectra

        kwargs : dict
            Set parameters of class different to default

        """

        # source properties
        self.gamma = gamma

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def __str__(self):
        r"""String representation showing some more or less useful information
        regarding the Injector class.

        """
        sout = ("\n{0:s}\n"+
                67*"-"+"\n"+
                "\tSpectral index     : {1:6.2f}\n"+
                "\tSource declination : {2:5.1f} deg\n"
                "\tlog10 Energy range : {3:5.1f} to {4:5.1f}\n").format(
                         self.__repr__(),
                         self.gamma, np.degrees(self.src_dec),
                         *self.e_range)
        sout += 67*"-"

        return sout

    @property
    def sinDec_range(self):
        return self._sinDec_range

    @sinDec_range.setter
    def sinDec_range(self, val):
        if len(val) != 2:
            raise ValueError("SinDec range needs only upper and lower bound!")
        if val[0] < -1 or val[1] > 1:
            logger.warn("SinDec bounds out of [-1, 1], clip to that values")
            val[0] = max(val[0], -1)
            val[1] = min(val[1], 1)
        if np.diff(val) <= 0:
            raise ValueError("SinDec range has to be increasing")
        self._sinDec_range = np.array([float(val[0]), float(val[1])])
        return

    @property
    def e_range(self):
        return self._e_range

    @e_range.setter
    def e_range(self, val):
        if len(val) != 2:
            raise ValueError("Energy range needs upper and lower bound!")
        if val[0] < 0. or val[1] < 0:
            logger.warn("Energy range has to be non-negative")
            val[0] = max(val[0], 0)
            val[1] = max(val[1], 0)
        if np.diff(val) <= 0:
            raise ValueError("Energy range has to be increasing")
        self._e_range = [float(val[0]), float(val[1])]
        return

    @property
    def GeV(self):
        return self._GeV

    @GeV.setter
    def GeV(self, value):
        self._GeV = float(value)

        return

    @property
    def E0(self):
        return self._E0

    @E0.setter
    def E0(self, value):
        self._E0 = float(value)

        return

    @property
    def random(self):
        return self._random

    @random.setter
    def random(self, value):
        self._random = value

        return

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, val):
        logger.info("Setting global seed to {0:d}".format(int(val)))
        self._seed = int(val)
        self.random = np.random.RandomState(self.seed)

        return

    @property
    def sinDec_bandwidth(self):
        return self._sinDec_bandwidth

    @sinDec_bandwidth.setter
    def sinDec_bandwidth(self, val):
        if val < 0. or val > 1:
            logger.warn("Sin Declination bandwidth {0:2e} not valid".format(
                            val))
            val = min(1., np.fabs(val))
        self._sinDec_bandwidth = float(val)

        self._setup()

        return

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        if not np.fabs(val) < np.pi / 2.:
            logger.warn("Source declination {0:2e} not in pi range".format(
                            val))
            return
        if not (np.sin(val) > self.sinDec_range[0]
                and np.sin(val) < self.sinDec_range[1]):
            logger.error("Injection declination not in sinDec_range!")
        self._src_dec = float(val)

        self._setup()

        return

    def _setup(self):
        r"""If one of *src_dec* or *dec_bandwidth* is changed or set, solid
        angles and declination bands have to be re-set.

        """

        A, B = self._sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(self.src_dec) + b

        min_sinDec = max(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = min(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # solid angle of selected events
        self._omega = 2. * np.pi * (max_sinDec - min_sinDec)

        return

    def _weights(self):
        r"""Setup weights for given models.

        """
        # weights given in days, weighted to the point source flux
        self.mc_arr["ow"] *= self.mc_arr["trueE"]**(-self.gamma) / self._omega

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        # normalized weights for probability
        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        # double-check if no weight is dominating the sample
        if self._norm_w.max() > 0.1:
            logger.warn("Warning: Maximal weight exceeds 10%: {0:7.2%}".format(
                            self._norm_w.max()))

        return

    def fill(self, src_dec, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s).

        Parameters
        -----------
        src_dec : float, array-like
            Source location(s)
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """

        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        self.src_dec = src_dec

        self.mc = dict()
        self.mc_arr = np.empty(0, dtype=[("idx", np.int), ("enum", np.int),
                                         ("trueE", np.float), ("ow", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            # get MC event's in the selected energy and sinDec range
            band_mask = ((np.sin(mc_i["trueDec"]) > np.sin(self._min_dec))
                         &(np.sin(mc_i["trueDec"]) < np.sin(self._max_dec)))
            band_mask &= ((mc_i["trueE"] / self.GeV > self.e_range[0])
                          &(mc_i["trueE"] / self.GeV < self.e_range[1]))

            if not np.any(band_mask):
                print("Sample {0:d}: No events were selected!".format(key))
                self.mc[key] = mc_i[band_mask]

                continue

            self.mc[key] = mc_i[band_mask]

            N = np.count_nonzero(band_mask)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            mc_arr["ow"] = self.mc[key]["ow"] * livetime[key] * 86400.
            mc_arr["trueE"] = self.mc[key]["trueE"]

            self.mc_arr = np.append(self.mc_arr, mc_arr)

            #~ print("Sample {0:s}: Selected {1:6d} events at {2:7.2f}deg".format(
                        #~ str(key), N, np.degrees(self.src_dec)))

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")

        #~ print("Selected {0:d} events in total".format(len(self.mc_arr)))

        self._weights()

        return 

    def flux2mu(self, flux):
        r"""Convert a flux to mean number of expected events.

        Converts a flux :math:`\Phi_0` to the mean number of expected
        events using the spectral index :math:`\gamma`, the
        specified energy unit `x GeV` and the point of normalization `E0`.

        The flux is calculated as follows:

        .. math::

            \frac{d\Phi}{dE}=\Phi_0\,E_0^{2-\gamma}
                                \left(\frac{E}{E_0}\right)^{-\gamma}

        In this way, the flux will be equivalent to a power law with
        index of -2 at the normalization energy `E0`.

        """

        gev_flux = (flux
                        * (self.E0 * self.GeV)**(self.gamma - 1.)
                        * (self.E0)**(self.gamma - 2.))

        return self._raw_flux * gev_flux

    def mu2flux(self, mu):
        r"""Calculate the corresponding flux in [*GeV*^(gamma - 1) s^-1 cm^-2]
        for a given number of mean source events.

        """

        gev_flux = mu / self._raw_flux

        return (gev_flux
                    * self.GeV**(1. - self.gamma) # turn from I3Unit to *GeV*
                    * self.E0**(2. - self.gamma)) # go from 1*GeV* to E0

    def sample(self, src_ra, mean_mu, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """

        # generate event numbers using poissonian events
        while True:
            num = (self.random.poisson(mean_mu)
                        if poisson else int(np.around(mean_mu)))

            logger.debug(("Generated number of sources: {0:3d} "+
                          "of mean {1:5.1f} sources").format(num, mean_mu))
            
            # create numpy array with *num* entries
            sam_ev = np.empty((num, ), dtype=[  ('ra', '<f8'), 
                                                ('dec', '<f8'), 
                                                ('logE', '<f8'), 
                                                ('sigma', '<f8'), 
                                                ('sinDec', '<f8')])            

            # if no events should be sampled, return nothing
            if num < 1:
                yield num, sam_ev
                continue

            sam_idx = self.random.choice(self.mc_arr, size=num, p=self._norm_w)

            # get the events that were sampled
            enums = np.unique(sam_idx["enum"])

            if len(enums) == 1 and enums[0] < 0:
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])

                yield num, rotate_struct(sam_ev, src_ra, self.src_dec)
                continue

            sam_ev = dict()
            for enum in enums:
                idx = sam_idx[sam_idx["enum"] == enum]["idx"]
                sam_ev_i = np.copy(self.mc[enum][idx])
                sam_ev[enum] = rotate_struct(sam_ev_i, src_ra, self.src_dec)

            yield num, sam_ev

class UHECRSourceInjector(PointSourceInjector):
    """
    Inject neutrino sources according to UHECR event distribution
    Also including magnetic deflection of UHECRs such that neutrino positions are randomly smeared
    
    Make it working for multiple source positions!
    
    CANNOT be used with PointSourceLLH.weighted_sensitivity
    CAN be used with do trials, as it samples the correct events needed for the trials
    """
    def __init__(self, gamma, D, e_thresh, **kwargs):
        
        self.set_UHECR_positions(D, e_thresh)
        self._omega = np.pi * 4.
        super(UHECRSourceInjector, self).__init__(gamma, **kwargs)
        
    @property
    def sinDec_bandwidth(self):
        return self._sinDec_bandwidth

    @sinDec_bandwidth.setter
    def sinDec_bandwidth(self, val):
        if np.any(val < 0.) or np.any(val > 1):
            logger.warn("Sin Declination bandwidth {0:2e} not valid".format(
                            val))
            val = np.minimum(1., np.fabs(val))
        self._sinDec_bandwidth = np.array(val, dtype=float)

        self._setup()

        return

    @property
    def src_dec(self):
        return self._src_dec

    @src_dec.setter
    def src_dec(self, val):
        if np.any(np.fabs(val) > np.pi / 2.):
            logger.warn("Source declination {} not in pi range".format(
                            val[np.fabs(val) > np.pi / 2.]))
            return
        if (np.any(np.sin(val) < self.sinDec_range[0])
                or np.any(np.sin(val) > self.sinDec_range[1])):
            logger.error("Injection declination not in sinDec_range!")
        self._src_dec = np.array(val, dtype=float)

        self._setup()

        return

    def _setup(self):
        r"""If one of *src_dec* or *dec_bandwidth* is changed or set, solid
        angles and declination bands have to be re-set.

        """

        A, B = self._sinDec_range

        m = (A - B + 2. * self.sinDec_bandwidth) / (A - B)
        b = self.sinDec_bandwidth * (A + B) / (B - A)

        sinDec = m * np.sin(self.src_dec) + b

        min_sinDec = np.maximum(A, sinDec - self.sinDec_bandwidth)
        max_sinDec = np.minimum(B, sinDec + self.sinDec_bandwidth)

        self._min_dec = np.arcsin(min_sinDec)
        self._max_dec = np.arcsin(max_sinDec)

        # solid angle of selected events
        self._omega = 2. * np.pi * (max_sinDec - min_sinDec)

        return

    def mu2flux(self, mu, weights, omega):
        r"""calculate flux

        """
        raw_flux = np.sum(weights*1./omega, dtype=np.float)

        flux = mu *self.GeV**(1. - self.gamma) * self.E0**(2. - self.gamma) / len(self.uhecr_dec) / raw_flux

        return flux
    
    def set_UHECR_positions(self, D, e_thresh, **kwargs):
        """
        Read the UHECR text file(s)
        Parameters:

            D : float
                    parameter for source extent, called "D" in paper
                    usually 3 or 6 degree (but plz give in radian k?)

            e_thresh : float
                                 threshold value for energy given in EeV
            
        kwargs : not yet implemented
        
        returns : three arrays
        dec, ra, sigma (i.e. assumed magnetic deflection)
        """

        path = "/home/home2/institut_3b/lschumacher/phd_stuff/phd_stuff_git/phd_code/CRdata"
        files_dict = {"auger" : {"f" : "AugerUHECR2014.txt", "data" : None}, "ta" : {"f" : "TelArrayUHECR.txt", "data" : None}}
        dec_temp = []
        ra_temp = []
        e_temp = []

        for k,f in files_dict.iteritems():
            f["data"] = np.genfromtxt(os.path.join(path, f["f"]), names=True)
            dec_temp.extend(np.radians(f["data"]['dec']))
            ra_temp.extend(np.radians(f["data"]['RA']))
            if k=="ta":
                # Implement energy shift of 13%, see paper
                e_temp.extend(files_dict[k]["data"]['E']*(1.-0.13))
            else:
                e_temp.extend(files_dict[k]["data"]['E'])

        # Check the threshold
        if e_thresh > max(e_temp):
            default = 85.
            print("Energy threshold {:1.2f} too large, max value {:1.2f}! Set threshold to {:1.2f} EeV".format(e_thresh, max(e_temp), default))
            e_thresh=default

        # Select events above energy threshold
        e_mask = np.where(np.array(e_temp)>=e_thresh)[0]

        self.uhecr_dec = np.array(dec_temp)[e_mask]
        self.uhecr_ra = np.array(ra_temp)[e_mask]

        # set source extent by formula sigma_CR = D*100EeV/E_CR
        self.uhecr_sigma = D*100./np.array(e_temp)[e_mask]
    
    def set_injection_position(self):
        """
        Find a source position some degree away from assumed source position given by dec/ra.
        Deviation chosen using sigma set in set_UHECR_positions.
        """

        #~ dec3=np.pi/2.-abs(self.random.normal(loc=self.uhecr_sigma, scale=self.uhecr_sigma))
        ra3=self.random.uniform(0, np.pi*2., size=len(self.uhecr_dec))
    
        dist0 = self.random.normal(loc=self.uhecr_sigma, scale=self.uhecr_sigma)
        dist0 = dist0[dist0>0]
        while len(dist0)<len(self.uhecr_sigma):
            temp = self.random.normal(loc=self.uhecr_sigma, scale=self.uhecr_sigma)
            dist0 = np.append(dist0, temp[temp>0])
        dist0 = dist0[:len(self.uhecr_sigma)]
        
        dec3=np.pi/2.-dist0

        
        scaling = np.ones_like(self.uhecr_dec)
        ra_rot, dec_rot = rotate(0.*scaling, np.pi/2.*scaling,
                                 self.uhecr_ra, self.uhecr_dec, 
                                 ra3*scaling, dec3*scaling)
        return dec_rot, ra_rot
    
    #@profile
    def fill(self, mc, livetime):
        r"""Fill the Injector with MonteCarlo events.
        Calculate the acceptance

        Parameters
        -----------
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """
        
        if isinstance(mc, dict) ^ isinstance(livetime, dict):
            raise ValueError("mc and livetime not compatible")

        self.signal_acceptance = []
        self.mc = dict()
        self.mc_arr = np.empty(0, dtype=[("idx", np.int), 
                                         ("enum", np.int),
                                         ("w", np.float)])
                                         #("trueE", np.float), ("ow", np.float)])

        if not isinstance(mc, dict):
            mc = {-1: mc}
            livetime = {-1: livetime}

        for key, mc_i in mc.iteritems():
            
            #~ n, bins = np.histogram(np.sin(mc_i["trueDec"]), 
                                   #~ bins=90, 
                                   #~ weights=mc_i["ow"]*mc_i["trueE"]**(-self.gamma)* livetime[key] * 86400., 
                                   #~ range=self.sinDec_range
                                  #~ )
            #~ self.signal_acceptance.append(InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2., n))
            
            self.mc[key] = mc_i

            N = len(mc_i)
            mc_arr = np.empty(N, dtype=self.mc_arr.dtype)
            # With idx and enum we are able to find the corresponding mc event
            mc_arr["idx"] = np.arange(N)
            mc_arr["enum"] = key * np.ones(N)
            # This is not yet normed with _omega
            mc_arr["w"] = (self.mc[key]["ow"] * livetime[key] * 86400. 
                           * self.mc[key]["trueE"]**(-self.gamma))

            self.mc_arr = np.append(self.mc_arr, mc_arr)

        if len(self.mc_arr) < 1:
            raise ValueError("Select no events at all")


        return
    #@profile
    def sample(self, mean_mu, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """

        # generate event numbers using poissonian events
        while True:
            # Set injection position somewhere near UHECR position
            self.src_dec, self.src_ra = self.set_injection_position()
            num = (self.random.poisson(mean_mu)
                        if poisson else int(np.around(mean_mu)))
  
            logger.debug(("Generate number of source events of mean {:5.1f}").format(mean_mu))     

            if num < 1:
                print("No events sampled")
                yield num, np.zeros(num, 
                                    dtype=[('ra', '<f8'), 
                                           ('dec', '<f8'), 
                                           ('logE', '<f8'), 
                                           ('sigma', '<f8'),
                                           ('sinDec', '<f8')]
                                   )
                continue
                
            sam_ev = defaultdict(partial(np.ndarray, 
                                         0, 
                                         dtype=[('ra', '<f8'), 
                                                ('dec', '<f8'), 
                                                ('logE', '<f8'), 
                                                ('sigma', '<f8'), 
                                                ('sinDec', '<f8')
                                               ]
                                        )
                                )             
            mask = np.empty((len(self.mc_arr),len(self.src_ra)), dtype=bool)
            ind=0
            for key, mc_i in self.mc.iteritems():
                mask[ind:ind+len(mc_i)] = ((np.sin(mc_i["trueDec"])[np.newaxis].T > np.sin(self._min_dec))
                                                                &(np.sin(mc_i["trueDec"])[np.newaxis].T < np.sin(self._max_dec))
                                                                &(mc_i["trueE"][np.newaxis].T / self.GeV > self.e_range[0])
                                                                &(mc_i["trueE"][np.newaxis].T / self.GeV < self.e_range[1])
                                            )
                                            
                ind += len(mc_i)
            
            single_mask=np.logical_or.reduce(mask, axis=1)
            sam_idx = self.random.choice(self.mc_arr[single_mask], size=num, p=self.mc_arr["w"][single_mask]/np.sum(self.mc_arr["w"][single_mask]))
            enums = np.unique(sam_idx["enum"])

            
            n_inj = 0
            flux = 0.

            if len(enums) == 1 and enums[0] < 0:
                raise NotImplementedError("D:")
                # only one sample, just return recarray
                sam_ev = np.copy(self.mc[enums[0]][sam_idx["idx"]])

                yield num, rotate_struct(sam_ev, src_ra, self.src_dec)
                continue
            else:
                for enum in enums:
                    idx = sam_idx[sam_idx["enum"] == enum]["idx"]           
                    for i,(src_ra_i, src_dec_i) in enumerate(zip(self.src_ra, self.src_dec)):
                    
                        src_mask = ((np.sin(self.mc[enum][idx]["trueDec"]) > np.sin(self._min_dec[i]))
                                    &(np.sin(self.mc[enum][idx]["trueDec"]) < np.sin(self._max_dec[i])))
                        if np.any(src_mask):
                            flux += self.mu2flux(np.count_nonzero(src_mask),
                                                self.mc_arr["w"][mask[:,i]],
                                                self._omega[i]
                                                )
                            sam_ev_i = np.copy(self.mc[enum][idx][src_mask])
                            sam_ev[enum] = np.append(sam_ev[enum], 
                                                     rotate_struct(sam_ev_i, 
                                                                   src_ra_i, 
                                                                   src_dec_i,
                                                                   rand=True
                                                                  )
                                                    )
                    n_inj += len(sam_ev[enum])

            yield n_inj, flux, sam_ev
            
class StackingSourceInjector(PointSourceInjector):
    
    def __init__(self, gamma):        
        super(StackingSourceInjector, self).__init__(gamma)
    
    #@profile
    def fill(self, src_dec, mc, livetime):
        r"""Fill the Injector with MonteCarlo events selecting events around
        the source position(s).

        Parameters
        -----------
        src_dec : float, array-like
            Source location(s)
        mc : recarray, dict of recarrays with sample enum as key (MultiPointSourceLLH)
            Monte Carlo events
        livetime : float, dict of floats
            Livetime per sample

        """

        self.src_dec = src_dec
        band_mask = OrderedDict()

        if not isinstance(mc, dict):
            mc = {-1: mc}

        for key, mc_i in mc.iteritems():
            # get MC event's in the selected energy and sinDec range
            band_mask[key] = ((np.sin(mc_i["trueDec"]) > np.sin(self._min_dec))
                         &(np.sin(mc_i["trueDec"]) < np.sin(self._max_dec)))
            band_mask[key] &= ((mc_i["trueE"] / self.GeV > self.e_range[0])
                          &(mc_i["trueE"] / self.GeV < self.e_range[1]))
            
        #self._weights()

        return band_mask
    
    #@profile
    def sample(self, src_ra, mean_mu, mc, mc_arr, poisson=True):
        r""" Generator to get sampled events for a Point Source location.

        Parameters
        -----------
        mean_mu : float
            Mean number of events to sample

        Returns
        --------
        num : int
            Number of events
        sam_ev : iterator
            sampled_events for each loop iteration, either as simple array or
            as dictionary for each sample

        Optional Parameters
        --------------------
        poisson : bool
            Use poisson fluctuations, otherwise sample exactly *mean_mu*

        """
        while True:
            self.mc = mc
            self.mc_arr = mc_arr
            self._weights()
            num, sam_ev = super(StackingSourceInjector, self).sample(src_ra, mean_mu, poisson=poisson).next()
            del self.mc
            del self.mc_arr
            yield num, sam_ev
            

class InjectorHandler(PointSourceInjector):
    
    def __init__(self, gamma):
        
        self.inj_dict = OrderedDict()
        self._sources = 0
        super(InjectorHandler, self).__init__(gamma)
        
    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, val):
        if type(val)!=int:
            val=np.int(np.round(val))
            print("Set number of sources to int(val)")
        if val < 0:
            val = max(val, 0)
            logger.warn("#sources has to be non-negative, now set to {}".format(val))
        if np.fabs(val, self._sources)>1:
            logger.warn("Decreasing or Increasing #sources by more than 1!")
        self._sources = val
        return

    #@profile    
    def add_injector(self, src_dec, mc, livetime):
        
        self.inj_dict.update({self._sources : {"inj" : StackingSourceInjector(self.gamma)}})
        self.inj_dict[self._sources]["band_mask"] = self.inj_dict[self._sources]["inj"].fill(src_dec, copy.copy(mc), livetime)
        self._sources +=1
        
    def reduce_mc(self):
        raise("Not yet implemented")

    #@profile    
    def fill(self, src_dec, mc, livetime):
        
        if not isinstance(mc, dict):
            self.mc = {-1: mc}
            self.livetime = {-1: livetime}
            
        else:
            self.mc = mc
            self.livetime = livetime

        self.signal_acceptance = []
            
        self.declinations = np.atleast_1d(src_dec)
        for src_dec_i in self.declinations:
            self.add_injector(src_dec_i, self.mc, self.livetime)
            
        # Get the dec-acceptance in order to weight the sources when sampling    
        nBins=100
        nRange=(-1., 1.)
        for key, mc_i in self.mc.iteritems():
            n, bins = np.histogram(np.sin(mc_i["trueDec"]), 
                                   bins=nBins, 
                                   weights=mc_i["ow"]*mc_i["trueE"]**(-self.gamma)* livetime[key] * 86400., 
                                   range=nRange
                                  )
            self.signal_acceptance.append(InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2., n))
            
    def sample(self, src_ra, mean_mu, poisson=True):

        while True:
            # Initialize the lists
            num=[]
            sampled_events = OrderedDict()
            for k in self.mc.keys():
                sampled_events[k] = []
            self.right_ascensions = np.atleast_1d(src_ra)
            
            # If there's some mismatch, better know before sampling
            assert(len(self.right_ascensions)==len(self.declinations))
            
            # Get the source strength weighted with the expected acceptance for signal
            acc = np.zeros_like(self.declinations)
            for s in self.signal_acceptance:
                acc += s(np.sin(self.declinations))
            acc /= sum(acc)
            assert(np.isclose(sum(acc),1.))
            M = acc * mean_mu * self.sources / (1.*len(src_ra))
            #~ print(M)
        
            for i,inj in enumerate(self.inj_dict.itervalues()):
                
                # Prepare temporary variables
                mc=dict()
                mc_arr = np.empty(0, dtype=[("idx", np.int), ("enum", np.int),
                                         ("trueE", np.float), ("ow", np.float)])
                for key, mc_i in self.mc.iteritems():
                    mc[key] = copy.copy(mc_i[inj["band_mask"][key]])
                    N = np.count_nonzero(inj["band_mask"][key])
                    mc_temp = np.empty(N, dtype=[("idx", np.int), ("enum", np.int),
                                             ("trueE", np.float), ("ow", np.float)])
                    mc_temp["idx"] = np.arange(N)
                    mc_temp["enum"] = key * np.ones(N) 
                    mc_temp["ow"] = mc[key]["ow"] * self.livetime[key] * 86400.
                    mc_temp["trueE"] = mc[key]["trueE"]
                    
                    mc_arr = np.append(mc_arr, mc_temp)
                # Inject events for each source
                num_temp, sam_ev_temp = inj["inj"].sample(self.right_ascensions[i], M[i], mc, mc_arr, poisson=poisson).next()
                
                # Enlarge the event sample           
                # Add the results to what will be yielded
                num.append(num_temp)
                # Only add something if there is something to add
                if num_temp>0:
                    for k in sam_ev_temp.keys():
                        if len(sam_ev_temp[k])>0:
                            sampled_events[k].extend(sam_ev_temp[k])
                    
                #print("sampled events source {}\n".format(i), sam_ev_temp["ra"], "\n total: ", sampled_events, "\n")
            # Convert to arrays
            self._nums = np.array(num)
            sam_ev = OrderedDict()
            for k,sam in sampled_events.iteritems():
                sam_ev[k] = np.array(sam, dtype=[('ra', '<f8'), ('dec', '<f8'), ('logE', '<f8'), ('sigma', '<f8'), ('sinDec', '<f8')])
            yield sum(num), sam_ev


class ModelInjector(PointSourceInjector):
    r"""PointSourceInjector that weights events according to a specific model
    flux.

    Fluxes are measured in percent of the input flux.

    """

    def __init__(self, logE, logFlux, *args, **kwargs):
        r"""Constructor, setting up the weighting function.

        Parameters
        -----------
        logE : array
            Flux Energy in units log(*self.GeV*)

        logFlux : array
            Flux in units log(*self.GeV* / cm^2 s), i.e. log(E^2 dPhi/dE)

        Other Parameters
        -----------------
        deg : int
            Degree of polynomial for flux parametrization

        args, kwargs
            Passed to PointSourceInjector

        """

        deg = kwargs.pop("deg", _deg)
        ext = kwargs.pop("ext", _ext)

        s = np.argsort(logE)
        logE = logE[s]
        logFlux = logFlux[s]
        diff = np.argwhere(np.diff(logE) > 0)
        logE = logE[diff]
        logFlux = logFlux[diff]

        self._spline = InterpolatedUnivariateSpline(
                            logE, logFlux, k=deg)

        # use default energy range of the flux parametrization
        kwargs.setdefault("e_range", [10.**np.amin(logE), 10.**np.amax(logE)])

        # Set all other attributes passed to the class
        set_pars(self, **kwargs)

        return

    def _weights(self):
        r"""Calculate weights, according to given flux parametrization.

        """

        trueLogGeV = np.log10(self.mc_arr["trueE"]) - np.log10(self.GeV)

        logF = self._spline(trueLogGeV)
        flux = np.power(10., logF - 2. * trueLogGeV) / self.GeV

        # remove NaN's, etc.
        m = (flux > 0.) & np.isfinite(flux)
        self.mc_arr = self.mc_arr[m]

        # assign flux to OneWeight
        self.mc_arr["ow"] *= flux[m] / self._omega

        self._raw_flux = np.sum(self.mc_arr["ow"], dtype=np.float)

        self._norm_w = self.mc_arr["ow"] / self._raw_flux

        return

    def flux2mu(self, flux):
        r"""Convert a flux to number of expected events.

        """

        return self._raw_flux * flux

    def mu2flux(self, mu):
        r"""Convert a mean number of expected events to a flux.

        """

        return float(mu) / self._raw_flux


