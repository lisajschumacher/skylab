# -*-coding:utf8-*-

r"""Example script to create data in the right format and load it correctly
to the LLH classes.

"""

from __future__ import print_function

# Python
import ConfigParser
from socket import gethostname
from os.path import join, exists
from os import chmod, makedirs
import logging
logging.basicConfig(level=logging.WARN)
#from memory_profiler import profile

# SciPy
from scipy.stats import rayleigh
import numpy as np

# skylab
from skylab.ps_llh import PointSourceLLH, MultiPointSourceLLH
from skylab.llh_models import UniformLLH, EnergyLLH
from skylab.ps_injector import PointSourceInjector
from skylab.ps_injector import PriorInjector
from skylab.datasets import Datasets

mrs = np.radians(1.)
mrs_min = np.radians(0.05)
log_sig = 0.2
logE_res = 0.1

GeV = 1
TeV = 1000 * GeV

def load_compressed_data(f):
    savepath = "/home/lschumacher/svn_repos/skylab/trunk/doc/analyses/lschumacher-UHECR/compressed_datasets/"
    mc = np.load(savepath+f+"_mc_compressed.npy")
    exp = np.load(savepath+f+"_exp_compressed.npy")
    sinDec_bins = np.load(savepath+f+"_sd-bins_compressed.npy")
    energy_bins = np.load(savepath+f+"_e-bins_compressed.npy")
    livetime = np.load(savepath+f+"_lt_compressed.npy")
    
    return exp, mc, livetime, sinDec_bins, energy_bins

#@profile
def load_data(filename, burn=False, seed=1):
    r""" 
    filename: one of the filenames listed below
    burn : bool, choose whether or not to reduce your mc
    data to 20%
    seed : if you want to downsample MC
    
    returns: mc, exp, livetime, sinDec_bins, energy_bins
    
    Possible filename:
    IC40
    IC59
    IC79
    IC86, 2011
    IC86, 2012
    IC86, 2013
    IC86, 2014
    antares
    
    """
    exp, mc, livetime, sinDec_bins, energy_bins = load_compressed_data(filename)
        
    if burn:
        rs = np.random.RandomState(seed)
        mc = rs.choice(mc, len(mc)/5)
        
    return exp, mc, livetime, sinDec_bins, energy_bins

'''@profile
def load_data(filename, burn=False, seed=1, rescale=1):
    r""" 
    filename: one of the filenames listed below
    burn : bool, choose whether or not to reduce your mc
    data to 20%
    seed : if you want to downsample MC
    rescale : only for antares data, as this is only one sample
    e.g. rescale=6 -> livetime_antares/9*6 rescales to 6 years
    
    returns: mc, exp, livetime, sinDec_bins, energy_bins
    
    Possible filename:
    IC40
    IC59
    IC79
    IC86, 2011
    IC86, 2012
    IC86, 2013
    IC86, 2014
    antares
    
    """
    if "antares" in filename.lower():
        exp, mc, livetime, sinDec_bins, energy_bins = load_antares_data(rescale=rescale)
        mc = mc[['ra', 'dec', 'logE', 'sigma', 'trueRa', 'trueDec', 'trueE', 'ow']]
        exp = exp[['ra', 'dec', 'logE', 'sigma']]
    else:
        exp, mc, livetime = Datasets['PointSourceTracks'].season(filename)
        mc = mc[['ra', 'dec', 'sinDec', 'logE', 'sigma', 'trueRa', 'trueDec', 'trueE', 'ow']]
        exp = exp[['ra', 'dec', 'sinDec', 'logE', 'sigma']]
        
        sinDec_bins = Datasets['PointSourceTracks'].sinDec_bins(filename)
        energy_bins = Datasets['PointSourceTracks'].energy_bins(filename)
        
    mc = compress_struct_to_float32(mc, except_keys=["ow"])
    exp = compress_struct_to_float32(exp)
    
    if burn:
        rs = np.random.RandomState(seed)
        mc = rs.choice(mc, len(mc)/5)
        #exp = rs.choice(exp, len(exp)/10)
        
    return exp, mc, livetime, sinDec_bins, energy_bins'''

def save_raw_antares_data(return_sets=False):
    r"""
    Load antares data from disk and save them in IC style format
    """
    path_m = "/data/user/lschumacher/projects/stacking/antares/AntaresMC_Muons.dat"
    path_nu = "/data/user/lschumacher/projects/stacking/antares/AntaresMC_Neutrinos.dat"
    path_exp = "/data/user/lschumacher/projects/stacking/antares/Antares_Blind_Data.dat"
    
    path_f_mc = "/data/user/lschumacher/projects/stacking/antares/antares_MC.npy"
    path_f_exp = "/data/user/lschumacher/projects/stacking/antares/antares_exp.npy"
    
    livetime = 2423.6 #days 

    exp = np.genfromtxt(path_exp, names=True, usecols=[0,1,3,5])
    #muons = np.genfromtxt(path_m, names=True)
    full_mc = np.genfromtxt(path_nu, names=True)
    #full_mc = np.concatenate([mc, muons])
    #del mc
    #del muons

    mask = np.isnan(full_mc["logE"])
    full_mc = full_mc[~mask]
    
    full_mc["trueE"] = np.power(10, full_mc["trueE"])
    full_mc["trueRa"] = np.radians(full_mc["trueRa"])
    full_mc["trueDec"] = np.radians(full_mc["trueDec"])
    full_mc["ra"] = np.radians(full_mc["ra"])
    full_mc["dec"] = np.radians(full_mc["dec"])
    full_mc["ow"] /= livetime
    full_mc["wbkg"] /= livetime
    
    exp["sigma"] = np.radians(exp["sigma"])
    full_mc["sigma"] = np.radians(full_mc["sigma"])
    
    np.save(path_f_exp, exp)
    np.save(path_f_mc, full_mc)
    if return_sets:
        return full_mc, exp, livetime
    
def load_antares_data(rescale=1):
    """
    returns: exp, mc, livetime
    """
    path_mc = "/data/user/lschumacher/projects/stacking/antares/antares_MC.npy"
    path_exp = "/data/user/lschumacher/projects/stacking/antares/antares_exp.npy"
    
    exp = np.load(path_exp)
    mc = np.load(path_mc)
    
    sinDec_bins = np.linspace(-1., 0.8, num=20)
    energy_bins = np.arange(2.5, 5.375 + 0.01, 0.125)
    
    livetime = 2423.6 #days
    if rescale!=1:
        livetime = livetime/9.*rescale
    
    return exp, mc, livetime, sinDec_bins, energy_bins

def prepare_directory(direc, mode=0754):
    """ 
    Prepare directory for writing
    Make directory if not yet existing
    Change chmod to desired value, default is 0754
    """
    if (not(exists(direc))):
        os.makedirs(direc)
    os.chmod(direc, mode)

    
#@profile
def init(exp, mc, livetime, sinDec_bins, energy_bins, **kwargs):
    
    gamma_range = kwargs.pop("gamma_range", (1., 4.))
    mode = kwargs.pop("mode", "all")
    energy = kwargs.pop("energy", True)
    fix_gamma = kwargs.pop("fix_gamma", False)

    
    Nexp = len(exp)
    nbounds = 5000.

    if energy and not fix_gamma:
        llh_model = EnergyLLH(twodim_bins=[energy_bins, sinDec_bins],
                              allow_empty=True, kernel=1,
                              bounds=gamma_range
                             )
    elif fix_gamma:
        kwargs.pop("fix_gamma_val", 2.)
        llh_model = EnergyLLH(twodim_bins=[energy_bins, sinDec_bins],
                              allow_empty=True, kernel=1,
                              bounds=(fix_gamma_val, fix_gamma_val)
                             )
    else:
        llh_model = UniformLLH(twodim_bins=[energy_bins, sinDec_bins],
                              allow_empty=True, kernel=1)
        
    llh = PointSourceLLH(exp, mc, livetime, 
                         llh_model=llh_model,
                         mode=mode, nsource=25,
                         nsource_bounds=(-nbounds, nbounds) if not energy else (0., nbounds),
                         **kwargs
                        )

    return llh

#@profile
def multi_init(filenames, **kwargs):
    
    Nsrc = kwargs.pop("Nsrc", 0)
    burn = kwargs.pop("burn", True)
    src_gamma = kwargs.pop("src_gamma", 2.)
    prior = kwargs.pop("prior", None)
    energy = kwargs.pop("energy", True)
    seed = kwargs.pop("seed", 1)
    bunchsize = kwargs.pop("bunchsize", 100)

    nbounds = 5000.    
    mcdict = dict()
    expdict = dict()
    ltdict = dict()
    
    llh = MultiPointSourceLLH(seed=seed, **kwargs)                            
    
    for i,fname in enumerate(filenames):
        exp, mc, livetime, sinDec_bins, energy_bins = load_data(fname, burn=burn, seed=seed+i)
        llh_i =  init(exp = exp,
                      mc = mc,
                      livetime = livetime,
                      energy = energy,
                      sinDec_bins = sinDec_bins,
                      energy_bins = energy_bins,
                      **kwargs
                     )
        llh.add_sample(str(i), llh_i)
        del mc
        del exp
            
        
    if Nsrc > 0:        
        if prior is not None:
            injector = PriorInjector(prior, 
                                     gamma=src_gamma, 
                                     seed=seed,
                                     bunchsize=bunchsize,
                                     E0=1 * GeV
                                    )
            injector.fill(llh.exp, llh.mc, llh.livetime)
        else:
            src_dec = kwargs.pop("src_dec", 0.)
            src_ra = kwargs.pop("src_ra", np.pi/2.)
            injector = PointSourceInjector(gamma=src_gamma,
                                           seed=seed,
                                           bunchsize=bunchsize
                                          )
            injector.fill(src_dec, llh.exp, llh.mc, llh.livetime)
    else:
        injector = None

    return llh, injector

def single_init(filename, **kwargs):
    
    Nsrc = kwargs.pop("Nsrc", 0)
    burn = kwargs.pop("burn", True)
    src_gamma = kwargs.pop("src_gamma", 2.)
    prior = kwargs.pop("prior", None)
    bunchsize = kwargs.pop("bunchsize", 100)
    if "seed" in kwargs:
        seed = kwargs["seed"]
    else:
        seed = 1
        
    exp, mc, livetime, sinDec_bins, energy_bins = load_data(filename, burn=burn, seed=seed)
    llh =  init(exp = exp, mc = mc,
                livetime = livetime, sinDec_bins = sinDec_bins, 
                energy_bins = energy_bins, **kwargs)
    if Nsrc > 0:        
        if prior is not None:
            injector = PriorInjector(prior, 
                                     gamma=src_gamma, 
                                     seed=seed,
                                     bunchsize=bunchsize
                                    )
            injector.fill(llh.exp, llh.mc, llh.livetime)
        else:
            src_dec = kwargs.pop("src_dec", 0.)
            src_ra = kwargs.pop("src_ra", np.pi/2.)
            injector = PointSourceInjector(gamma=src_gamma,
                                           seed=seed,
                                           bunchsize=bunchsize
                                          )
            injector.fill(dec, llh.exp, llh.mc, llh.livetime)
    else:
        injector = None
        
    

    return llh, injector

def get_compressed_data(path, isMC=False):
    r"""Get data from file with all info and reduce info to what is used by skylab. Keep no additional keys.
        Also precission is reduced to floating point precision ('<f4').
        Tests have shown that this has a very minor effect of less then 1% on the TS value.
        Cuts also events out that are not in the diffuse fit range. 
        

        Parameters
        -----------
        path : string
            Path to data with full information.

        isMC : bool
            Default: False

        Returns
        --------
        data : ndarray
            Array of data with keys skylab needs. No additional info.

        Other Parameters
        -----------------

        erangeCut : bool
            IF True also a cut on logE is applyed. Should not be done for IC59. Is just requiered for IC79.
    
    """
    with FileStager(path, "r") as open_file:
        data = np.load(open_file)

    # use pull corrected sigma paraboloid
    data["sigma"] = data["sigma_pull_corrected"]

    # just use vars that are needed (save RAM)
    var = ["ra", "dec", "logE", "sigma"]
    if isMC: 
        if not "ow" in data.dtype.names:
            ow_name = "best_fit_OW" if "best_fit_OW" in data.dtype.names else "orig_OW"
            data.dtype.names = tuple([k if k!=ow_name else "ow" for k in data.dtype.names])
        var.extend([ "trueRa", "trueDec", "trueE", "ow", "conv", "prompt", "astro"])
    data = data[var]
    
    # compress to float32 to save RAM
    # do not compress ow, because it may cause problems whern normalizing the sum of weights
    data = compress_struct_to_float32(data, except_keys=["ow"])
    
    return data

def compress_struct_to_float32(data, except_keys=[]):
    r"""Compress structured array to float32 precision to save RAM. 
    ATTENTION: Lost in precision. That may cause problems for some variables.
    
        Parameters
        -----------
        data : structured array
            Structured data array that should be reduced to float32.

        Returns
        --------
        data : structured array
            Structured data array that is reduced to float32.

        Other Parameters
        -----------------
        except_keys : array
            Keys that are given will not be reduced to float32.
    """

    dt = data.dtype.descr
    for i in range(len(dt)):
        # do not compress except_keys
        if not dt[i][0] in except_keys: dt[i] = (dt[i][0], "<f4")
    data = data.astype(dt)
    return data