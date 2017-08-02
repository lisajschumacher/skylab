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
# SciPy
import numpy as np

# skylab
from skylab.psLLH import PointSourceLLH, MultiPointSourceLLH
from skylab.ps_model import UniformLLH, EnergyLLH, PowerLawLLH
from skylab.ps_injector import PointSourceInjector
from skylab.prior_injector import PriorInjector
from skylab.priorllh import PriorLLH
from skylab.stacking_priorllh import StackingPriorLLH, MultiStackingPriorLLH

mrs = np.radians(1.)
mrs_min = np.radians(0.05)
log_sig = 0.2
logE_res = 0.1

# fix seed to reproduce same results
np.random.seed(1+3+3+7)

def load_data(basepath, inipath, filename, shuffle_bool=True, burn=True):
    """ 
    shuffle_bool: do or do not shuffle experimental data
                                (currently setting to false has no effect)
    basepath:
                        Madison: 	/data/user/coenders/data/MultiYearPointSource/npz/
                                            ini_path: /data/user/lschumacher/config_files_ps/*.ini
                        Aachen: 	/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub/
                                            ini_path: same as above
    Possible filenames:
    Style: IC*_(corrected_MC|exp).npy
    IC40
    IC59
    IC79
    IC86
    IC86-2012
    IC86-2013
    IC86-2014
    """
    config = ConfigParser.ConfigParser()
    config.read(join(inipath, filename+"_config.ini"))
    livetime = float(config.get("properties", "Livetime")) 

    if filename in ["IC86-2013","IC86-2014"]:
        mc = np.load(join(basepath, "IC86-2012"+"_corrected_MC.npy"))
    else:
        mc = np.load(join(basepath, filename+"_corrected_MC.npy"))
        
    mc = mc[mc["logE"] > 1.]
    mc = mc[['ra', 'dec', 'logE', 'sigma', 'sinDec', 'trueRa', 'trueDec', 'trueE', 'ow']]
    exp = np.load(join(basepath, filename+"_exp.npy"))
    exp = exp[['ra', 'logE', 'sigma', 'sinDec']]  # 'dec',
    exp = exp[exp["logE"] > 1.]
    exp['ra']=np.random.uniform(0, 2*np.pi, len(exp['ra']))
    if burn:
        part=10
        print("Using {:.0f}% of experimental data to reduce cpu time ...".format(100*1./part))
        exp = np.random.choice(exp, len(exp)/part)
        part=1
        print("Using {:.0f}% of mc data to reduce cpu time ...".format(100*1./part))
        mc = np.random.choice(mc, len(mc)/part)
    
    if shuffle_bool==False:
        print("Data is scrambled nevertheless, sorry, not sorry :)")
    return mc, exp, livetime

def prepare_directory(direc, mode=0754):
    """ 
    Prepare directory for writing
    Make directory if not yet existing
    Change chmod to desired value, default is 0754
    """
    if (not(exists(direc))):
        os.makedirs(direc)
    os.chmod(direc, mode)

def init(arr_exp, arr_mc, livetime, energy=True, **kwargs):
    
    fixed_gamma = kwargs.pop("fixed_gamma", False)
    add_prior = kwargs.pop("add_prior", False)
    fit_gamma = kwargs.pop("fit_gamma", 2.)
    
    Nexp = len(arr_exp)
    nbounds = 5000.

    if energy and not fixed_gamma:
        llh_model = EnergyLLH(sinDec_bins=min(50, Nexp // 50),
                                sinDec_range=[-1., 1.],
                                bounds=(0, 5))
    elif fixed_gamma:
        llh_model = EnergyLLH(sinDec_bins=min(50, Nexp // 50),
                                sinDec_range=[-1., 1.],
                                bounds=(fit_gamma, fit_gamma))
    else:
        llh_model = UniformLLH(sinDec_bins=max(3, Nexp // 200),
                               sinDec_range=[-1., 1.])
    if add_prior:
        llh = StackingPriorLLH(arr_exp, arr_mc, livetime, llh_model=llh_model,
                             mode="all", nsource=25, scramble=False,
                             nsource_bounds=(-nbounds, nbounds) if not energy else (0., nbounds),
                             seed=np.random.randint(2**32),
                             **kwargs)
    else:
        llh = PointSourceLLH(arr_exp, arr_mc, livetime, llh_model=llh_model,
                             mode="all", nsource=25, scramble=False,
                             nsource_bounds=(-nbounds, nbounds) if not energy else (0., nbounds),
                             seed=np.random.randint(2**32),
                             **kwargs)

    return llh

def multi_init(n, basepath, inipath, **kwargs):
    
    energy = kwargs.pop("energy", True)
    Nsrc = kwargs.pop("Nsrc", 0)
    fit_gamma = kwargs.pop("fit_gamma", 2.)
    src_gamma = kwargs.pop("src_gamma", 2.)
    fixed_gamma = kwargs.pop("fixed_gamma", True)
    add_prior = kwargs.pop("add_prior", True)
    nside_param = kwargs.pop("nside_param", 4)
    n_uhecr = kwargs.pop("n_uhecr", 0)
    prior = kwargs.pop("prior", [])
    burn = kwargs.pop("burn", True)
    inj_seed = kwargs.pop("inj_seed", None)
    
    # Current standard is to load first 4 files
    if "M16" in gethostname():
	# In case of me working on my laptop ...
	# load only 79 and 86 which I copied to local disc
	filenames = ["IC79", 
                    "IC86"]
	n = len(filenames)
    else:
	filenames = ["IC40",
                    "IC59",
                    "IC79", 
                    "IC86",
                    "IC86-2012",
                    "IC86-2013",
                    "IC86-2014"]

    nbounds = 5000.    
    mcdict = dict()
    expdict = dict()
    ltdict = dict()
    num = np.zeros(n)
    
    llh = MultiStackingPriorLLH(nsource=Nsrc+10,
                                nsource_bounds=(-nbounds, nbounds) if not energy else (0., nbounds),
                                seed=np.random.randint(2**32),
                                **kwargs)

    for i in xrange(n):
        mcdict[i], expdict[i], ltdict[i] = load_data(basepath, inipath, filenames[i], burn=burn)
        
    if Nsrc > 0:
        if add_prior:
            injector = PriorInjector(src_gamma,
                                        prior,
                                        nside_param=nside_param,
                                        n_uhecr=n_uhecr,
                                        seed=inj_seed)
            injector.fill(mcdict, ltdict)
            sampler = injector.sample(Nsrc, poisson=True)
            num, sam = sampler.next()
        else:
            pass
    else:
        pass
            
    for i in xrange(n):    
        llh_i =  init(arr_exp = np.append(expdict[i], sam[i]) if (Nsrc > 0 and num[i] != 0) else expdict[i],
                        arr_mc = mcdict[i],
                        livetime = ltdict[i],
                        energy=energy,
                        fixed_gamma=fixed_gamma,
                        fit_gamma=fit_gamma,
                        add_prior=add_prior,
                        **kwargs
                        )
        llh.add_sample(str(i), llh_i)

    return llh, injector
