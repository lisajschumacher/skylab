import os
import ConfigParser
import numpy as np
import scipy.optimize
from skylab.utils import poisson_percentile

def load_data(basepath, inipath, filename, shuffle_bool=True):
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
	config.read(os.path.join(inipath, filename+"_config.ini"))
	livetime = float(config.get("properties", "Livetime")) 

	if filename in ["IC86-2013","IC86-2014"]:
		mc = np.load(os.path.join(basepath, "IC86-2012"+"_corrected_MC.npy"))
	else:
		mc = np.load(os.path.join(basepath, filename+"_corrected_MC.npy"))
		
	mc = mc[mc["logE"] > 1.]
	mc = mc[['ra', 'dec', 'logE', 'sigma', 'sinDec', 'trueRa', 'trueDec', 'trueE', 'ow']]
	exp = np.load(os.path.join(basepath, filename+"_exp.npy"))
	exp = exp[['ra', 'dec', 'logE', 'sigma', 'sinDec']]  
	exp = exp[exp["logE"] > 1.]
	exp['ra']=np.random.uniform(0, 2*np.pi, len(exp['ra']))
	
	if shuffle_bool==False:
		print "Data is scrambled nevertheless, sorry :)"
	return mc, exp, livetime

def prepare_directory(direc, mode=0754):
	""" 
	Prepare directory for writing
	Make directory if not yet existing
	Change chmod to desired value, default is 0754
	"""
	if (not(os.path.exists(direc))):
		os.makedirs(direc)
		os.chmod(direc, mode)

	print "Prepared writing directory: " + direc

def angular_distance(x1, x2):
	""" 
	Compute the angular distance between two vectors on a unit sphere
	Parameters :
		x1/2: Vector with [declination, right-ascension]
	Return :
		cosine of angular distance, in order to get the angular
		distance in rad, take arccos of result
	"""
	x1=np.array(x1)
	x2=np.array(x2)
	assert(len(x1)==len(x2)==2)
	return np.sin(x1[0]) * np.sin(x2[0]) + np.cos(x1[0]) * np.cos(x2[0]) * np.cos(x1[1]-x2[1])

def do_estimation_standalone(TSval, beta, trials):
    r""" 
    standalone calculation of sensitivity/disc.pot.
    """
    _ub_perc = 1.
    print("\tTS    = {0:6.2f}\n".format(TSval) +
          "\tbeta  = {0:7.2%}".format(beta))
    print()

    # start estimation
    # use existing scrambles to determine best starting point
    fun = lambda n: np.log10((poisson_percentile(n,
                                                  trials["n_inj"],
                                                  trials["TS"],
                                                  TSval)[0] - beta)**2)

    # fit values in region where sampled before
    bounds = np.percentile(trials["n_inj"][trials["n_inj"] > 0],
                           [_ub_perc, 100. - _ub_perc])

    if bounds[0] == 1:
        bounds[0] = (float(np.count_nonzero(trials["n_inj"] == 1))
                        / np.sum(trials["n_inj"] < 2))

    print("\tEstimate sens. in region {0:5.1f} to {1:5.1f}".format(
                *bounds))

    # get best starting point
    ind = np.argmin([fun(n_i) for n_i in np.arange(0., bounds[-1])])

    # fit closest point to beta value
    x, f, info = scipy.optimize.fmin_l_bfgs_b(
                        fun, [ind], bounds=[bounds],
                        approx_grad=True)

    mu_eff = np.asscalar(x)

    # get the statistical uncertainty of the quantile
    b, b_err = poisson_percentile(mu_eff, trials["n_inj"],
                                        trials["TS"], TSval)

    print("\t\tBest estimate: "
          "{0:6.2f}, ({1:7.2%} +/- {2:8.3%})".format(mu_eff,
                                                     b, b_err))


    return mu_eff, b, b_err
