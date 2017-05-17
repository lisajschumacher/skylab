import os
import ConfigParser
import numpy as np

def load_data(basepath, filename, shuffle_bool=True):
	"""
	shuffle_bool: do or do not shuffle (currently setting to false has no effect)
	basepath: "/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub/"

	Possible filenames:
	IC40
	IC59
	IC79
	IC86
	IC86-2012
	IC86-2013
	IC86-2014

	"""
	config = ConfigParser.ConfigParser()
	config.read(os.path.join(basepath, filename+"_config.ini"))
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
