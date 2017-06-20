# coding: utf-8

from __future__ import print_function

# python packages
import os
import glob
import socket

import numpy as np

# scipy-project imports
from scipy.special import erf, erfinv, erfc

# other
import pickle
import json

# own stuff
from skylab.utils import delta_chi2
from my_functions import do_estimation_standalone


# Set here where to read the trial files from, see also stack_multi_trials.py
if "physik.rwth-aachen.de" in socket.gethostname():
    path = "/net/scratch_icecube4/user/lschumacher/projects/stacking"
    save = "/home/home2/institut_3b/lschumacher/Pictures/uhecr_correlation/own_results"
    read_from = "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data/"
    if "lx3b74" in socket.gethostname():
        # Local scratch
        basepath="/user/scratch/lschumacher/ps_data"
    else:
        basepath="/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub"
elif "M16" in socket.gethostname():    
    path = "/home/icecube/Desktop/pScratch4/lschumacher/projects/stacking"
    save = "/home/icecube/Desktop/pHome/Pictures/uhecr_correlation/own_results"
    basepath="/home/icecube/Desktop/pScratch4/lschumacher/projects/data/ps_sample/coenders_pub"
    read_from = "/home/icecube/Desktop/pHome/phd_stuff/phd_code_git/data/"
elif "icecube.wisc.edu" in socket.gethostname():
    path = "/data/user/lschumacher/projects/stacking"
    save = "/home/lschumacher/public_html/uhecr_stacking"
    read_from = "/home/lschumacher/git_repos/general_code_repo/data"
else:
    raise("Unknown host")


# Set parameters for evaluation, i.e. choose which files you want to read
detectors=["IC86", "IC79", "IC59", "IC40"]
e_thresh = [70, 75, 80, 85, 90, 95]
md = [3,6]
info_dict = dict()
info_dict['signal'] = dict()
info_dict['dc'] = dict()
additional = "rayleigh_"


for m in md:
    info_dict['signal'][m] = dict()
    info_dict['dc'][m] = dict()
    for et in e_thresh:
        
        bg_inputfiles = sorted(glob.glob(os.path.join(path, 
                        "bg_md{}_i100_detIC86-IC79-IC59-IC40_mu0.0_et{}/trials_job*".format(m, et))))
        
        trials = []
        for ipf in bg_inputfiles:
            trials.append(np.genfromtxt(ipf, names=True))
        trials = np.array(trials).flatten()
        info_dict['signal'][m][et] = np.zeros((0, ), dtype=trials.dtype)
        info_dict['dc'][m][et] = None
        if len(trials)==0: continue

        info_dict['dc'][m][et]=delta_chi2(trials["TS"])
        ts_vals = np.array([0.1, erfc(3./np.sqrt(2)), erfc(5./np.sqrt(2))])
        ts_alpha = info_dict['dc'][m][et].isf(ts_vals)
            
        s_inputpath = sorted(glob.glob(os.path.join(path, 
        							additional+"md{0}_i100_detIC86-IC79-IC59-IC40_mu*_et{1}".format(m, et))))
        
        for s,sp in enumerate(s_inputpath):
            mu = float(sp.split('_')[-2][2:])
            if mu == 0: continue
            s_inputfiles = sorted(glob.glob(os.path.join(sp, "trials_job*")))
            signal_trials = []
            for sif in s_inputfiles:
                signal_trials.append(np.genfromtxt(sif, names=True))
            signal_trials = np.asarray(signal_trials, dtype=trials.dtype).flatten()
            """ 
            print("Mean nsources: ", np.mean(signal_trials['n_inj']))            
            print("Median nsources: ", np.median(signal_trials['n_inj']))
            print()
            print("Mean flux-per-source: ", np.mean(signal_trials['flux']))
            print("Median flux-per-source: ", np.median(signal_trials['flux']))
            """
            info_dict['signal'][m][et] = np.append(info_dict['signal'][m][et], signal_trials) 
            
# Use the stand-alone version of do_estimation from psLLH
info_dict['mu_eff'] = dict()
info_dict['beta'] = dict()
info_dict['beta_err'] = dict()
info_dict['flux_eff']= dict()
for m in md:
    info_dict['mu_eff'][m] = dict()
    info_dict['beta'][m] = dict()
    info_dict['beta_err'][m] = dict()
    info_dict['flux_eff'][m] = dict()
    for et in e_thresh:
    		# Signal 50 % (beta)  equals Background 5-sigma
        beta = 0.5
        # calculate sensitivity
        if len(info_dict['signal'][m][et])>0:
        		# 5-sigma = TSval
            TSval = np.asscalar(info_dict['dc'][m][et].isf(erfc(5./np.sqrt(2))))
            info_dict['mu_eff'][m][et], info_dict['beta'][m][et], info_dict['beta_err'][m][et] = do_estimation_standalone(TSval, beta, info_dict['signal'][m][et])
            print(len(info_dict['signal'][m][et]['n_inj']))
            # Translate mu_eff to flux, by interpolating given mu-flux relations
            fit = np.poly1d(np.polyfit(info_dict['signal'][m][et]['n_inj'], info_dict['signal'][m][et]['flux'], 1))
            info_dict['flux_eff'][m][et] = fit(info_dict['mu_eff'][m][et])
        else:
            info_dict['mu_eff'][m][et] = 0
            info_dict['beta'][m][et] = 0
            info_dict['beta_err'][m][et] = 0
            info_dict['flux_eff'][m][et] = 0
print(info_dict.keys())

# Save dictionaries to pickle file
## Signal trials
with open(os.path.join(path, additional+"signal_trials_dict.pickle"), 'w') as f:
	pickle.dump(info_dict['signal'], f)
	
## Background delta_chi2 fit
with open(os.path.join(path, additional+"bckg_fit_result_dict.pickle"), 'w') as f:
	pickle.dump(info_dict['dc'], f)
	
# Save important results to human-readable json file
save_keys = ['flux_eff', 'mu_eff', 'beta', 'beta_err']
with open(os.path.join(path, additional+"result_dict.json"), 'w') as f:
	json.dump({k : info_dict[k] for k in save_keys}, f)

print("....done!")

# plot and save the flux figure (i.e. info_dict['flux_eff']), if you want to
""" 
import matplotlib.pyplot as plt
import seaborn as sns
 
plt.figure(33)

et_x = []
flux_y = []
for m in md:
    flux_y.append([])
    et_x.append([])

    for et in e_thresh:
        if info_dict['flux_eff'][m][et]!=0:
            et_x[-1].append(et)
            flux_y[-1].append(info_dict['flux_eff'][m][et])

for i in range(len(et_x)):
    plt.plot(et_x[i], 
             np.log10(flux_y[i]), 
             label=additional[:-1]+" disc. pot., flux per source, D={} deg".format(md[i]),
             marker="o",
             color=colors[i%len(colors)])
            
plt.title(r"5$\sigma$ discovery potential, average flux per source")
plt.text(77, -11, "Very Preliminary", color="red")
plt.xlim([69, 96])
plt.ylim([-12, -10.5])
plt.ylabel(r"$\log ( E^2 \mathrm{d}N/\mathrm{d}E \, [\mathrm{TeV}/\mathrm{cm}^2/\mathrm{s}^2]$")# /N_{\mathrm{Sources}})$")
plt.xlabel(r"$E_{th} \, [\mathrm{EeV}] $")
plt.legend(ncol=2)
plt.savefig(os.path.join(save, additional+"preliminary_disc_pot.png"))
"""
