#!/bin/env python
import os
import time
from argparse import ArgumentParser
from yaml import load as yload
from yaml import dump as ydump
from ic_utils import get_paths
from socket import gethostname
from numpy import linspace, pi, concatenate, radians


def get_paths(hostname):
    r""" Set paths depending on hostname
    Returns : path, executable
    """
    if "physik.rwth-aachen.de" in hostname:
        path = "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/skylab_stacking"
        executable = os.path.join(path, "dag_scripts", "rwth_executable.sh")
	dag_path = os.path.join(path, "dag_files")
	script_path = "/home/home2/institut_3b/lschumacher/phd_stuff/skylab_git/doc/examples"
        iwd_path = "/net/scratch_icecube4/user/lschumacher/"
	log_scratch = os.path.join(iwd_path, "projects")
        log_path = os.path.join(log_scratch, "dagman/logs")
	
    elif "icecube.wisc.edu" in hostname:
	iwd_path = "/home/lschumacher/"
        path = os.path.join(iwd_path, "git_repos/general_code_repo/skylab_stacking")
	executable = os.path.join(path, "dag_scripts", "madison_executable.sh")
	dag_path = os.path.join(path, "dag_files")
        script_path = os.path.join(iwd_path, "git_repos/skylab/doc/examples")
        log_path = "/data/user/lschumacher/projects/stacking/dag_logs"
        log_scratch = "/scratch/lschumacher/dag_files/logs"
    else:
        print("Unknown Host, please go to this function and set your paths accordingly")
        raise NotImplementedError(":(")
    return path, script_path, executable, dag_path, iwd_path, log_path, log_scratch

#~ scripts = {
#~ "bckg": "prior_bg_trials_yaml.py",
#~ "signal": "prior_signal_trials_yaml.py"
#~ }


parser = ArgumentParser()
path, script_path, executable, dag_path, iwd_path, log_path, log_scratch = get_paths(gethostname())
script = os.path.join(script_path, "prior_declination_trials_yaml.py")
typ = "dec"

parser.add_argument("yaml_file", type=str, help="yaml file for setting parameters")
yaml_file = parser.parse_args().yaml_file
yaml_dict = yload(file(yaml_file))
#~ yaml_gen = yaml_dict.pop('start', None)

if not "test" in yaml_dict["add"].lower(): identifier = "_".join([str(int(time.time())), yaml_dict["add"]])
else: identifier = yaml_dict["add"]

dag_names = []
single_dag_name = "trial_submit.dag"
single_submit_file_name = "trial_submit.job"

declinations = linspace(-pi/2., pi/2., num=yaml_dict["jobs"]+1-20)
declinations = (declinations[:-1]+declinations[1:])*0.5
declinations = concatenate([declinations, linspace(radians(-10), 0, 20)])

yaml_out_dict = yaml_dict.copy()
#~ yaml_out_dict.update(yd)
ident = "_".join([typ, identifier]) #str(key), 
yaml_out_dict["add"] = ident
dag_name = "_".join([ident, single_dag_name])

print "create dags..."

submit_file_name = "_".join([ident, single_submit_file_name])
memory = str(yaml_out_dict.pop("mem", 2000))

output_file = ident + ".out"
error_file = ident + ".err"

with open(os.path.join(dag_path, dag_name), "w") as f:
    for i in xrange(yaml_out_dict.pop("jobs", 1)):
	yaml_out_file = os.path.join(dag_path, "{}_{}.yaml".format(ident, i))
	yaml_out_dict["src_dec"] = float(declinations[i])
	print yaml_out_dict["src_dec"]
	with open(yaml_out_file, "w") as yf:
	    ydump(yaml_out_dict, yf, default_flow_style=False)
    
	os.chmod(yaml_out_file, 0754)
	job_name = "trials_{0}_{1}".format(ident, i)
	f.write("JOB {} {}\n".format(job_name, submit_file_name))
	f.write("VARS {0} script=\"{1}\" yaml=\"{2}\" job=\"{3}\" \n".format(job_name, script, yaml_out_file, i))
os.chmod(os.path.join(dag_path, dag_name), 0754)
    
submit_file = """executable = """ + executable + """
Iwd = """ + iwd_path + """ 
universe = vanilla
notification = Error
request_memory = """ + memory + """M
log = """ + log_scratch + """/$ENV(USER).condor.submit
arguments = $(script) $(yaml) $(job)
output = """ + os.path.join(log_path, output_file) + """
error = """ + os.path.join(log_path, error_file) + """
queue
"""

with open(os.path.join(dag_path,submit_file_name), "w") as f:
    f.write(submit_file)
os.chmod(os.path.join(dag_path,submit_file_name), 0754)

print("Current dag name: ", dag_name)
#~ with open(os.path.join(path, "dag_files", "dag_names.txt"), "w") as f:
    #~ for dag_name in dag_names:
	#~ f.write(dag_name+"\n")
#~ os.chmod(os.path.join(path, "dag_files", "dag_names.txt"), 0754)
