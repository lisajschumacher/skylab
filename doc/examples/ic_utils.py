# -*-coding:utf8-*-

import logging
import os
import json
import healpy as hp
import numpy as np

import ic_data

from seaborn import cubehelix_palette, set_palette
'''
# Custom colors and colormaps using seaborn
# Cubehelix palettes have a gradient in color and lightness/darkness
# Makes them look nice in in both color and gray-scale prints 
'''
colors = cubehelix_palette(6, start=.5, rot=-1.1, dark=0.15, light=0.7, reverse=True, hue=2)
cmap = cubehelix_palette(as_cmap=True, start=.5, rot=-0.9, dark=0.05, light=0.9, reverse=True, hue=1)
cmap.set_under("black")
set_palette(colors)


tw = 5.31

logging.basicConfig(level=logging.WARN)

def startup(basepath, inipath, seed=0, multi=False, n_samples=2, **kwargs):
    ic_data.set_seed(seed)
    if multi:
        """ Initialize multiple data sets: MC and EXP
        Initialize injector with multiple data sets
        Generate Multi LLH Object
        - load data -> mc, exp
        - multi_init -> llh (? not useful here ?)
            -- init -> generate single LLH objects
        """
        if n_samples > 7: n_samples = 7
        if n_samples < 2: n_samples = 2
        llh, injector = ic_data.multi_init(n_samples, basepath, inipath, **kwargs)
    else:
        """ Initialize single data sets: MC and EXP
        Initialize injector with one
        Generate Normal LLH Object
        """
        filename = kwargs.pop("filename", "IC86")
        llh, injector = ic_data.single_init(filename, basepath, inipath, **kwargs)

    return llh, injector

def plotting(backend="QT4Agg"):
    import matplotlib as mpl
    # The cycler is included in Matplotlib installations > 2.0
    # Needed for color and other style cyclings
    from cycler import cycler

    # This does not work with Matplotlib 2.0
    #if backend is not None:
    #    mpl.use(backend)

    # Start with default settings
    mpl.rcdefaults()
    rcParams = dict()
    # ... better set backend as rcParam
    rcParams["backend"] = backend
    rcParams["font.size"] = 10
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern"]
    rcParams["mathtext.fontset"] = "cm"
    rcParams["text.usetex"] = True
    rcParams["lines.linewidth"] = 1.1
    rcParams["figure.dpi"] = 72.27
    rcParams["figure.figsize"] = (tw, tw / 1.6)
    rcParams["figure.autolayout"] = True
    # Prop_cycle is new in Matplotlib 2.0
    rcParams["axes.prop_cycle"] = cycler("color", colors)
    rcParams["axes.labelsize"] = 10
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10

    mpl.rcParams.update(rcParams)

    import matplotlib.pyplot as plt

    return plt

def skymap(plt, vals, **kwargs):
    fig, ax = plt.subplots(subplot_kw=dict(projection="aitoff"))

    gridsize = 1000

    x = np.linspace(np.pi, -np.pi, 2 * gridsize)
    y = np.linspace(np.pi, 0., gridsize)

    X, Y = np.meshgrid(x, y)

    r = hp.rotator.Rotator(rot=(-180., 0., 0.))

    YY, XX = r(Y.ravel(), X.ravel())

    pix = hp.ang2pix(hp.npix2nside(len(vals)), YY, XX)

    Z = np.reshape(vals[pix], X.shape)

    lon = x[::-1]
    lat = np.pi /2.  - y

    cb = kwargs.pop("colorbar", dict())
    cb.setdefault("orientation", "horizontal")
    cb.setdefault("fraction", 0.075)

    title = cb.pop("title", None)

    p = ax.pcolormesh(lon, lat, Z, **kwargs)
    plt.hlines(np.radians(-5.), -np.pi, np.pi, color="gray", alpha=0.75, linestyle="--")

    cbar = fig.colorbar(p, **cb)

    cbar.solids.set_edgecolor("face")
    cbar.update_ticks()
    if title is not None:
        cbar.set_label(title)

    ax.xaxis.set_ticks([])

    return fig, ax

def get_paths(hostname):
    r""" Set paths depending on hostname
    Returns : basepath, inipath, savepath, crpath
    """
    if "physik.rwth-aachen.de" in hostname:
        basepath="/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub"
        inipath="/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub"
        savepath = "/net/scratch_icecube4/user/lschumacher/projects/stacking/hotspot_fitting"
        crpath = "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data"
        figurepath = "/home/home2/institut_3b/lschumacher/Pictures/uhecr_correlation/plots"

    elif "M16" in hostname:
        savepath = "/home/icecube/Documents/test_data/hotspot_fitting"
        basepath="/home/icecube/Documents/DataPS"
        inipath="/home/icecube/Documents/DataPS"
        crpath = "/home/icecube/general_code_repo/data"
        figurepath = "/home/icecube/Pictures/plots"
        
    elif "icecube.wisc.edu" in hostname:
        basepath = "/data/user/coenders/data/MultiYearPointSource/npz"
        inipath = "/data/user/lschumacher/config_files_ps"
        savepath = "/data/user/lschumacher/projects/stacking/hotspot_fitting"
        crpath = "/home/lschumacher/git_repos/general_code_repo/data"
        figurepath = "/home/lschumacher/plots"
    else:
        print("Unknown Host, please go to this function and set your paths accordingly")
        return None
    return basepath, inipath, savepath, crpath, figurepath

def prepare_directory(direc, mode=0754):
    """ 
    Prepare directory for writing
    Make directory if not yet existing
    Change chmod to desired value, default is 0754
    """
    if (not(os.path.exists(direc))):
        os.makedirs(direc)
        os.chmod(direc, mode)

def save_json_data(data, path, name):
    r"""Save serializable data (e.g. dictionary) with json"""
    with open(os.path.join(path,name+'.json'), 'w') as fp:
        json.dump(data, fp)
