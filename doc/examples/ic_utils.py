# -*-coding:utf8-*-

import logging
import os
import json
import healpy as hp
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import ic_data
from matplotlib import gridspec
from matplotlib.colors import LogNorm

from seaborn import cubehelix_palette, set_palette
'''
# Custom colors and colormaps using seaborn
# Cubehelix palettes have a gradient in color and lightness/darkness
# Makes them look nice in in both color and gray-scale prints 
'''
colors = cubehelix_palette(6, start=.5, rot=-1.1, dark=0.15, light=0.7, reverse=True, hue=2)
cmap = cubehelix_palette(as_cmap=True, start=.5, rot=-0.9, dark=0., light=0.9, reverse=True, hue=1)
cmap.set_under("black")
cmap.set_bad("white")
#~ cmap_r = cubehelix_palette(as_cmap=True, start=.75, rot=0.5, dark=0.1, light=1., reverse=False, hue=1)
cmap_r = cubehelix_palette(as_cmap=True, start=1., rot=-0.8, dark=0.2, light=1., reverse=False, hue=1)
cmap_r.set_under("white")
cmap.set_bad("white")
set_palette(colors)
linestyles = ["-", "--", ":"]
markers = ["o", "s", "d"]

tw = 8

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
    fontsize=15
    rcParams["backend"] = backend
    rcParams["font.size"] = fontsize
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern"]
    rcParams["mathtext.fontset"] = "cm"
    rcParams["text.usetex"] = True
    rcParams["lines.linewidth"] = 2
    rcParams["figure.dpi"] = 120
    rcParams["figure.figsize"] = (tw, tw / 1.6)
    rcParams["figure.autolayout"] = True
    # Prop_cycle is new in Matplotlib 2.0
    rcParams["axes.prop_cycle"] = (cycler("color", colors*len(linestyles))
                                   + cycler("linestyle", linestyles*len(colors)))
    rcParams["axes.labelsize"] = fontsize
    rcParams["axes.titlesize"] = fontsize
    rcParams["axes.grid"] = True
    rcParams["xtick.labelsize"] = fontsize
    rcParams["ytick.labelsize"] = fontsize

    rcParams['figure.subplot.bottom'] = 0.15 # Abstand von unterem Plot Ende bis zum Rand des Bildes - nuetzlich um Achsenbeschriftung nach oben zu schieben undgroesser zu machen
    rcParams['figure.subplot.wspace'] = 0.15
    rcParams['figure.subplot.hspace'] = 0.15
    #rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.1

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
    plt.hlines(np.radians(-5.), -np.pi, np.pi, color="gray", alpha=0.75, linestyle="--", lw=1)

    cbar = fig.colorbar(p, **cb)

    cbar.solids.set_edgecolor("face")
    cbar.update_ticks()
    if title is not None:
        cbar.set_label(title)

    ax.xaxis.set_ticks([])
    plt.text(0,0, r"$180^\circ$", horizontalalignment='center')
    plt.text(np.pi+0.1, 0, r"$0^\circ$", horizontalalignment='left')

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

def aeff_vs_dec_energy(mc, logE_range=(2,9), sinDec_range=(-1,1), bins=[25,10]):
    """FROM: http://icecube.wisc.edu/~mfbaker/bootcamp/FastEffectiveArea.C
    int EBins = 32;
    double LogEMin = 1;
    double LogEMax = 9;

    if (gDirectory->FindObject(hname)) { delete gDirectory->FindObject(hname);}
    TH1D *hFastEffArea = new TH1D(hname,"Effective Area; log_{10} E / GeV; m^{2}",
                   EBins, LogEMin, LogEMax);

    double solidAngle = 4 * TMath::Pi();

    tree->SetAlias("SolidAngle",TStringify(solidAngle));

    double EBinsPerDecade = EBins/(LogEMax-LogEMin);
    tree->SetAlias("EBinsPerDecade", TStringify(EBinsPerDecade) );
    tree->SetAlias("mcLogEBin", "int(log10(mcEn)*EBinsPerDecade)");
    tree->SetAlias("mcEMin", "pow(10., mcLogEBin/EBinsPerDecade)");
    tree->SetAlias("mcEMax", "pow(10., (1+mcLogEBin)/EBinsPerDecade)");


    tree->Draw("log10(mcEn)>>"+hname, cut * 
         "1e-4*(OW/NEvents/NFiles)/"
         "(SolidAngle*(mcEMax-mcEMin))","goff");
    """
    logEEdges = np.linspace(logE_range[0],logE_range[1], bins[0], endpoint=True)
    deltaLogE = (logEEdges[-1]-logEEdges[0])/(len(logEEdges)-1)
    lowerLogEEdge = np.floor((np.log10(mc["trueE"]) - logEEdges[0])/deltaLogE)*deltaLogE + logEEdges[0]
    upperLogEEdge = lowerLogEEdge + deltaLogE
    EBinWidth = pow(10, upperLogEEdge) - pow(10, lowerLogEEdge)
    print "EBinWidth {}, MC length {}".format(len(EBinWidth), len(mc["trueE"]))

    sinDecEdges = np.linspace(sinDec_range[0],
                                                        sinDec_range[1],
                                                        bins[1], 
                                                        endpoint=True)
    deltaSinDec = (sinDecEdges[-1]-sinDecEdges[0])/(len(sinDecEdges)-1)
    solidAngle = deltaSinDec*2*np.pi
    w_aeff = mc["ow"]*10**(-4)/EBinWidth/solidAngle
    H, logEEdges, sinDecEdges = np.histogram2d(np.log10(mc["trueE"]), 
                                                 np.sin(mc["trueDec"]), 
                                                 bins=[logEEdges, sinDecEdges], 
                                                 weights=w_aeff)
    H = np.ma.masked_array(H)
    H.mask = H == 0
    center_logE_bins = np.array(logEEdges[:-1])+(logEEdges[1]-logEEdges[0])/2
    center_sinDec_bins = np.array(sinDecEdges[:-1]) + (sinDecEdges[1]-sinDecEdges[0])/2
    spline_logE_sinDec = interpolate.RectBivariateSpline(center_logE_bins, center_sinDec_bins, H, kx=3, ky=1)
    return H, logEEdges, sinDecEdges, spline_logE_sinDec
	
def plot_effective_area(mc, sinDec_range=(-1, 1), bins=[25,40], nFig=1, figsize=(20,20), cm="YlOrRd", title=""):
    plt = plotting("pdf")
    fig=plt.figure(nFig, figsize=figsize)
    gs = gridspec.GridSpec(2, 
                           2, 
                           width_ratios=[3,2],
                           height_ratios=[3,2])
    gs.update(hspace=0.08, wspace=0.05)
    sinDec_steps=np.sin(np.radians([-90, -30, -5, 30, 90]))
    hist, logEEdges, sinDecEdges, spline_logE_sinDec = aeff_vs_dec_energy(mc, 
                                                                          sinDec_range=sinDec_range, 
                                                                          bins=bins)
    Y, X = np.meshgrid(sinDecEdges, logEEdges)
    #plt.subplot(221)
    ax0=plt.subplot(gs[0])
    im=ax0.pcolormesh(X,Y, hist, norm=LogNorm(vmin=hist.min(), vmax=hist.max()), cmap=cm)
    plt.title(title) 
    plt.ylim(sinDec_range)
    plt.xlim([min(logEEdges), max(logEEdges)])
    plt.xticks([])
    plt.ylabel(r"$\sin(\delta)$")

    #plt.subplot(222)
    ax1=plt.subplot(gs[1])
    aeff_d = [quad(lambda x: spline_logE_sinDec(x,i), logEEdges[0], logEEdges[-1])[0]/(logEEdges[-1]-logEEdges[0])
              for i in (sinDecEdges[1:]+sinDecEdges[:-1])/2.]
    ax1.plot(np.array(aeff_d), (sinDecEdges[1:]+sinDecEdges[:-1])/2., color=cm.colors[0])
    plt.ylim(min(sinDecEdges), max(sinDecEdges))
    plt.semilogx(nonposx="clip")
    plt.xlabel(r"$\log_{10}(A_{eff}/ \mathrm{m}^2)$")
    plt.yticks([])
    #plt.xticks([1e1,1e2,1e3], [10,100,1000])

    #plt.subplot(223)
    ax2=plt.subplot(gs[2])
    aeff_e = [quad(lambda x: spline_logE_sinDec(i,x), sinDecEdges[0], sinDecEdges[-1])[0]/(sinDecEdges[-1]-sinDecEdges[0])
              for i in (logEEdges[1:]+logEEdges[:-1])/2.]
    ax2.plot((logEEdges[1:]+logEEdges[:-1])/2.,np.array(aeff_e), color=cm.colors[0])
    plt.xlim(min(logEEdges), max(logEEdges))
    plt.semilogy(nonposy="clip")
    plt.xlabel(r"$\log_{10}(E/\mathrm{GeV})$")
    plt.ylabel(r"$\log_{10}(A_{eff}/ \mathrm{m}^2)$")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.47, 0.025, 0.4])
    fig.colorbar(im,cax=cbar_ax, ticks=[1e-3, 1e-1, 1e1, 1e3]).set_label(r"$\log_{10}(A_{eff}/ \mathrm{m}^2)$")
    return fig, ax0, hist, logEEdges, sinDecEdges, spline_logE_sinDec
