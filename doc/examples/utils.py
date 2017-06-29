# -*-coding:utf8-*-

import logging

import healpy as hp
import numpy as np

import data

from seaborn import cubehelix_palette, set_palette
'''
# Custom colors and colormaps using seaborn
# Cubehelix palettes have a gradient in hue and darkness
# Makes them look nice in in both color and gray-scale prints 
'''
colors = cubehelix_palette(6, start=0, rot=1, dark=0.2, light=0.75, reverse=True, hue=1)
cmap = cubehelix_palette(start=0, rot=1, dark=0., light=0.9, reverse=True, hue=1, as_cmap=True)
set_palette(colors)


tw = 5.31

logging.basicConfig(level=logging.WARN)

def startup(NN=1, multi=False, **kwargs):
    n = 4
    Nexp = 10000 // NN
    NMC = 500000 // NN
    if multi:
        llh = data.multi_init(n, Nexp, NMC, ncpu=4, **kwargs)
        mc = dict([(i, data.MC(NMC)) for i in range(n)])
    else:
        llh = data.init(Nexp, NMC, ncpu=4, **kwargs)
        mc = data.MC(NMC)

    return llh, mc

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

    cbar = fig.colorbar(p, **cb)

    cbar.solids.set_edgecolor("face")
    cbar.update_ticks()
    if title is not None:
        cbar.set_label(title)

    ax.xaxis.set_ticks([])

    return fig, ax

