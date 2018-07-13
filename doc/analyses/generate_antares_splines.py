#!/bin/env python

import numpy as np
import cPickle as pickle
from os.path import join

from glob import glob

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline


def psf2D_gamma(ra, dec, ra_0, dec_0, gamma):
    r"""
    Well defined on np.power(10, [-4.5, 0])
    ra, dec: float, position in radian
    ra_0, dec_0: float, source position in radian
    gamma : float, source spectral index
    """
    # shape 5 x 4
    b = [[6.807,6.149,3.888,0.865],
         [115.115,182.058,96.405,16.482],
         [-465.904,-789.718,-428.301,-74.354],
         [697.181,1217.255,706.056,129.503],
         [-257.106, -710.004, -432.294, -76.251]]

    # double a[5];
    a = np.array([np.sum(np.poly1d(np.flip(bj, axis=0))(gamma)) for bj in b])
    

    sindec = np.sin(dec)
    cosdec = np.sqrt(1.- sindec**2)
    cosa = np.cos(dec_0)*cosdec*np.cos(ra-ra_0) + np.sin(dec_0)*sindec
    sina = np.sqrt(1.-cosa**2)
    ang = np.degrees(np.arccos(cosa))

    if np.abs(ang<0.001): return a[0]

    num = a[0]*ang
    denom = 1. + np.sum(np.poly1d(np.flip(a, axis=0))(ang))
    return num/(sina*denom)

def generate_inv_function(x, vals):
    norm = np.sum(vals)
    f_int = np.cumsum(vals)*1./norm
    assert(abs(f_int[-1]-1.)<1e-5)
    return InterpolatedUnivariateSpline(f_int, x, k=1)

def generate_inv_functions_from_2d_hist(data, x, y, z):
    
    data_1d_x_dict = dict()
    data_1d_y_dict = dict()
    
    xval = np.unique(data[x])
    yval = np.unique(data[y])
    xbins = len(xval)
    ybins = len(yval)
    pix_y = yval[1]-yval[0]
    pix_x = xval[1]-xval[0]
    pix_area = pix_x*pix_y

    # marginalize function over y
    data_2d = data[z].reshape(xbins, ybins)
    data_1d_x = data_2d.sum(axis=1)*pix_y
    
    data_1d_x_dict["inv_spline_0"] = generate_inv_function(xval, data_1d_x)
    data_1d_x_dict["spline_0"] = InterpolatedUnivariateSpline(xval, data_1d_x, k=1)
    data_1d_x_dict["raw_0"] = [xval, data_1d_x]
    
    for i in range(xbins):
        data_1d_y = data_2d[i]
        data_1d_y_dict["inv_spline_{}".format(i)] = generate_inv_function(yval, data_1d_y)
        data_1d_y_dict["spline_{}".format(i)] = InterpolatedUnivariateSpline(yval, data_1d_y, k=1)
        data_1d_y_dict["raw_{}".format(i)] = [yval, data_1d_y]
    return data_1d_x_dict, data_1d_y_dict

if __name__=="__main__":

    ant_path = "/home/lschumacher/antares/LL_ingredients/"
    logE_range = [2.5, 6]
    sindec_range = [-1., 0.8]
    files = sorted(glob(join(ant_path, "*")))

    for f in files:
        data = np.genfromtxt(f)
        identifier = f.split("/")[-1].split(".")[0]
        logE_sindec_pdf_dict = dict()
        s_mask = np.logical_and(data[:,0]>=sindec_range[0], data[:,0]<=sindec_range[1])
        e_mask = np.logical_and(data[:,1]>=logE_range[0], data[:,1]<=logE_range[1])
        mask = np.logical_and(s_mask, e_mask)
        logE_sindec_pdf_dict["sinDec"] = data[:,0][mask]
        logE_sindec_pdf_dict["logE"] = data[:,1][mask]
        logE_sindec_pdf_dict["value"] = data[:,2][mask]
        logE_sindec_pdf_dict["value"][logE_sindec_pdf_dict["value"]<=0] = min(logE_sindec_pdf_dict["value"][logE_sindec_pdf_dict["value"]>0])
        data_1d_x_dict, data_1d_y_dict = generate_inv_functions_from_2d_hist(logE_sindec_pdf_dict, "sinDec", "logE", "value")
        
        with open("/data/user/lschumacher/projects/stacking/antares/{}_1d_x.pkl".format(identifier), "wb") as f:
            pickle.dump(data_1d_x_dict, f)

        with open("/data/user/lschumacher/projects/stacking/antares/{}_1d_y.pkl".format(identifier), "wb") as f:
            pickle.dump(data_1d_y_dict, f)        
        
        
    # Generate PSF inverse splines
    ra_0 = 0.
    dec_0 = 0.
    num = 100
    ra = np.zeros((num,))
    dec = np.logspace(-4.5, -1, num=num)
    gamma = np.arange(-2.7, -1.7, step=0.1)
    gamma = np.concatenate([gamma, [-2.19]])
    PSF_inv_splines = dict()
    for gm in gamma:
        res = np.empty((num,))
        for i,(rr,dd) in enumerate(zip(ra,dec)):
            res[i] = psf2D_gamma(rr,dd,ra_0,dec_0,gm)
        PSF_inv_splines[str(gm)] = generate_inv_function(dec, res*np.sin(dec))
            
    with open("/data/user/lschumacher/projects/stacking/antares/PSF_inv_splines.pkl", "wb") as f:
        pickle.dump(data_1d_y_dict, f) 
    
    

        
        
        
        