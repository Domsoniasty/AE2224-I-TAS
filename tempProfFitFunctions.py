#!/usr/bin/env python3
# -*-

import numpy as np
import xarray as xr


def extDataFromNetCDF(ds, lat, long, hMax, tInd):
    
    # Open the NetCDF file
    # ds = xr.open_dataset(inp_file) # ds will be passed from main.py

    # Dimensions: (valid_time: 48, pressure_level: 37, latitude: 13, longitude: 13)
    # Coordinates and variables in the dataset
    # valid_time      (valid_time) datetime64[ns] 384B 2023-08-23 ... 2023-08-2...
    # pressure_level  (pressure_level) float64 296B 1e+03 975.0 950.0 ... 2.0 1.0
    # latitude        (latitude) float64 104B 38.0 37.75 37.5 ... 35.5 35.25 35.0
    # longitude       (longitude) float64 104B -99.0 -98.75 -98.5 ... -96.25 -96.0
    # d: Divergence
    # cc: Fraction of cloud cover
    # z: Geopotential
    # o3: Ozone mass mixing ratio
    # pv: Potential vorticity
    # r: Relative humidity
    # ciwc: Specific cloud ice water content
    # clwc: Specific cloud liquid water content
    # q: Specific humidity
    # crwc: Specific rain water content
    # cswc: Specific snow water content
    # t: Temperature
    # u, v, and w: Components of velocity
    # vo: Vorticity

    # Convert vertical coordinates from pressure to altitude
    # Formula was obtained from the below webpage
    # https://www.weather.gov/epz/wxcalc_pressurealtitude
    h = ( 1 - (ds.pressure_level.values/1013.25)**0.190284 ) * 145366.45 * 0.3048
    hMaxInd = np.count_nonzero(h <= hMax)
    z = h[:hMaxInd]

    # Indices of latitude and longitude of interest
    lat_i = np.argmin(abs(ds.latitude.values) - lat)
    long_i = np.argmin(abs(ds.longitude.values - long))

    # Sanity Check
    # print(f"Maximum height: {h[hMaxInd-1]} m")
    # print(f"Chosen latitude: {ds.latitude[lat_i].values}")
    # print(f"Chosen longitude: {ds.longitude[long_i].values}")

    # Extract time array
    # time = ds.valid_time.values
    # if tStart == 'Not provided':
    #     print('\nStart time is not provided!')
    #     print('Starting from the first time stamp in the dataset!')
    #     tStartInd = 0
    # else:
    #     tStartInd = np.count_nonzero(ds.valid_time.values < tStart)
    #
    # if tEnd == 'Not provided':
    #     print('\nEnd time is not provided!')
    #     print('Ending with the last time stamp in the dataset!')
    #     tEndInd = len(ds.valid_time)
    # else:
    #     tEndInd = np.count_nonzero(ds.valid_time.values <= tEnd)



    time = ds.valid_time.values[tInd:tInd+1]
    # Extract other needed variables
    ds_t = ds.t[:, :, lat_i, long_i].values
    #ds_r = ds.r[:, :, lat_i, long_i].values
    ds_r = np.zeros((len(ds_t), len(ds_t[0])))
    ds_q = ds.q[:, :, lat_i, long_i].values
    ds_clwc = ds.clwc[:, :, lat_i, long_i].values
    #ds_ciwc = ds.ciwc[:, :, lat_i, long_i].values
    ds_ciwc = np.zeros((len(ds_t), len(ds_t[0])))
    #ds_crwc = ds.crwc[:, :, lat_i, long_i].values
    ds_crwc = np.zeros((len(ds_t), len(ds_t[0])))
    #ds_cswc = ds.cswc[:, :, lat_i, long_i].values
    ds_cswc = np.zeros((len(ds_t), len(ds_t[0])))

    # Convert temperature to potential temperature
    theta = ds_t * (10*100/ds.pressure_level).values**0.286

    # Convert potential temperature to virtual potential temperature
    sat_sts = ds_r/100   # Saturation status
    np.minimum(sat_sts, 1.0, out=sat_sts)   # 1 corresponds to saturated air
    moi_dry_ratio = 1/(1 - ds_q - ds_clwc - ds_ciwc - ds_crwc - ds_cswc) # Moist air to dry air ratio
    r_sat = moi_dry_ratio * (ds_q - sat_sts * (ds_r/100 - 1) * ds_q)
    r_L = moi_dry_ratio * (ds_clwc + ds_crwc + sat_sts * (ds_r/100 - 1) * ds_q) 
    theta_v = theta * (1 + 0.61*r_sat - r_L)
    theta_v = theta_v[tInd:tInd+1, :hMaxInd]

    # Return potential temperature, height and time
    return theta_v, z, time



def calc_eta(l, del_h, z):
    # Constants
    c = 1/3
    eta = (z - l) / (c*del_h)
    return eta



def calc_f_eta(eta):
    f_eta = (np.tanh(eta) + 1) / 2
    return f_eta



def calc_g_eta(eta):
    g_eta = ( np.log( 2*np.cosh(eta) ) + eta ) / 2
    return g_eta



def calc_theta_eta(x, eta):
    theta_eta = x[0] + x[1]*calc_f_eta(eta) + x[2]*calc_g_eta(eta)
    return theta_eta



def solveForCoeff(z, theta_i, l, del_h):

    # Number of data points
    N = len(theta_i)

    # Non-dimensional height parameter
    eta = calc_eta(l, del_h, z)

    # Functions used in the model
    f_eta = calc_f_eta(eta)
    g_eta = calc_g_eta(eta)

    # Forming the system of linear algebraic equations (A6)
    # given in Rampanelli and Zardi 2004
    # Ax = b
    # where x = [theta_m   a   b]^T
    A = np.array([[N, np.sum(f_eta), np.sum(g_eta)], \
        [np.sum(f_eta), np.sum(f_eta**2), np.sum(g_eta*f_eta)], \
            [np.sum(g_eta), np.sum(f_eta*g_eta), np.sum(g_eta**2)]])

    b = np.array([np.sum(theta_i), np.sum(theta_i*f_eta), np.sum(theta_i*g_eta)])

    # Solve the system of equation and return the results
    x = np.linalg.solve(A, b)

    # Calculate the variance between the measurements and the model
    theta_eta = calc_theta_eta(x, eta)
    sumSqDif = np.sum((theta_eta - theta_i)**2)
    theta_bar = np.mean(theta_i)
    totSumSq = np.sum((theta_i-theta_bar)**2)
    rSq = 1 - sumSqDif/totSumSq
    
    return sumSqDif, x, rSq



def findProfParam(ds, lat, long, hMax, del_l, del2_h, tInd):

    # Extract height, time and temperature and compute virtual potential temperature
    theta_v, z, time = extDataFromNetCDF(ds, lat, long, hMax, tInd)

    # Initialize arrays
    invH = np.zeros(len(time))   # Inversion Height
    invThic = np.zeros(len(time))   # Inversion Thickness
    invStren = np.zeros(len(time))   # Inversion Strength
    blTemp = np.zeros(len(time))   # Boundary Layer Temperature
    gamma = np.zeros(len(time))
    sumSqDifArr = np.zeros(len(time))   # Sum of squared difference
    rSqArr = np.zeros(len(time))  # R^2

    for i in range(len(time)):

        # Virtual potential temperature
        theta_i = theta_v[i]

        # To have a value for comparison in the loop
        # The end points that the loops will ignore are calculated here
        lCurr = z[-1]
        del_hCurr = np.max( z[1:]-z[:-1] )
        sumSqDifBest, xBest, rSqBest = solveForCoeff(z, theta_i, lCurr, del_hCurr)
        lBest = lCurr
        del_hBest = del_hCurr

        # Compute the model residual for each l and del_h
        for lCurr in np.arange(z[0], z[-1], del_l):
            for del_hCurr in np.arange(np.min(z[1:]-z[:-1]), np.max(z[1:]-z[:-1]), del2_h):
                sumSqDifCurr, xCurr, rSqCurr = solveForCoeff(z, theta_i, lCurr, del_hCurr)
                # Check if this model is better
                if sumSqDifCurr < sumSqDifBest:
                    sumSqDifBest = sumSqDifCurr
                    xBest = xCurr
                    lBest = lCurr
                    del_hBest = del_hCurr

        # Save the best model parameters
        invH[i] = lBest
        invThic[i] = del_hBest
        blTemp[i] = xBest[0]
        invStren[i] = xBest[1]   # a+b gives the potential temperature jump in the Entrainment Layer,
                                    # However the paper suggests to use 'a' for inversion strength
        gamma[i] = 3 * xBest[2] / del_hBest   # The paper says 2*b/del_h, but it seems like a typo
        sumSqDifArr[i] = sumSqDifBest
        rSqArr[i] = rSqBest


    return time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr, rSqArr