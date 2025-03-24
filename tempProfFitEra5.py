#!/usr/bin/env python3
# -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
import tempProfFitFunctions as my
#

#######################################################
# Inputs
#######################################################

# Input NetCDF file
inp_file = 'gemini-testdata.grib'

# Inputs
lat = 36.5   # latitude in deg.
long = -97.5   # longitude in deg.
hMax = 3000   # Maximum height of data used for profile fitting
# Start and stop time of data used for profile fitting
timeStart = np.datetime64('2023-08-23T10:00:00.000000000')
timeEnd = np.datetime64('2023-08-23T15:00:00.000000000')

# Step sizes in finding the optimum values
del_l=1   # Step size in m for inversion height
del2_h=1   # Step size in m for inversion thickness



#######################################################
# Outputs
#######################################################

# Full names of the output variables and their units
# time: Array of time of the profiles
# z: Array of heights (m)
# theta_v: Time- and height- series of virtual potential temperatures (K)
# invH: Time series of inversion heights (m)
# invThic: Time series of inversion thickness (m)
# blTemp: Time series of boundary layer temperature (K)
# invStren: Time series of inversion strength (K)
# gamma: Time series of free atmospheric lapse rate (K/m)
# sumSqDifArr: Time series of the sum of the squared difference (K^2)



# Find the profile fit
time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr = \
    my.findProfParam(inp_file, lat, long, hMax, del_l, del2_h, tStart=timeStart, tEnd=timeEnd)



#######################################################
# Plots
#######################################################

# Cut off value to identify good fits
cutOff_S = 2

# Plot the time series of all the parameters
fig, ax = plt.subplots(6, 1, figsize=(8,10), dpi=300, sharex=True)

ax[0].plot(time, invH)
ax[1].plot(time, invThic)
ax[2].plot(time, invStren)
ax[3].plot(time, gamma*1000)
ax[4].plot(time, blTemp)
ax[5].plot(time, sumSqDifArr)
ax[5].plot(time, cutOff_S*np.ones(time.shape), linestyle=":")

ax[5].set_xlabel('$time$')
ax[0].set_ylabel('$H_i \ (m)$')
ax[1].set_ylabel('$\Delta H \ (m)$')
ax[2].set_ylabel('$\Delta \Theta \ (K)$')
ax[3].set_ylabel('$\Gamma \ (K/km)$')
ax[4].set_ylabel('$\Theta _M \ (K)$')
ax[5].set_ylabel('$S$')

ax[5].tick_params(axis='x', labelrotation=90)



#%%

# Plot the actual ERA5 profile and the model profile
# for either one time stamp (first line below)
# or all time stamps (second line below)
tPlotInd = 0
# for tPlotInd in range(len(time)):

# Step size in z for model profile plotting
del_zPlot = 1

# Compute the model profile
zPlot = np.arange(0, hMax+del_zPlot, del_zPlot)
eta = my.calc_eta(invH[tPlotInd], invThic[tPlotInd], zPlot)
xPlot = [blTemp[tPlotInd], invStren[tPlotInd], gamma[tPlotInd]*invThic[tPlotInd]/3]
theta_eta = my.calc_theta_eta(xPlot, eta)

plt.plot(theta_v[tPlotInd, :], z, label='ERA5')
plt.plot(theta_eta, zPlot, label='Fit')

# The following plots can be helpful in better understanding the model
# To see individual parameters from the model in the plot
# plt.plot(theta_eta, invH[tPlotInd]*np.ones(theta_eta.shape), label='$H_i$', linestyle="--")
# plt.plot(theta_eta, (invH[tPlotInd] - invThic[tPlotInd]/2)*np.ones(theta_eta.shape), label='$H_{i,Bottom}$', linestyle="--")
# plt.plot(theta_eta, (invH[tPlotInd] + invThic[tPlotInd]/2)*np.ones(theta_eta.shape), label='$H_{i,Top}$', linestyle="--")
# plt.plot(blTemp[tPlotInd] + gamma[tPlotInd]*zPlot, zPlot, label='$\Gamma$', linestyle="--")
# plt.plot(blTemp[tPlotInd]*np.ones(zPlot.shape), zPlot, label='$\Theta _M$', linestyle="--")

# To compare the different methods of finding the inversion top theta_v as discussed in the paper
# plt.plot((blTemp[tPlotInd] + invStren[tPlotInd])*np.ones(zPlot.shape), zPlot, label='$\Theta _{i,Top,a}$', linestyle="--")
# plt.plot((blTemp[tPlotInd] + invStren[tPlotInd] + gamma[tPlotInd]*invThic[tPlotInd]/3)*np.ones(zPlot.shape), zPlot, label='$\Theta _{i,Top,a+b}$', linestyle="--")
# plt.plot((blTemp[tPlotInd] + invStren[tPlotInd] - gamma[tPlotInd]*invThic[tPlotInd]/3)*np.ones(zPlot.shape), zPlot, label='$\Theta _{i,Top,a-b}$', linestyle="--")

plt.xlabel('$\Theta_v \ (K)$')
plt.ylabel('$h \ (m)$')
plt.legend()

# plt.savefig(f'plots/tempProfFit{str(tPlotInd).zfill(3)}.jpeg', bbox_inches='tight')
# plt.clf()


#%%