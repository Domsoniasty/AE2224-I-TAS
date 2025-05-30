
import numpy as np
import matplotlib.pyplot as plt
import tempProfFitFunctions as my
import pathlib
#

#######################################################
# Inputs
#######################################################

# Input NetCDF file
#inp_file = '().nc'



# # Inputs
# lat = 36.5   # latitude in deg.
# long = -97.5   # longitude in deg.
# hMax = 3000   # Maximum height of data used for profile fitting
# # Start and stop time of data used for profile fitting
# #timeStart = np.datetime64('2023-08-23T10:00:00.000000000')
# #timeEnd = np.datetime64('2023-08-23T15:00:00.000000000')
# timeStart = 'Not provided'
# timeEnd = 'Not provided'

def inversion(ds, lat, long, tInd, hMax=3000):
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
    time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr, rSqArr = my.findProfParam(ds, lat, long, hMax, del_l, del2_h, tInd)



    #######################################################
    # Plots
    #######################################################

    # Cut off value to identify good fits
    cutOff_S = 2

    # Plot the time series of all the parameters

    # fig, ax = plt.subplots(6, 1, figsize=(8,10), dpi=300, sharex=True)
    #
    # ax[0].plot(time, invH)
    # ax[1].plot(time, invThic)
    # ax[2].plot(time, invStren)
    # ax[3].plot(time, gamma*1000)
    # ax[4].plot(time, blTemp)
    # ax[5].plot(time, sumSqDifArr)
    # ax[5].plot(time, cutOff_S*np.ones(time.shape), linestyle=":")
    #
    # ax[5].set_xlabel(r'$time$')
    # ax[0].set_ylabel(r'$H_i \ (m)$')
    # ax[1].set_ylabel(r'$\Delta H \ (m)$')
    # ax[2].set_ylabel(r'$\Delta \Theta \ (K)$')
    # ax[3].set_ylabel(r'$\Gamma \ (K/km)$')
    # ax[4].set_ylabel(r'$\Theta _M \ (K)$')
    # ax[5].set_ylabel(r'$S$')
    #
    # ax[5].tick_params(axis='x', labelrotation=90)
    # plt.show()



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

    # plt.plot(theta_v[tPlotInd, :], z, label='ERA5')
    # plt.plot(theta_eta, zPlot, label='Fit')
    # plt.legend()
    # plt.show()
    # The following plots can be helpful in better understanding the model
    # To see individual parameters from the model in the plot
    # plt.plot(theta_eta, invH[tPlotInd]*np.ones(theta_eta.shape), label=r'$H_i$', linestyle="--")
    # plt.plot(theta_eta, (invH[tPlotInd] - invThic[tPlotInd]/2)*np.ones(theta_eta.shape), label='$H_{i,Bottom}$', linestyle="--")
    # plt.plot(theta_eta, (invH[tPlotInd] + invThic[tPlotInd]/2)*np.ones(theta_eta.shape), label='$H_{i,Top}$', linestyle="--")
    # plt.plot(blTemp[tPlotInd] + gamma[tPlotInd]*zPlot, zPlot, label=r'$\Gamma$', linestyle="--")
    # plt.plot(blTemp[tPlotInd]*np.ones(zPlot.shape), zPlot, label=r'$\Theta _M$', linestyle="--")

    # To compare the different methods of finding the inversion top theta_v as discussed in the paper
    # plt.plot((blTemp[tPlotInd] + invStren[tPlotInd])*np.ones(zPlot.shape), zPlot, label='$\Theta _{i,Top,a}$', linestyle="--")
    # plt.plot((blTemp[tPlotInd] + invStren[tPlotInd] + gamma[tPlotInd]*invThic[tPlotInd]/3)*np.ones(zPlot.shape), zPlot, label='$\Theta _{i,Top,a+b}$', linestyle="--")
    # plt.plot((blTemp[tPlotInd] + invStren[tPlotInd] - gamma[tPlotInd]*invThic[tPlotInd]/3)*np.ones(zPlot.shape), zPlot, label='$\Theta _{i,Top,a-b}$', linestyle="--")

    # plt.xlabel(r'$\Theta_v \ (K)$')
    # plt.ylabel(r'$h \ (m)$')
    # plt.legend()
    # plt.show()

    # plt.savefig(f'plots/tempProfFit{str(tPlotInd).zfill(3)}.jpeg', bbox_inches='tight')
    # plt.clf()

    return time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr, rSqArr