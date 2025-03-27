import pathlib
import xarray as xr
import tempProfFitEra5 as inv
import numpy as np
import netCDFreadingdata as ncr


# Create a list of datafiles' names
def var_arrays(tInd=1, direction = -1, pressLvl=975):

    '''
    :param tInd: index of the time stamp to use; 1 is the middle (closest to gravity waves occurrence)
    :param direction: -1 is upstream, 1 is downstream, 0 is at the farm
    :return: array of the following arrays (in order): bool of whether gravity waves occur, virtual pot. temp, inversion height, inversion thickness, inversion strength,
    free atm. lapse rate, horizontal windspeed, vertical windspeed
    '''

    downstream_dist = 0.25 # deg lat/long
    files = []
    for path in list(pathlib.Path('').iterdir()):
        if str(path)[0] != '.' and str(path)[-2:] == 'nc':
            files.append(str(path))

    # theta_vArr = np.empty(len(files))
    # invHArr = np.empty(len(files))
    # invThicArr = np.empty(len(files))
    # invStrenArr = np.empty(len(files))
    # gammaArr = np.empty(len(files))
    # rSqArr = np.empty(len(files))
    results = np.empty((8, len(files)))
    for i in range(len(files)): # for each datafile
        # name processing
        case = files[i]
        caseSplit = case.split('_') # extract info from name
        lat = float(caseSplit[0])
        long = float(caseSplit[1])
        lat, long = np.array([lat, long]) + direction*get_direction(case)
        gravity_waves = bool(caseSplit[2])
        #timeStart = np.datetime64(timeStart.replace('(', ':')) # colons can't be used in python filenames
        #timeEnd = np.datetime64(timeEnd.replace('(', ':').replace('.nc', '')) # also remove the extension '.nc'

        # read in the data
        ds = xr.open_dataset(case)
        _, _, theta_v, invH, invThic, _, invStren, gamma, _, invrSq = inv.inversion(ds, lat, long, tInd) # time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr
        uArr = ncr.get_variable(case, 'u', pressLvl, lat, long)[tInd]
        vArr = ncr.get_variable(case, 'v', pressLvl, lat, long)[tInd]
        horSpeedArr = np.sqrt(uArr**2 + vArr**2)
        verSpeedArr = ncr.get_variable(case, 'w', pressLvl, lat, long)[tInd]
        results[i] = np.array([gravity_waves, theta_v, invH, invThic, invStren, gamma, horSpeedArr, verSpeedArr])


