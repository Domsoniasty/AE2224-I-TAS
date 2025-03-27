import pathlib
import xarray as xr
import tempProfFitEra5 as inv
import numpy as np

# create a list of datafiles' names
def var_arrays(tInd=1, direction = -1):
    '''

    :param tInd: index of the time stamp to use; 1 is the middle (closest to gravity waves occurrence)
    :param direction: -1 is upstream, 1 is downstream, 0 is at the farm
    :return:
    '''
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
    results = np.empty((6, len(files)))
    for i in range(len(files)): # for each datafile
        # name processing
        case = files[i]
        caseSplit = case.split('_') # extract info from name
        lat = float(caseSplit[0])
        long = float(caseSplit[1])
        gravity_waves = bool(caseSplit[2])
        #timeStart = np.datetime64(timeStart.replace('(', ':')) # colons can't be used in python filenames
        #timeEnd = np.datetime64(timeEnd.replace('(', ':').replace('.nc', '')) # also remove the extension '.nc'

        # read in the data
        ds = xr.open_dataset(case)
        _, _, theta_v, invH, invThic, _, invStren, gamma, _, invrSq = inv.inversion(ds, lat, long, tInd) # time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr
        results[i] = np.array([theta_v, invH, invThic, invStren, gamma, invrSq])

