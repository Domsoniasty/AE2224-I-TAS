import pathlib
import xarray as xr
import tempProfFitEra5 as inv
import numpy as np
import functions as fnc


# Create a list of datafiles' names
def var_arrays(tInd=1, direction=-1, pressLvl=975):

    # :param tInd: index of the time stamp to use; 1 is the middle (closest to gravity waves occurrence)
    # :param direction: -1 is upstream, 1 is downstream, 0 is at the farm
    # :return: array of the following arrays (in order): bool of whether gravity waves occur, virtual pot. temp,
    # inversion height, inversion thickness, inversion strength,
    # free atm. lapse rate, horizontal windspeed, vertical windspeed

    downstream_dist = 0.25  # deg lat/long
    files = []

    # print(list(pathlib.Path('data').glob('*/*')))
    for path in list(pathlib.Path('data').glob('*/*')):
        if str(path)[-2:] == 'nc':  # str(path)[0] != '.' and
            files.append(str(path))

    results = np.empty((len(files), 9))
    rSqMax = 0
    count = 0  # count used for progress tracking of loading files
    percentage = 0  # percentage for loading count
    print_load = True
    print((len(files)))
    for i in range(len(files)):  # for each datafile

        # name processing
        case = files[i]
        print(case)

        if (count % 5) == 0:
            percentage = count*100/(len(files))
            print(f"Load is {percentage}% complete")

        count +=1

        caseSplit = case.split('\\')[-1].split('_') # extract info from name
        lat = float(caseSplit[0])
        long = float(caseSplit[1])
        lat, long = np.array([lat, long])

        try:
            caseSplit[2] = caseSplit[2].replace('.nc', '')
        except:
            pass

        if caseSplit[2] == 'Yes':
            gravity_waves = True
        else:
            gravity_waves = False

        # timeStart = np.datetime64(timeStart.replace('(', ':')) # colons can't be used in python filenames
        # timeEnd = np.datetime64(timeEnd.replace('(', ':').replace('.nc', '')) # also remove the extension '.nc'

        # read in the data
        ds = xr.open_dataset(case)

        # time, z, theta_v, invH, invThic, blTemp, invStren, gamma, sumSqDifArr
        _, _, theta_vArr, invH, invThic, _, invStren, gamma, _, invrSq = inv.inversion(ds, lat, long, tInd)

        theta_v = theta_vArr[0][int((1000-pressLvl)/25)]
        uArr = fnc.get_variable(case, 'u', pressLvl, lat, long)[tInd]
        vArr = fnc.get_variable(case, 'v', pressLvl, lat, long)[tInd]
        horSpeed = np.sqrt(uArr**2 + vArr**2)
        verSpeed = fnc.get_variable(case, 'w', pressLvl, lat, long)[tInd]
        verShear = np.sqrt(fnc.get_variable(case, 'u', 850, lat, long)[tInd]**2 + fnc.get_variable(case, 'v', 850, lat, long)[tInd]**2) - np.sqrt(fnc.get_variable(case, 'u', 500, lat, long)[tInd]**2 + fnc.get_variable(case, 'v', 500, lat, long)[tInd]**2)
        results[i] = np.array([gravity_waves, theta_v, invH[0], invThic[0], invStren[0], gamma[0], horSpeed, verSpeed, verShear])
        # print(verShear)
        if invrSq[0] > rSqMax:
            rSqMax = invrSq[0]
        print('r2:', invrSq)
    export = open('data_exported.npy', 'wb')
    np.save(export, results)
    export.close()
    return results

var_arrays()

