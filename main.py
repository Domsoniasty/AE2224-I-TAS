import pathlib
import xarray as xr
import tempProfFitEra5 as inv
import numpy as np

# create a list of datafiles' names
files = []
for path in list(pathlib.Path('').iterdir()):
    if str(path)[0] != '.' and str(path)[-2:] == 'nc':
        files.append(str(path))

for case in files: # for each datafile
    # name processing
    lat, long, timeStart, timeEnd = case.split('_') # extract info from name
    timeStart = np.datetime64(timeStart.replace('(', ':')) # colons can't be used in python filenames
    timeEnd = np.datetime64(timeEnd.replace('(', ':').replace('.nc', '')) # also remove the extension '.nc'

    # read in the data
    ds = xr.open_dataset(case)
    inv_layer = inv.inversion(ds, lat, long, timeStart, timeEnd)
    print(inv_layer)

