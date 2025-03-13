import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

'''
The data is stored as numpy arrays, so all numpy functions can be used on it.
'''

# Open the GRIB file
ds = xr.open_dataset("testdata2.grib", engine="cfgrib")

var_points = ds.variables['w'][:].data  # all values of 't' in the dataset
var_points_filtered = ds.variables['w'][:].data[0, 0, 0]  # [a, b, c] are indices of a: time, b: latitude, c: longitude

time = ds.variables['valid_time'][:].data  # [s]
time = (time - time[0]) / 3600  # normalise to t=0 at the first point, convert to hrs

# use pandas to present data as a table (difficult to extract data though)
df = ds['w'].to_dataframe()
print(df)
