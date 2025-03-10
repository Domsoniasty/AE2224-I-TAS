import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

data = nc.Dataset('testdata3.nc') # read data

'''
to extract data:    data.variables['t'][:, :].data[0][0] (2D array)
                    data.variables['t'][:].data[:, 0, :] results in a 2D grid for each timestamp
                    
'''
ys = data.variables['t'][:].data[:, 0, 0, 0]  # temp at 8 different timestamp for point (0, 0)
xs = data.variables['valid_time'][:].data
xs = (xs - xs[0])/3600 # normalise to t=0 at the first point, convert to hrs

plt.plot(xs, ys)
plt.show()

