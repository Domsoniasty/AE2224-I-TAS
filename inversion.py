import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from functions import get_potential_temp

ds = xr.open_dataset("gemini-testdata.grib", engine="cfgrib")

def h_inversion(ds, long, lat, time_start=None, time_stop=None):
    #t = ds.variables['t'][:].data[time_start:time_stop, :, long, lat][0] # temperatures at given time stamps, all pressure levels, single point in space
    t = get_potential_temp(ds, time_start, lat, long)
    dt = np.gradient(t) # derivative of t
    p = ds.variables['isobaricInhPa'][:].data # pressure levels

    # plt.plot(dt, np.linspace(0,1, len(t)))
    # plt.axvline(0)
    # plt.show()
    plt.plot(t, p)
    plt.show()
#print(h_inversion(ds, 0, 0, 0, 1))
for i in range(3):
    for j in range(4):
        h_inversion(ds, j, 0, i, i+1)