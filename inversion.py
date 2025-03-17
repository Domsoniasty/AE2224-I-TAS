import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from functions import get_potential_temp

ds = xr.open_dataset("gemini-testdata.grib", engine="cfgrib")

def h_inversion(ds, long, lat, time_start=None, time_stop=None):
    #t = ds.variables['t'][:].data[time_start:time_stop, :, long, lat][0] # temperatures at given time stamps, all pressure levels, single point in space
    t = get_potential_temp(ds, time_start, lat, long)[:10]
    dt = np.gradient(t) # derivative of t
    ddt = np.gradient(dt)
    p = ds.variables['isobaricInhPa'][:].data[:10] # pressure levels; only 10 first values
    h = ds.variables['z'][:].data[time_start:time_stop, :10, long, lat][0]/9.80655/1000 # only values for first 10 pressure levels (i.e. closest to the SL)

    # plt.plot(dt, np.linspace(0,1, len(t)))
    # plt.axvline(0)
    # plt.show()
    plt.plot(t, h)
    #plt.axis([0, 2000, 1000, -10])
    plt.xlabel(r'$\theta_v$ [K]')
    plt.ylabel(r'$h$ [km]')
    plt.grid(True)
    plt.show()

    # plot first derivative
    # plt.plot(dt, h)
    # plt.grid(True)
    # plt.show()
    #
    # plot second derivative
    # plt.plot(ddt, h)
    # plt.grid(True)
    # plt.show()

    return(h[np.argmax(ddt)]) # height at which second derivative is the highest

print(h_inversion(ds, 0, 0, 0, 1))
# for i in range(3):
#     for j in range(4):
#         h_inversion(ds, j, 0, i, i+1)

def free_atm_lapse_rate(ds, long, lat, time_start=None, time_stop=None):
    t = ds.variables['t'][:].data[time_start:time_stop, :, long, lat][0]  # temperatures at given time stamps, all pressure levels, single point in space
    dt = np.gradient(t)
    p = ds.variables['isobaricInhPa'][:].data  # pressure levels
    h = ds.variables['z'][:].data[time_start:time_stop, :, long, lat][0] / 9.80655 / 1000
    plt.plot(t[:25], h[:25])
    plt.show()

#free_atm_lapse_rate(ds, 0, 0, 0, 1)