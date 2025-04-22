import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from functions import get_potential_temp

ds = xr.open_dataset("gemini-testdata.grib", engine="cfgrib")


def h_inversion(ds, long, lat, time_start=None, time_stop=None):
    '''
    - check the time indexing works
    :param ds:
    :param long:
    :param lat:
    :param time_start:
    :param time_stop:
    :return:
    '''
    # t = ds.variables['t'][:].data[time_start:time_stop, :, long, lat][0] # temperatures at given time stamps, all pressure levels, single point in space
    t = get_potential_temp(ds, time_start, lat, long)[:10]
    dt = np.gradient(t)  # derivative of t
    ddt = np.gradient(dt)
    p = ds.variables['isobaricInhPa'][:].data[:10]  # pressure levels; only 10 first values
    h = ds.variables['z'][:].data[time_start:time_stop, :10, long, lat][
            0] / 9.80655 / 1000  # only values for first 10 pressure levels (i.e. closest to the SL)

    # plt.plot(dt, np.linspace(0,1, len(t)))
    # plt.axvline(0)
    # plt.show()
    plt.plot(t, h)
    # plt.axis([0, 2000, 1000, -10])
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

    return (h[np.argmax(ddt)], np.argmax(ddt))  # height at which second derivative is the highest and its index


# print(h_inversion(ds, 0, 0, 0, 1))

def free_atm_lapse_rate(ds, long, lat, i_inv, time_start=None, time_stop=None):
    t = ds.variables['t'][:].data[time_start:time_stop, :10, long, lat][
        0]  # temperatures at given time stamps, all pressure levels, single point in space
    dt = np.gradient(t)
    h = ds.variables['z'][:].data[time_start:time_stop, :10, long, lat][0] / 9.80655 / 1000
    plt.plot(dt, h)
    plt.show()
    return dt[i_inv]


# free_atm_lapse_rate(ds, 0, 0, 6, 0, 1)

def inversion_new(ds, long, lat, time_start=0, time_stop=1, l=0):
    t = get_potential_temp(ds, time_start, lat, long)[:10]
    h = ds.variables['z'][:].data[time_start:time_stop, :10, long, lat][0] / 9.80655 / 1000
    delta_h = 1  # dummy value
    N = 10
    mu = (h - l) / (1.5 * delta_h)
    f = (np.tanh(mu) + 1) / 2
    g = (np.log(2 * np.cosh(mu)) + mu) / 2
    A = np.array([[N, np.sum(f), np.sum(g)], [np.sum(f), np.sum(f**2), np.sum(f*g)], [np.sum(g), np.sum(f*g), np.sum(g**2)]])
    b = np.array([[np.sum(t)], [np.sum(f*t)], [np.sum(g*t)]])
    #print(np.linalg.lstsq(A, b)[0])
    delta_h = 10
    l = 5
    #print(np.linalg.lstsq(A, b)[0])


    # def theta(z):


inversion_new(ds, 0, 0)
