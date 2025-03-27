import numpy as np
import xarray as xr
import math
import pandas as pd
from pandas.core.common import index_labels_to_array


'''def get_direction(dataset):
    u_components = dataset.variables['u'].data[0, :, 0, 0]
    v_components = dataset.variables['v'].data[0, :, 0, 0]

    direction = []

    for i in range(len(u_components)):
        angle = math.atan2(u_components[i], v_components[i]) * 180 / math.pi
        direction.append(angle)

    return direction'''


def get_variable(dataset, var, pressure_lvl, lat, long):

    # Getting dataset
    ds = xr.open_dataset(dataset)

    # Getting indecies for presseure levels, latitudes and longitudes
    index_pressure_lvl = math.ceil(abs(pressure_lvl - ds['pressure_level'].values[0]) / 25)
    index_lat = math.ceil(abs(lat - ds['latitude'].values[0]) / 0.25)
    index_long = math.ceil(abs(long - ds['longitude'].values[0]) / 0.25)

    # Extracting the variable using the coordinates (pressure lvl, lat, long)
    variable = ds.variables[var].data[:, index_pressure_lvl, index_lat, index_long]

    return variable

