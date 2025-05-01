import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from functions import get_variable


# Getting latitude for the x-axis
def xaxis(dataset):
    ds = xr.open_dataset(dataset)
    return ds["latitude"].data


# Getting longitude for the y-axis
def yaxis(dataset):
    ds = xr.open_dataset(dataset)
    return ds["longitude"].data


# Creating 3D graphs plotting a variable vs latitude & longitude
def create_var_graph(filepath, var, p_lvl):

    # Getting lats and longs for the given file
    latitudes = xaxis(filepath)
    longitudes = yaxis(filepath)

    # Create meshgrid for surface plot
    lat, long = np.meshgrid(latitudes, longitudes)

    # Creating variable array
    var_array = np.zeros(lat.shape)

    # Using function get_variable for each lat and long, getting the data to the variable array
    for i in range(lat.shape[0]):
        for j in range(lat.shape[1]):
            var_array[i, j] = get_variable(filepath, var, p_lvl, lat[i, j], long[i, j])[1]

    # Plotting everything
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(lat, long, var_array, cmap='viridis')

    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Temperature')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


file = "./data/borkum/54.0_6.50_No_8-7-2023.nc.nc"

create_var_graph(file, "t", 1000)

plt.show()
