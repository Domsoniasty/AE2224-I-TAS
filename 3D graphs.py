import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from functions import get_variable  # Make sure this returns a float, not a tuple


def xaxis(dataset):
    ds = xr.open_dataset(dataset)
    return ds["latitude"].data


def yaxis(dataset):
    ds = xr.open_dataset(dataset)
    return ds["longitude"].data


path = "./data/borkum/54.0_6.50_No_8-7-2023.nc.nc"
latitudes = xaxis(path)
longitudes = yaxis(path)
pressure_lvl = 1000

# Create meshgrid for surface plot
LAT, LON = np.meshgrid(latitudes, longitudes)

# Generate temperature values for the surface (Z)
U = np.zeros(LAT.shape)
for i in range(LAT.shape[0]):
    for j in range(LAT.shape[1]):
        U[i, j] = get_variable(path, "u", pressure_lvl, LAT[i, j], LON[i, j])[1]

V = np.zeros(LAT.shape)
for i in range(LAT.shape[0]):
    for j in range(LAT.shape[1]):
        V[i, j] = get_variable(path, "v", pressure_lvl, LAT[i, j], LON[i, j])[1]

speed = np.sqrt(U**2 + V**2)

# Plotting
fig = plt.figure()

ax1 = fig.add_subplot(projection='3d')
surf = ax1.plot_surface(LAT, LON, speed, cmap='viridis')

ax1.set_xlabel('Latitude')
ax1.set_ylabel('Longitude')
ax1.set_zlabel('Temperature')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

plt.show()
