import xarray as xr
import numpy as np
import math


def get_direction(dataset):
    u_components = dataset.variables['u'].data[0, :, 0, 0]
    v_components = dataset.variables['v'].data[0, :, 0, 0]

    # direction unit vector
    direction = np.arctan2(u_components, v_components)/np.sqrt(u_components**2 + v_components**2)
    return direction


# Example usage
dataset = xr.open_dataset("54.2_5_2020-07-11T01(00(00.00_2020-07-11T03(00(00.00.nc")
direction_vector = get_direction(dataset)
print(direction_vector)


