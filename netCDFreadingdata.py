import numpy as np
import xarray as xr
import math
import pandas as pd


def get_direction(dataset):
    u_components = dataset.variables['u'].data[0,:,0,0]
    v_components = dataset.variables['v'].data[0,:,0,0]

    direction = []

    for i in range(len(u_components)):
        angle = math.atan2(u_components[i],v_components[i]) * 180 / math.pi
        direction.append(angle)

    return direction


ds = xr.open_dataset("testdata_multipledays.nc")
df = ds.to_dataframe()

print(df)

#print(ds.variables['t'].data[0,1,:,:])