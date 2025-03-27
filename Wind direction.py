import xarray as xr
import numpy as np

ds = xr.open_dataset("54.2_5_2020-07-11T01(00(00.00_2020-07-11T03(00(00.00.nc")
print (ds)
u_component = ds['u'].values
v_component = ds['v'].values
time = ds['t'].values
direction_vector = (v_component/u_component)/(np.sqrt(u_component**2 + v_component**2))
print("direction_vector")
