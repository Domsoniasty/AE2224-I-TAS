import numpy as np
import xarray as xr

ds = xr.open_dataset("gemini-testdata.nc")

print(ds)
