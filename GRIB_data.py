import xarray as xr
import matplotlib.pyplot as plt

data = xr.load_dataset("testdata1.grib", engine='cfgrib')

data.t[0].plot(cmap=plt.cm.coolwarm)

#plt.show()