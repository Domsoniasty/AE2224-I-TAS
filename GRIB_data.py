import xarray as xr
import matplotlib.pyplot as plt

# Open the GRIB file
ds = xr.open_dataset("testdata1.grib", engine="cfgrib")

# Print dataset information
#print(ds)

# Access a variable
temperature = ds['t']  # Example: 2-meter temperature
#print(temperature)

latitude = ds.variables['latitude'][:]
#print(latitude)


# Convert to pandas DataFrame
df = temperature.to_dataframe()
print(df)
