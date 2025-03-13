import xarray as xr
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# Open the GRIB file
ds = xr.open_dataset("windspeedtestdata.grib", engine="cfgrib")
# print(ds)

# Access a variable
time = ds['time'].values
u_component = ds['u'].values
avg_u_component = u_component.mean(axis=(1, 2))
v_component = ds['v'].values
avg_v_component = v_component.mean(axis=(1, 2))

l: int = len(avg_u_component)
wind_speed = np.ones(l)

for i in range(l):
    wind_speed[i] = math.sqrt(avg_u_component[i]**2 + avg_v_component[i]**2)

# Convert to pandas DataFrame
#df_temp = temperature.to_dataframe()
time_series = pd.to_datetime(time)
time_hours = time_series.hour
#print(df_temp)

#print(time_hours)
#print(avg_temperature)


# Plot wind speed vs. time
plt.figure(figsize=(8, 5))
plt.plot(time_hours, wind_speed, marker='o', linestyle='-', color='b', alpha=0.7)

# Labels and title
plt.xlabel("Hour of the Day")
plt.ylabel("Wind speed (m/s)")
plt.title("Wind Speed Variation Over Time")
plt.grid(True)

# Show plot
plt.show()
