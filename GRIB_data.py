import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Open the GRIB file
ds = xr.open_dataset("testdata3.grib", engine="cfgrib")


# Access a variable
time = ds['time'].values # time
temperature = ds['t2m'].values  # 2-meter temperature
avg_temperature = temperature.mean(axis=(1, 2))

# Convert to pandas DataFrame
#df_temp = temperature.to_dataframe()
time_series = pd.to_datetime(time)
time_hours = time_series.hour
#print(df_temp)

#print(time_hours)
#print(avg_temperature)


# Plot average temperature vs. time
plt.figure(figsize=(8, 5))
plt.plot(time_hours, avg_temperature, marker='o', linestyle='-', color='b', alpha=0.7)

# Labels and title
plt.xlabel("Hour of the Day")
plt.ylabel("Average Temperature (°C)")
plt.title("Average Temperature Variation Over Time")
plt.grid(True)

# Show plot
plt.show()
