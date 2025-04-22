import xarray as xr
import math

def get_potential_temp(dataset, time, lat, long):
    temperature = dataset["t"].isel(time=time, latitude=lat, longitude=long).values
    pressure = dataset["isobaricInhPa"].values
    specific_humidity = dataset["q"].isel(time=time, latitude=lat, longitude=long).values
    temperature_pressure_pairs = list(zip(pressure, temperature, specific_humidity))
    pot_temp = []
    for p, t, q in temperature_pressure_pairs:
        p_t = (t * (1000/p) ** 0.286)
        v_p_t = p_t * (1 + 0.61 * q)
        #pot_temp.append(v_p_t)
        pot_temp.append(v_p_t)
    return pot_temp


def get_variable(dataset, var, pressure_lvl, lat, long):

    # Getting dataset
    ds = xr.open_dataset(dataset)

    # Getting indices for presseure levels, latitudes and longitudes
    index_pressure_lvl = math.ceil(abs(pressure_lvl - ds['pressure_level'].values[0]) / 25)
    index_lat = math.ceil(abs(lat - ds['latitude'].values[0]) / 0.25)
    index_long = math.ceil(abs(long - ds['longitude'].values[0]) / 0.25)

    # Extracting the variable using the coordinates (pressure lvl, lat, long)
    variable = ds.variables[var].data[:, index_pressure_lvl, index_lat, index_long]

    return variable

