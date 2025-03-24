import xarray as xr


ds = xr.open_dataset("gemini-testdata.grib", engine="cfgrib")
# print(ds.variables['q'])


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


print(get_potential_temp(ds, 0, 0, 0))
