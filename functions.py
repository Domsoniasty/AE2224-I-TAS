import xarray as xr
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("gemini-testdata.grib", engine="cfgrib")


def get_potential_temp(dataset):
    temperature = dataset["t"].isel(time=0, latitude=0, longitude=0).values
    pressure = dataset["isobaricInhPa"].values
    temperature_pressure_pairs = list(zip(pressure, temperature))
    pot_temp = []
    for p, t in temperature_pressure_pairs:
        p_t = (t * (1000/p) ** 0.286)
        pot_temp.append(p_t)
    return pot_temp


print(get_potential_temp(ds)[0])