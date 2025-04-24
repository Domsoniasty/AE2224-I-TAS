from pygam import LogisticGAM, s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from main import var_arrays

load_data = var_arrays() #data extracted from function in separate file


# Load data here - Different index for different data

'''
    Index locations of information

    0 = presence of gravity waves
    1 = virtual potential temperature
    2 = inversion height
    3 = inversion thickness
    4 = inversion strength
    5 = free atm. lapse rate
    6 = horizontal wind speed
    7 = vertical windspeed

'''

df = pd.DataFrame({
    'virtual_potential_temperature': load_data[:,1],
    'wind_speed_horizontal':  load_data[:,6],
    'wind_speed_vertical':  load_data[:,7],
    'gravity_wave':load_data[:,0],
    'inversion_layer_height': load_data[:,2],
    'inversion_layer_strength': load_data[:,4],
    'lapse_rate': load_data[:,5],
    'inversion_layer_thickness':load_data[:,3]
})

X = df[['virtual_potential_temperature', 'wind_speed_horizontal',
        'wind_speed_vertical', 'inversion_layer_height',
        'inversion_layer_strength', 'lapse_rate', 'inversion_layer_thickness']].values

y = df['gravity_wave'].values




# Fit a GAM
'''
Both manual and automatic fitting were tried. Uncomment respective code to try

'''

#manual fitting

#gam = (LogisticGAM(
    #s(0,lam=10, n_splines=15) + s(1,lam=10, n_splines=15) + s(2,lam=10, n_splines=15) + s(3,lam=10, n_splines=15) + s(4,lam=10, n_splines=15) + s(5,lam=10, n_splines=15) + s(6,lam=10, n_splines=15)
#).fit(X,y))

#automatic fitting

gam = (LogisticGAM(
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6)
))
gam.gridsearch(X, y)

# Plot the effect of each condition
fig=plt.figure(figsize=(10,7))
rows = 3
columns = 3

feature_names = ['virtual_potential_temperature', 'wind_speed_horizontal',
                 'wind_speed_vertical', 'inversion_layer_height',
                 'inversion_layer_strength', 'lapse_rate', 'inversion_layer_thickness']

# Plot each smooth function
for i, feature in enumerate(feature_names):
    fig.add_subplot(rows, columns, i + 1)
    x_grid = gam.generate_X_grid(term=i)  # Generate grid
    probability = gam.partial_dependence(term=i, X = x_grid)  # Log-odds from the GAM
    #probability = 10**(probability)/(1+10**(probability)) #converting to probability using formula from the textbook

    plt.plot(x_grid[:, i], probability)  # Plot probability curve
    plt.title(f'Effect of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability of Gravity Wave')


plt.tight_layout()
plt.show()