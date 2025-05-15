from pygam import LogisticGAM, s, LinearGAM, GAM, l
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


#from main import var_arrays

#load_data = var_arrays() #data extracted from function in separate file


file = open('data_exported.npy', 'rb')
load_data = np.load(file)
file.close()


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
    8 = vertical wind shear

'''

print(load_data)

df = pd.DataFrame({
    'virtual_potential_temperature': load_data[:,1],
    'wind_speed_horizontal':  load_data[:,6],
    'wind_speed_vertical':  load_data[:,7],
    'gravity_wave':load_data[:,0],
    'inversion_layer_height': load_data[:,2],
    'inversion_layer_strength': load_data[:,4],
    'lapse_rate': load_data[:,5],
    'inversion_layer_thickness': load_data[:,3],
    'vertical_wind_shear': load_data[:,8]
})



X = df[['virtual_potential_temperature', 'wind_speed_horizontal',
        'wind_speed_vertical', 'inversion_layer_height',
        'inversion_layer_strength', 'lapse_rate',
        'inversion_layer_thickness', 'vertical_wind_shear']].values

y = df['gravity_wave'].values


print(f'The number of data points is {len(y)}')


# Fit a GAM
'''
Both manual and automatic fitting were tried. Uncomment respective code to try

'''

lamb = 1.6
no_splines = 11
#manual fitting


#logisticGam
gam = (LogisticGAM(
    s(0,lam=lamb, n_splines=no_splines) + s(1,lam=lamb, n_splines=no_splines) + s(2,lam=lamb, n_splines=no_splines) + s(3,lam=lamb, n_splines=no_splines) + s(4,lam=lamb, n_splines=no_splines) + s(5,lam=lamb, n_splines=no_splines) + s(6,lam=lamb, n_splines=no_splines) + s(7,lam=lamb, n_splines=no_splines)).fit(X,y))
gam.summary()


'''
#LinearGam
gam = LinearGAM(s(0, n_splines = no_splines) + s(1, n_splines = no_splines) + s(2, n_splines = no_splines) + s(3, n_splines = no_splines) + s(4, n_splines = no_splines) + s(5, n_splines = no_splines) + s(6, n_splines = no_splines) + s(7, n_splines = no_splines)).fit(X,y)
gam.summary()
'''


#automatic fitting

#lams = np.exp(np.random.random(50) * 6 - 3)

'''
gam = LogisticGAM( s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) )
gam.gridsearch(X, y)
gam.summary()
'''



# Plot the effect of each condition
fig=plt.figure(figsize=(10,8))
rows = 3
columns = 3

feature_names = ['virtual_potential_temperature [K]', 'wind_speed_horizontal [m/s]',
                 'wind_speed_vertical [m/s]', 'inversion_layer_height [m]',
                 'inversion_layer_strength [K/m]', 'lapse_rate [K/m]', 'inversion_layer_thickness [km]', 'vertical_wind_shear [m/s]']

# Plot each smooth function
for i, feature in enumerate(feature_names):
    fig.add_subplot(rows, columns, i + 1)
    x_grid = gam.generate_X_grid(term=i)  # Generate grid
    probability = gam.partial_dependence(term=i, X = x_grid)  # Log-odds from the GAM
    #probability = 10**(probability)/(1+10**(probability)) #converting to probability using formula from the textbook
    probability = 1 / (1 + np.exp(-probability))  # Convert to probability using sigmoid function (chat said this could be a reason)
    #plt.scatter(df[feature_names[i]].values, y)
    plt.plot(x_grid[:, i], probability)
    #plt.scatter(df[feature_names[i]].values, y)# Plot probability curve
    plt.title(f'Effect of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability of Gravity Wave')

for i, feature in enumerate(feature_names):
    x_grid = gam.generate_X_grid(term=i)  # Generate grid
    probability = gam.partial_dependence(term=i, X = x_grid)  # Log-odds from the GAM
    probability = 10**(probability)/(1+10**(probability)) #converting to probability using formula from the textbook
    #probability = 1 / (1 + np.exp(-probability))  # Convert to probability using sigmoid function (chat said this could be a reason)
    plt.plot(x_grid[:, i], probability)  # Plot probability curve
    #plt.scatter(X[:, i], y)
    plt.title(f'Effect of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability of Gravity Wave')
    safe_filename = re.sub(r'[^\w\-_\. ]', '_', feature) + 'manual.png'
    plt.savefig(safe_filename)
    plt.show()




plt.tight_layout()
plt.show()

#Finding data about data set
num_gravity_waves = (df['gravity_wave'] == 1).sum()

print(f"The size of the data set is {len(load_data)} with {num_gravity_waves} cases of gravity waves. This represents {round(num_gravity_waves*100/len(load_data),2)} % of the data set.")







