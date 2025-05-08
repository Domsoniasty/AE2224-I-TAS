from pygam import LogisticGAM, s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from main import var_arrays

#load_data = var_arrays() #data extracted from function in separate file


file = open('data_exported.npy', 'rb')
load_data = np.load(file)
file.close()

data_products = np.empty([60, 0])

for i in range(1, 9):
    for j in range(i+1, 9):
        data_products = np.append(data_products, np.reshape(load_data[:, i] * load_data[:, j], [60, 1]), axis=1)

load_data = np.append(load_data, data_products, axis=1)


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
    8 - vertical wind shear

'''
feature_names = ['virtual_potential_temperature', 'inversion_layer_height', 'inversion_layer_thickness', 'inversion_layer_strength', 'lapse_rate', 'wind_speed_horizontal', 'wind_speed_vertical', 'vertical wind shear']
feature_names_prods = np.empty(0)
for i in range(8):
    for j in range(i+1, 8):
        feature_names_prods = np.append(feature_names_prods, feature_names[i] + ' x ' + feature_names[j])

feature_names = np.append(feature_names, feature_names_prods)

#data_combined = np.concatenate((np.array([feature_names]), load_data), axis=0)

# df = pd.DataFrame({
#     'virtual_potential_temperature': load_data[:,1],
#     'wind_speed_horizontal':  load_data[:,6],
#     'wind_speed_vertical':  load_data[:,7],
#     'gravity_wave':load_data[:,0],
#     'inversion_layer_height': load_data[:,2],
#     'inversion_layer_strength': load_data[:,4],
#     'lapse_rate': load_data[:,5],
#     'inversion_layer_thickness': load_data[:,3]
# })

df = pd.DataFrame.from_records(load_data, columns = np.insert(feature_names, 0, 'gravity_wave'))



X = df[feature_names].values
print(X)
y = df['gravity_wave'].values


print(f'The number of data points is {len(y)}')


# Fit a GAM
'''
Both manual and automatic fitting were tried. Uncomment respective code to try

'''

#manual fitting

#gam = (LogisticGAM(
    #s(0,lam=10, n_splines=15) + s(1,lam=10, n_splines=15) + s(2,lam=10, n_splines=15) + s(3,lam=10, n_splines=15) + s(4,lam=10, n_splines=15) + s(5,lam=10, n_splines=15) + s(6,lam=10, n_splines=15)
#).fit(X,y))

#automatic fitting

gam = LogisticGAM(s((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37)))
gam.gridsearch(X, y)

# Plot the effect of each condition
fig=plt.figure(figsize=(30, 30))
rows = 6
columns = 6



# Plot each smooth function
for i, feature in enumerate(feature_names):
    fig.add_subplot(rows, columns, i + 1)
    x_grid = gam.generate_X_grid(term=i)  # Generate grid
    probability = gam.partial_dependence(term=i, X = x_grid)  # Log-odds from the GAM
    #probability = 10**(probability)/(1+10**(probability)) #converting to probability using formula from the textbook
    probability = 1 / (1 + np.exp(-probability))  # Convert to probability using sigmoid function (chat said this could be a reason)

    plt.plot(x_grid[:, i], probability)  # Plot probability curve
    plt.plot([1, 1], [2, 2])
    plt.title(f'Effect of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability of Gravity Wave')


plt.tight_layout()
plt.show()

#Finding data about data set
num_gravity_waves = (df['gravity_wave'] == 1).sum()

print(f"The size of the data set is {len(load_data)} with {num_gravity_waves} cases of gravity waves. This represents {round(num_gravity_waves*100/len(load_data),2)} % of the data set.")







