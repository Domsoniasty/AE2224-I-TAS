import numpy as np

file = open('data_exported_full.npy', 'rb')
load_data = np.load(file)
file.close()

print(load_data)