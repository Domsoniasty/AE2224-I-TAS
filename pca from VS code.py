import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


#pre-processing the data
scalar = StandardScaler()
file = open('data_exported_full.npy', 'rb')
load_file = np.load(file)
file.close()
X = load_file[:,1:]
y = load_file[:,0]


X_scaled = scalar.fit_transform(X)
pca = PCA(n_components= len(X_scaled[0]))
X_transformed = pca.fit_transform(X_scaled)


coverage_list_sklearn = np.cumsum(pca.explained_variance_ratio_)*100

plt.plot(np.arange(1, len(X[0])+1), coverage_list_sklearn, label = 'PCA from Sklearn')
plt.xlabel('Number of Principal Components')
plt.ylabel('Coverage (%)')
plt.title('PCA Coverage vs Number of Principal Components')
plt.xlim(0, len(X[0]))
plt.ylim(0, 100)
plt.legend()
plt.show()