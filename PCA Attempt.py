import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def apply_pca(X_train):
    # TODO: modify {n_components} with the lowest number of principal components that

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    n_components = 7

    # the sklearn {PCA} tool
    pca = PCA(n_components)

    # TODO: fit the {PCA} on the training set
    # ...
    pca.fit(X_train_scaled)

    # get the explained variance of each of the {n_components} components
    explained_variance_components = pca.explained_variance_ratio_

    # TODO: modify the line below to calculate the total {explained_variance}
    # of all components
    explained_variance = np.sum(explained_variance_components)

    # report
    print(f"Explained variance = {explained_variance}")

    # return the fitted {pca}
    return pca



def train_test_validation_split(X, y, test_size, cv_size):
    # collective size of test and cv sets
    test_cv_size = test_size+cv_size

    # split data into train and test - cross validation subsets
    X_train, X_testcv, y_train, y_testcv = train_test_split(
        X, y, test_size=test_cv_size, random_state=0, shuffle=True)

    # split test - cross validation sets into test and cross validation subsets
    X_test, X_cv, y_test, y_cv = train_test_split(
        X_testcv, y_testcv, test_size=cv_size/test_cv_size, random_state=0, shuffle=True)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]

#pre-processing the data
file = open('data_exported_full.npy', 'rb')
load_file = np.load(file)
file.close()
X = load_file[:,1:]
print(X)
y = load_file[:,0]

""""
The names of the variables in position X
    0 = virtual potential temperature
    1 = inversion height
    2 = inversion thickness
    3 = inversion strength
    4 = free atm. lapse rate
    5 = horizontal wind speed
    6 = vertical windspeed
"""

[X_train, y_train, X_test, y_test, X_cv, y_cv] = train_test_validation_split(X, y, 0.3, 0.2)

pca = apply_pca(X_train)


#Obtaining the pca components with highest "weights"
weights = np.abs(pca.components_)
pc_index = 0
# Sort feature indices by strength (importance)
top_features_indices = np.argsort(weights[pc_index])[::-1]


feature_names = [
    "virtual potential temperature",
    "inversion height",
    "inversion thickness",
    "inversion strength",
    "free atm. lapse rate",
    "horizontal wind speed",
    "vertical windspeed"
]
# Print top 5 features for PC1
print(f"Top features for principal component with highest variance:")
for idx in top_features_indices:
    print(f"- {feature_names[idx]} (loading = {pca.components_[pc_index, idx]:.3f})")

