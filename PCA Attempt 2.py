import numpy as np

def PCA_scratch(X, num_components=None):
    """
    Perform PCA on the input data using a manual implementation.

    Parameters:
    X (ndarray): The dataset to perform PCA on.
    num_components (int, optional): The number of principal components to keep. If not specified, all components will be kept.

    Returns:
    X_reduced (ndarray): The dataset projected onto the selected principal components.
    coverage (float): The proportion of variance explained by the selected principal components.
    eigenvalues (ndarray): The eigenvalues of the covariance matrix, sorted in descending order.
    eigenvectors (ndarray): The eigenvectors of the covariance matrix, sorted in descending order by the corresponding eigenvalues.
    """

    # Step 1: Center and scale the data
    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered / (np.std(X_centered, axis=0))

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(X_scaled, rowvar=False)
    # YOUR CODE HERE

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    eigenvalues_before = eigenvalues.copy()

    # YOUR CODE HERE

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # YOUR CODE HERE

    # Step 5: Select the top num_components eigenvectors, if None, keep all components
    sum_eigenvalues = np.sum(eigenvalues)

    if num_components is not None:
        eigenvectors = eigenvectors[:, :num_components]
        eigenvalues = eigenvalues[:num_components]

    # YOUR CODE HERE

    # Step 6: Transform the data onto the selected principal components
    X_reduced = np.dot(X_scaled, eigenvectors)

    # YOUR CODE HERE

    # Step 7: Compute the proportion of variance explained by the selected principal components.
    # This is done by taking the sum of the eigenvalues of the selected components and dividing by the sum of all eigenvalues.
    coverage = np.sum(eigenvalues) / sum_eigenvalues
    # YOUR CODE HERE

    return X_reduced, coverage, eigenvalues, eigenvectors, idx, eigenvalues_before

#pre-processing the data
file = open('data_exported_full.npy', 'rb')
load_file = np.load(file)
file.close()
X = load_file[:,1:]

X_reduced, coverage, eigenvalues, eigenvectors, idx, eigenvalues_before = PCA_scratch(X, num_components=None)

feature_names = [
    "virtual potential temperature",
    "inversion height",
    "inversion thickness",
    "inversion strength",
    "free atm. lapse rate",
    "horizontal wind speed",
    "vertical windspeed"
]

#print(f"The eigen values are:{eigenvalues}, for the corresponding variables {idx}")

print("")
print("In descending order the variables and their corresponding eigenvalues are:")
print("")

for y in range(len(eigenvalues)):
    print(f"The variable {feature_names[idx[y]]} had an eigenvalue of {eigenvalues[y]}")
    print("")

#print(f"The eigenvalues before sorting are {eigenvalues_before}")

