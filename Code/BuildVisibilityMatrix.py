import numpy as np

def BuildVisibilityMatrix(X, feature_idx, num_camera):
    # the relationship between a camera and a point
    # construct a I x J matrix V, where Vij is one
    # if the point j is visible in the camera i and zero otherwise
    zeros = np.zeros((feature_idx.shape[0]), dtype=int)
    for n in range(num_camera+1):
        zeros = zeros | feature_idx[:, n]

    X_idx = np.where((X.reshape(-1)) & (zeros))
    v_mat = X[X_idx].reshape(-1, 1)

    for n in range(num_camera+1):
        v_mat = np.hstack((v_mat, feature_idx[X_idx, n].reshape(-1, 1)))
    
    return X_idx, v_mat[:, 1:v_mat.shape[1]]