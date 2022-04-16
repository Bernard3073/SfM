import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    E = K.T @ F @ K
    U, S, V = np.linalg.svd(E)
    # Due to the noise in K, the singular values of E are not necessarily (1, 1, 0)
    # This can be corrected by reconstructing it with (1, 1, 0) singular values
    S[0] = 1
    S[1] = 1
    S[2] = 0
    S_new = np.diag(S)
    E_new = U @ S_new @ V
    return E_new