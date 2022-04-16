import numpy as np

def ExtractCameraPose(E):
    U, D, V = np.linalg.svd(E)
    V = V.T
    W = np.reshape([0, -1, 0, 1, 0, 0, 0, 0, 1], (3, 3))
    C_1 = U[:, 2]
    R_1 = U @ W @ V.T
    C_2 = -U[:, 2]
    R_2 = U @ W @ V.T
    C_3 = U[:, 2]
    R_3 = U @ W.T @ V.T
    C_4 = -U[:, 2]
    R_4 = U @ W.T @ V.T

    if np.linalg.det(R_1) < 0:
        R_1 = -R_1
        C_1 = -C_1
    if np.linalg.det(R_2) < 0:
        R_2 = -R_2
        C_2 = -C_2
    if np.linalg.det(R_3) < 0:
        R_3 = -R_3
        C_3 = -C_3
    if np.linalg.det(R_4) < 0:
        R_4 = -R_4
        C_4 = -C_4

    C_1 = C_1.reshape((3, 1))
    C_2 = C_2.reshape((3, 1))
    C_3 = C_3.reshape((3, 1))
    C_4 = C_4.reshape((3, 1))

    return [R_1, R_2, R_3, R_4], [C_1, C_2, C_3, C_4]