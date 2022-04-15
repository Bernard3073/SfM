import numpy as np

def EstimateFundamentalMatrix(pts_1, pts_2):
    if pts_1.shape[0] < 8:
        return None
    A = np.zeros((len(pts_1),9))
    for i in range(0, len(pts_1)):
        x_1,y_1 = pts_1[i][0], pts_1[i][1]
        x_2,y_2 = pts_2[i][0], pts_2[i][1]
        A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

    U, S, VT = np.linalg.svd(A, full_matrices=True)
    F = VT.T[:, -1]
    F = F.reshape(3,3)

    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2,2] = 0
    F = np.dot(u, np.dot(s, vt))
    return F