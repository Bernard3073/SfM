import numpy as np

def LinearPnP(X_set, x_set, K):
    X_homo = np.hstack((X_set, np.ones((X_set.shape[0], 1))))
    x_homo = np.hstack((x_set, np.ones((x_set.shape[0], 1))))
    # 2D points can be normalized by 
    # the intrinsic parameter to isolate camera parameters (R, C), i.e K^(-1)x
    x_homo = np.linalg.inv(K) @ x_homo.T
    x_homo = x_homo.T
    A = []
    for i in range(X_set.shape[0]):
        X_i = X_homo[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        u, v, _ = x_homo[i]
        # x (cross) PX = 0
        c = np.array([[0, -1, v],
                    [1,  0 , -u],
                    [-v, u, 0]])
        X_tilde = np.vstack((np.hstack(( X_i, zeros, zeros)), 
                            np.hstack((zeros,   X_i, zeros)), 
                            np.hstack((zeros, zeros,   X_i))))
        a = c @ X_tilde
        if i > 0:
            A = np.vstack((A, a))
        else:
            A = a
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape((3, 4))
    R = P[:, :3]
    # Enforce Orthonormality
    U_r, _, V_rT = np.linalg.svd(R) 
    R = U_r.dot(V_rT)
    
    C = P[:, 3]
    C = - np.linalg.inv(R).dot(C)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C

    return R, C