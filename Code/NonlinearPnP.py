import scipy.optimize as opt
from scipy.spatial.transform import Rotation 
from utils import *

def ReprojectionError(X0, X, x, K):
    Q, C = X0[:4], X0[4:].reshape(-1, 1)
    R = Rotation.from_quat(Q)
    R = R.as_matrix()
    P = ProjectionMatrix(R, C, K)
    error = []
    for X_i, x_i in zip(X, x):
        p1, p2, p3 = P
        p1, p2, p3 = p1.reshape(1, -1), p2.reshape(1, -1), p3.reshape(1, -1)
        X_i = X_i.reshape(1, -1)
        X_i_homo = np.hstack((X_i, np.ones((X_i.shape[0], 1))))
        X_i_homo = X_i_homo.reshape(-1, 1)
        u, v = x_i[0], x_i[1]
        u_proj = np.divide(p1 @ X_i_homo, p3 @ X_i_homo)
        v_proj = np.divide(p2 @ X_i_homo, p3 @ X_i_homo)

        error.append(np.square(u - u_proj) + np.square(v - v_proj))

    error_sum = np.mean(np.array(error).squeeze())
    return error_sum

def NonlinearPnP(X, x_i, K, R_i, C_i):
    Q = Rotation.from_matrix(R_i)
    Q = Q.as_quat()
    X_0 = [Q[0] ,Q[1],Q[2],Q[3], C_i[0], C_i[1], C_i[2]]

    optimized_param = opt.least_squares(
        fun=ReprojectionError, x0=X_0, args=(X, x_i, K))
    X_opt = optimized_param.x
    Q_opt = X_opt[:4]
    C_opt = X_opt[4:]
    R_opt = Rotation.from_quat(Q_opt)
    R_opt = R_opt.as_matrix()
    return R_opt, C_opt