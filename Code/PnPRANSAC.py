from LinearPnP import *
from utils import *

def error_func(X, x, R, C, K):
    u, v = x[0], x[1]
    X = X.reshape(1, -1)
    X = np.hstack((X, np.ones((X.shape[0], 1)))).reshape(-1, 1)
    p1, p2, p3 = ProjectionMatrix(R, C, K)
    u_proj = np.divide(p1.dot(X) , p3.dot(X))
    v_proj =  np.divide(p2.dot(X) , p3.dot(X))

    x_proj = np.hstack((u_proj, v_proj))
    x = np.hstack((u, v))
    error = np.linalg.norm(x - x_proj)

    return error

def PnPRANSAC(X, x, K):
    num_iter = 1000
    max_inliers = 0
    best_C = np.zeros(3)
    best_R = np.identity(3)
    error_threshold = 5
    max_inliers = 0
    for i in range(num_iter):
        # choose 6 correspondences, X and x, randomly
        idx = np.random.choice(X.shape[0], 6, replace = False)
        X_sample = X[idx, :]
        x_sample = x[idx, :]
        R_i, C_i = LinearPnP(X_sample, x_sample, K)
        # compute the reprojection error
        inliers = 0
        if R_i is not None:
            for j in range(x.shape[0]):
                x_j = x[j, :]
                X_j = X[j, :]
                error = error_func(X_j, x_j, R_i, C_i, K)
                if error < error_threshold:
                    inliers += 1

        if inliers > max_inliers:
            max_inliers = inliers
            best_R = R_i
            best_C = C_i

    return best_R, best_C