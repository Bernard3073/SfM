import numpy as np

def DisambiguateCameraPose(R_set, C_set, X_set):
    best_i = 0
    max_positive_depth_cnt = 0
    for i in range(len(R_set)):
        R, C = R_set[i], C_set[i]
        X = X_set[i]
        X = X / X[:, 3].reshape(-1, 1)
        X = X[:, :3]
        positive_depth_cnt = 0
        for x in X:
            x = x.reshape(-1, 1)
            if np.dot(R[2, :], x - C) > 0 and x[2] > 0:
                positive_depth_cnt += 1
        if positive_depth_cnt > max_positive_depth_cnt:
            best_i = i
            max_positive_depth_cnt = positive_depth_cnt
    
    R_new, C_new, X_new = R_set[best_i], C_set[best_i], X_set[best_i]
    
    return R_new, C_new, X_new