from EstimateFundamentalMatrix import *

def error_func(pts_1, pts_2, F):
    # check for epipolar constraint
    error = np.abs(np.dot(pts_2.T, F.dot(pts_1.T)))
    return error

def GetInliersRANSAC(pts_1, pts_2, idx):
    num_iteration = 500
    error_threshold = 0.005
    inliers_threshold = 0
    new_idx = []
    F_best = None
    ones = np.ones((pts_1.shape[0], 1))
    pts_1_norm = np.hstack((pts_1, ones))
    pts_2_norm = np.hstack((pts_2, ones))
    i = 0
    while i < num_iteration:
        idx_rand = np.random.choice(idx.shape[0], 8, replace=False)
        pts_1_rand = pts_1[idx_rand, :]
        pts_2_rand = pts_2[idx_rand, :]
        F_mat_rand = EstimateFundamentalMatrix(pts_1_rand, pts_2_rand)
        indices = []
        if F_mat_rand is not None:
            for j in range(pts_1.shape[0]):
                error = error_func(pts_1_norm[j, :], pts_2_norm[j, :], F_mat_rand)
                if error < error_threshold:
                    indices.append(idx[j])
                    
        if len(indices) > inliers_threshold:
            inliers_threshold = len(new_idx)
            new_idx = indices
            F_best = F_mat_rand
    
        i += 1

    return F_best, new_idx