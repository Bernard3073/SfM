from logging import error
from BuildVisibilityMatrix import *
from scipy.spatial.transform import Rotation 
from scipy.sparse import lil_matrix
import scipy.optimize as opt

def get2DPoints(X_idx, V_mat, feature_x, feature_y):
    pts2D = []
    visible_feature_x = feature_x[X_idx]
    visible_feature_y = feature_y[X_idx]
    h, w = V_mat.shape
    for i in range(h):
        for j in range(w):
            if V_mat[i,j] == 1:
                pt = np.hstack((visible_feature_x[i,j], visible_feature_y[i,j]))
                pts2D.append(pt)
    return np.array(pts2D).reshape(-1, 2)  

def getCameraPointIndices(V_mat):

    camera_indices = []
    point_indices = []
    h, w = V_mat.shape
    for i in range(h):
        for j in range(w):
            if V_mat[i,j] == 1:
                camera_indices.append(j) # camera index: w
                point_indices.append(i) # point index: h

    camera_indices = np.array(camera_indices).reshape(-1)
    point_indices = np.array(point_indices).reshape(-1)

    return camera_indices, point_indices

# def bundle_adjustment_sparsity(X_found, feature_idx, nCam):
def bundle_adjustment_sparsity(X_idx, V_mat, nCam):
    
    """
    To create the Sparsity matrix
    """
    number_of_cam = nCam + 1
    # X_idx, V_mat = BuildVisibilityMatrix(X_found.reshape(-1), feature_idx, nCam)
    n_observations = np.sum(V_mat)
    n_points = len(X_idx[0])

    m = n_observations * 2
    n = number_of_cam * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = getCameraPointIndices(V_mat)

    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 6 + point_indices * 3 + s] = 1

    return A

def project(points_3d, camera_params, K):
    def projectPoint_(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_4 = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_4.T)
        x_proj /= x_proj[-1]
        return x_proj

    x_proj = []
    for i in range(len(camera_params)):
        R = Rotation.from_rotvec(camera_params[i, :3]).as_matrix()
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points_3d[i]
        pt_proj = projectPoint_(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def reprojection_error(x0, num_cam, num_pts, camera_indices, point_indices, pts_2d, K):
    num_cam = num_cam + 1
    camera_params = x0[:num_cam * 6].reshape(num_cam, 6)
    pts_3d = x0[num_cam * 6:].reshape(num_pts, 3)
    pts_proj = project(pts_3d[point_indices], camera_params[camera_indices], K)
    error = (pts_proj - pts_2d).ravel()

    return error

def BundleAdjustment(X_all, X_found, K, R_set_new, C_set_new, feature_x, feature_y, new_feature_idx, num_camera):
    X_idx, V_mat = BuildVisibilityMatrix(X_found, new_feature_idx, num_camera)
    pts_3d = X_all[X_idx]
    pts_2d = get2DPoints(X_idx, V_mat, feature_x, feature_y)
    
    RC_list = []
    for i in range(num_camera+1):
        Ri, Ci = R_set_new[i], C_set_new[i]
        Qi = Rotation.from_matrix(Ri).as_rotvec()
        RC = [Qi[0], Qi[1], Qi[2], Ci[0], Ci[1], Ci[2]]
        RC_list.append(RC)
    RC_list = np.array(RC_list).reshape(-1, 6)

    x0 = np.hstack((RC_list.ravel(), pts_3d.ravel()))
    camera_indices, point_indices = getCameraPointIndices(V_mat)

    A = bundle_adjustment_sparsity(X_idx, V_mat, num_camera)
    
    res = opt.least_squares(reprojection_error, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, 
                            args=(num_camera, pts_3d.shape[0], camera_indices, point_indices, pts_2d, K))
    x1 = res.x
    num_camera = num_camera + 1
    opt_camera_params = x1[:num_camera * 6].reshape(num_camera, 6)
    opt_pts_3d = x1[num_camera * 6:].reshape((pts_3d.shape[0], 3))

    opt_X_all = np.zeros_like(X_all)
    opt_X_all[X_idx] = opt_pts_3d

    opt_R_set, opt_C_set = [], []
    for i in range(len(opt_camera_params)):
        R = Rotation.from_rotvec(opt_camera_params[i, :3]).as_matrix()
        C = opt_camera_params[i, 3:].reshape(3,1)
        opt_R_set.append(R)
        opt_C_set.append(C)
    
    return opt_X_all, opt_R_set, opt_C_set
