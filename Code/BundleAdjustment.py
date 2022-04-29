from logging import error
from BuildVisibilityMatrix import *
from scipy.spatial.transform import Rotation 
from scipy.sparse import lil_matrix
import scipy.optimize as opt

def get_2D_pts(X_idx, V_mat, feature_x, feature_y):
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

def get_camera_pts_indices(V_mat):

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

# Ref: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def bundle_adjustment_sparsity(X_idx, V_mat, nCam):
    """
    Create the sparsity matrix
    """
    number_of_cam = nCam + 1
    n_observations = np.sum(V_mat)
    n_points = len(X_idx[0])

    m = n_observations * 2
    n = number_of_cam * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    print(m, n)


    i = np.arange(n_observations)
    camera_indices, point_indices = get_camera_pts_indices(V_mat)

    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, (nCam)* 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, (nCam) * 9 + point_indices * 3 + s] = 1

    return A

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    Args:
        points (array): points to rotate
        rot_vecs (TYPE): rotation vector
    Returns:
        TYPE: rotated points
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(
        v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images.
    Args:
        points (array): 2D points
        camera_params (array): Intrinsic paramters matrix
    Returns:
        TYPE: Projected 3D points
    """
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def reprojection_error(x0, num_cam, num_pts, camera_indices, point_indices, pts_2d, K):
    num_cam = num_cam + 1
    camera_params = x0[:num_cam * 9].reshape(num_cam, 9)
    pts_3d = x0[num_cam * 9:].reshape(num_pts, 3)
    # pts_proj = project(pts_3d[point_indices], camera_params[camera_indices], K)
    pts_proj = project(pts_3d[point_indices], camera_params[camera_indices])
    error = (pts_proj - pts_2d).ravel()

    return error

def BundleAdjustment(X_3d, X_idx, V_mat, K, R_set_new, C_set_new, feature_x, feature_y, num_camera):
    
    pts_3d = X_3d[X_idx]
    pts_2d = get_2D_pts(X_idx, V_mat, feature_x, feature_y)
    
    camera_params = []
    f = K[1, 1] # focal length
    for i in range(num_camera+1):
        Ri, Ci = R_set_new[i], C_set_new[i]
        Qi = Rotation.from_matrix(Ri).as_rotvec()
        params = [Qi[0], Qi[1], Qi[2], Ci[0], Ci[1], Ci[2], f, 0, 0]
        camera_params.append(params)
    camera_params = np.array(camera_params).reshape(-1, 9)

    x0 = np.hstack((camera_params.ravel(), pts_3d.ravel()))
    camera_indices, point_indices = get_camera_pts_indices(V_mat)

    A = bundle_adjustment_sparsity(X_idx, V_mat, num_camera)
    
    res = opt.least_squares(reprojection_error, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
                            args=(num_camera, pts_3d.shape[0], camera_indices, point_indices, pts_2d, K))
    x1 = res.x
    num_camera = num_camera + 1
    opt_camera_params = x1[:num_camera * 9].reshape(num_camera, 9)
    opt_pts_3d = x1[num_camera * 9:].reshape((pts_3d.shape[0], 3))

    opt_X_all = np.zeros_like(X_3d)
    opt_X_all[X_idx] = opt_pts_3d

    opt_R_set, opt_C_set = [], []
    for i in range(len(opt_camera_params)):
        R = Rotation.from_rotvec(opt_camera_params[i, :3]).as_matrix()
        C = opt_camera_params[i, 3:6].reshape(3,1)
        opt_R_set.append(R)
        opt_C_set.append(C)
    
    return opt_X_all, opt_R_set, opt_C_set
