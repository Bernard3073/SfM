import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from GetInliersRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from PnPRANSAC import *
from NonlinearPnP import *
from BundleAdjustment import *
from BuildVisibilityMatrix import *

# camera intrinsic matrix
K = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038], [0, 0, 1]])
# number of image files
num_imgs = 6

def readImageSet(folder_name, n_images):
    print("Reading images from ", folder_name)
    images = []
    for n in range(1, n_images+1):
        image_path = folder_name + "/" + str(n) + ".jpg"
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def makeImageSizeSame(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)

    return images_resized

def showMatches(image_1, image_2, pts1, pts2, color, file_name):

    image_1, image_2 = makeImageSizeSame([image_1, image_2])
    concat = np.concatenate((image_1, image_2), axis = 1)

    if pts1 is not None:
        corners_1_x = pts1[:,0].copy().astype(int)
        corners_1_y = pts1[:,1].copy().astype(int)
        corners_2_x = pts2[:,0].copy().astype(int)
        corners_2_y = pts2[:,1].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]), (corners_2_x[i] ,corners_2_y[i]), color, 1)
    cv2.imshow(file_name, concat)
    cv2.waitKey() 
    # if file_name is not None:    
    #     cv2.imwrite(file_name, concat)
    cv2.destroyAllWindows()
    return concat
def get_inliers(feature_idx, feature_x, feature_y):
    new_feature_idx = np.zeros_like(feature_idx)
    all_F_mat = np.empty(shape=(num_imgs, num_imgs), dtype=object)
    for i in range(num_imgs-1):
        for j in range(i+1, num_imgs):
            idx = np.where(feature_idx[:, i] & feature_idx[:, j])
            pts_1 = np.hstack((feature_x[idx, i].T, feature_y[idx, i].T))
            pts_2 = np.hstack((feature_x[idx, j].T, feature_y[idx, j].T))
            # showMatches(imgs[i], imgs[j], pts_1, pts_2, (0,255,0), None)
            idx = np.array(idx).T
            if len(idx) > 8:
                F_mat_best, new_idx = GetInliersRANSAC(pts_1, pts_2, idx)
                print('number of inliers: ', len(new_idx), '/', len(idx), 'between images:', i,j)  
                all_F_mat[i, j] = F_mat_best
                new_feature_idx[new_idx, i] = 1
                new_feature_idx[new_idx, j] = 1

    return new_feature_idx, all_F_mat

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_path', default="./Data/", help='directory to the image data')
    Args = Parser.parse_args()
    data_path = Args.data_path
    feature_idx, feature_x, feature_y = extract_feature(data_path, num_imgs)

    # imgs = readImageSet(data_path, num_imgs)
    feature_idx, all_F_mat = get_inliers(feature_idx, feature_x, feature_y)
    
    # select the first two images for the initial pair reconstruction
    F_mat_12 = all_F_mat[0, 1]
    E_mat_12 = EssentialMatrixFromFundamentalMatrix(F_mat_12, K)
    # four camera pose configurations
    R_set, C_set = ExtractCameraPose(E_mat_12)
    idx = np.where(feature_idx[:, 0] & feature_idx[:, 1])
    pts_1 = np.hstack((feature_x[idx, 0].T, feature_y[idx, 0].T))
    pts_2 = np.hstack((feature_x[idx, 1].T, feature_y[idx, 1].T))

    R1 = np.identity(3)
    C1 = np.zeros((3, 1))
    X_set = []
    for i in range(len(C_set)):
        X = LinearTriangulation(K, R1, C1, R_set[i], C_set[i], pts_1, pts_2)
        X = X / X[:, 3].reshape((-1, 1))
        X_set.append(X)
    # the correct camera pose and its 3D triangulated points
    R_new, C_new, X_new = DisambiguateCameraPose(R_set, C_set, X_set)
    X_new = X_new / X_new[:, 3].reshape((-1, 1))
    X_new_opt = NonlinearTriangulation(K, R1, C1, R_new, C_new, X_new, pts_1, pts_2)
    X_new_opt = X_new_opt/X_new_opt[:, 3].reshape((-1, 1))
    # print("X_new: ", X_new)
    # print("X_new_opt: ", X_new_opt)

    X_3d = np.zeros((feature_x.shape[0], 3)) # shape: N x 3
    camera_indices = np.zeros((feature_x.shape[0], 1), dtype = int) 
    # binary array for checking the feature points
    X_binary = np.zeros((feature_x.shape[0], 1), dtype = int) # shape: N x 1

    X_3d[idx] = X_new[:, :3]
    X_binary[idx] = 1
    camera_indices[idx] = 1

    X_binary[np.where(X_3d[:, 2] <0) ] = 0
    C_set_new = []
    R_set_new = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set_new.append(C0)
    C_set_new.append(C_new)
    R_set_new.append(R0)
    R_set_new.append(R_new)
    # Register camera and add 3D points for the rest of the images
    for i in range(2, num_imgs):
        feature_idx_i = np.where(X_binary[:, 0] & feature_idx[:, i])
        if len(feature_idx_i[0]) < 8:
            continue

        x_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), 
                            feature_y[feature_idx_i, i].reshape(-1,1)))
        X = X_3d[feature_idx_i, :].reshape(-1,3)
        R_i, C_i = PnPRANSAC(X, x_i, K)
        R_new, C_new = NonlinearPnP(X, x_i, K, R_i, C_i)

        R_set_new.append(R_new)
        C_set_new.append(C_new)

        # triangulate the 3D points
        for j in range(i):
            X_j_idx = np.where(feature_idx[:, i] & feature_idx[:, j])
            if len(X_j_idx[0]) < 8:
                continue
            x1 = np.hstack((feature_x[X_j_idx, j].reshape(-1,1), feature_y[X_j_idx, j].reshape(-1,1)))
            x2 = np.hstack((feature_x[X_j_idx, i].reshape(-1,1), feature_y[X_j_idx, i].reshape(-1,1)))
            
            X_new = LinearTriangulation(K, R_set_new[j], C_set_new[j], R_new, C_new,  x1, x2)
            X_new = X_new / X_new[:, 3].reshape((-1, 1))
            X_new = NonlinearTriangulation(K, R_set_new[j], C_set_new[j], R_new, C_new, X_new, x1, x2)
            X_new = X_new / X_new[:, 3].reshape((-1, 1))

            X_3d[X_j_idx] = X_new[:, :3]
            X_binary[X_j_idx] = 1
            print("appended ", len(X_j_idx[0]), " points between ", j ," and ", i)
            X_idx, V_mat = BuildVisibilityMatrix(X_binary, feature_idx, i)

            X_3d, R_set_new, C_set_new = BundleAdjustment(X_3d, X_idx, V_mat, K, R_set_new, C_set_new, feature_x, feature_y, i)

            for k in range(i+1):
                X_pts_idx = np.where(X_binary[:, 0] & feature_idx[:, k])
                x = np.hstack((feature_x[X_pts_idx, k].reshape(-1,1), feature_y[X_pts_idx, k].reshape(-1,1)))
                X = X_3d[X_pts_idx]
                BundleAdjustment_error = reprojectionErrorPnP(X, x, K, R_set_new[k], C_set_new[k])
                print("BundleAdjustment_error: ", BundleAdjustment_error)

    X_binary[X_3d[:,2] < 0] = 0
    feature_idx = np.where(X_binary[:, 0])
    X = X_3d[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    
    # 2D plotting
    plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
    for i in range(0, len(C_set_new)):
        R1 = Rotation.from_matrix(R_set_new[i]).as_rotvec()
        R1 = np.rad2deg(R1)
        plt.plot(C_set_new[i][0],C_set_new[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
    
    plt.savefig('2D_plot.png')
    plt.show()
    # 3D plotting
    ax = plt.axes(projection ="3d")
    # Creating plot
    ax.scatter3D(x, y, z, color = "blue")
    plt.savefig('3D_plot.png')
    plt.show()

if __name__ == '__main__':
    main()