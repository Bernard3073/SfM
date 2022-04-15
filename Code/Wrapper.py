import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *
from GetInliersRANSAC import *
import EssentialMatrixFromFundamentalMatrix
import ExtractCameraPose

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
#     file_name =  None
    # image_1 = img_1
    # image_2 = img_2

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

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_path', default="./Data/", help='directory to the image data')
    Args = Parser.parse_args()
    data_path = Args.data_path
    feature_idx, feature_x, feature_y = extract_feature(data_path, num_imgs)
    
    new_feature_idx = np.zeros_like(feature_idx)
    all_F_mat = np.empty(shape=(num_imgs, num_imgs), dtype=object)

    imgs = readImageSet(data_path, num_imgs)

    for i in range(num_imgs-1):
        for j in range(i+1, num_imgs):
            idx = np.where(feature_idx[:, i] & feature_idx[:, j])
            pts_1 = np.hstack((feature_x[idx, i].T, feature_y[idx, i].T))
            pts_2 = np.hstack((feature_x[idx, j].T, feature_y[idx, j].T))
            # showMatches(imgs[i], imgs[j], pts_1, pts_2, (0,255,0), None)
            idx = np.array(idx).T
            if len(idx) > 8:
                F_mat_best, new_idx = GetInliersRANSAC(pts_1, pts_2, idx)
                print('At image : ',  i,j, '|| Number of inliers: ', len(new_idx), '/', len(idx) )  
                all_F_mat[i, j] = F_mat_best
                feature_idx[new_idx, i] = 1
                feature_idx[new_idx, j] = 1

if __name__ == '__main__':
    main()