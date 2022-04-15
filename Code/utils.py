import cv2
import numpy as np

def extract_feature(data_path, num_imgs):
    feature_x = []
    feature_y = []
    feature_idx = []
    # feature_discriptor = []

    for n in range(1, num_imgs):
        matching_txt = open(data_path + "matching" + str(n) + ".txt", "r")
        for i, line in enumerate(matching_txt):
            # the first line is number of feature
            if i == 0:
                continue
            x_row = np.zeros((1, num_imgs))
            y_row = np.zeros((1, num_imgs))
            idx_row = np.zeros((1, num_imgs), dtype=int)

            line = line.split()
            features = [np.float32(x) for x in line]
            features = np.array(features)
            num_matches = features[0]
            # r, g, b = features[1], features[2], features[3]
            src_x, src_y = features[4], features[5]

            # feature_discriptor.append([r, g, b])
            x_row[0, n-1] = src_x
            y_row[0, n-1] = src_y
            idx_row[0, n-1] = 1

            m = 1
            while num_matches > 1:
                img_id = int(features[m+5]) 
                img_id_x = features[m+6]
                img_id_y = features[m+7]
                x_row[0, img_id-1] = img_id_x
                y_row[0, img_id-1] = img_id_y
                idx_row[0, img_id-1] = 1
                m += 3
                num_matches -= 1

            feature_x.append(x_row)
            feature_y.append(y_row)
            feature_idx.append(idx_row)

    return np.array(feature_idx).reshape(-1, num_imgs) ,np.array(feature_x).reshape(-1, num_imgs), np.array(feature_y).reshape(-1, num_imgs)