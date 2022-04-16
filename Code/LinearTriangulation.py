import numpy as np

def LinearTriangulation(K, R1, C1, R2, C2, x1, x2):
    """
    Triangulate the 3D points, given two camera poses
    Input:
        K: Camera matrix
        C1: Camera position in image 1
        R1: Camera orientation in image 1
        C2: Camera position in image 2
        R2: Camera orientation in image 2
        x1: Feature position in image 1
        x2: Feature position in image 2
    Output:
        X: 3D point
    """
    I = np.identity(3)
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))

    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    p11 = P1[0,:]
    p12 = P1[1,:]
    p13 = P1[2,:]

    p21 = P2[0,:]
    p22 = P2[1,:]
    p23 = P2[2,:]

    all_X = []
    for i in range(x1.shape[0]):
        x = x1[i,0]
        y = x1[i,1]
        x_prime = x2[i,0]
        y_prime = x2[i,1]
        A = []
        # Direct Linear Transformation
        # x (cross) PX = 0 and concatenate the 2D points from both images
        # Then, solve SVD
        A.append((y * p13.T) -  p12.T)
        A.append(p11.T -  (x * p13.T))
        A.append((y_prime * p23.T) -  p22.T)
        A.append(p21.T -  (x_prime * p23.T))

        A = np.array(A).reshape(4,4)

        _, _, Vt = np.linalg.svd(A)
        v = Vt.T
        x = v[:,-1]
        all_X.append(x)

    return np.array(all_X)