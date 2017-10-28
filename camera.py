import numpy as np
import cv2
import glob
import json
import yaml


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg') # make .png if required

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    # print corners
    # with open("testtext.txt", "wb") as myfile:
    # np.write(corners'\n')


    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(5000)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# It's very important to transform the matrix to list.
data = {'ret_matrix': np.asarray(ret).tolist(),'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist(),'rot_mat': np.asarray(rvecs).tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)

    data2 = {'ret_matrix': np.asarray(ret).tolist(),'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist(),'rot_mat': np.asarray(rvecs).tolist(), 't_vector': np.asarray(tvecs).tolist()}
with open("calibration2.yaml", "w") as f:
    yaml.dump(data2, f)