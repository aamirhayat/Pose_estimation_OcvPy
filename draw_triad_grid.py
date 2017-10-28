# Project: Pose estimation

import scipy.io
import numpy as np
import cv2
import glob
import json
import yaml
import time


# mtx = np.zeros((3,3),dtype=np.float64 )
# mtx[0][0] = 536.0734367759593
# mtx[0][2] = 342.3703824413255
# mtx[1][1] = 536.0163520780469
# mtx[1][2] = 235.5368541488102
# mtx[2][2] = 1.0
#
#
# dist = np.array([-0.26509011033645813, -0.046743552163847554, .0018330093181183355, -0.0003147148202299227,   0.25231509402326396])

with open('calibration2.yaml') as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix')
mtx = np.array(mtx)
dist = loadeddict.get('dist_coeff')
dist = np.array(dist)


#takes the corners in the chessboard
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.arrowedLine(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 2)
    cv2.arrowedLine(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 2)
    cv2.arrowedLine(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 2)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Arrays to store object points and image points from all the images.
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)



# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.png') # make .png if required
for fname in glob.glob('*.png'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # Write on image
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img,'The triad on Grid depicts the POSE estimated by camera',(10,30), font, 2,(155,155,255),1,cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img,'Experiment by: A. A. Hayat',(10,60), font, 0.5,(0,0,255),1,cv2.LINE_AA)


        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)

cv2.destroyAllWindows()
