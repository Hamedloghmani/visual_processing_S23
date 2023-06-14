import numpy as np
import cv2 as cv
import glob

# Just for reference, this calibration is inspired by a tutorial in OpenCV documentations and the link is provided as well:
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('../data/chessboard.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()
mean_error = 0

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Printing out the camera matrix
print(f'M matrix: \n {mtx}')

for i in range(len(objpoints)):
    # Project the object points into image space
    img_points_reprojected, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # Compute the difference between the projected points and the detected points
    error = cv.norm(imgpoints[i], img_points_reprojected, cv.NORM_L2) / len(img_points_reprojected)
    mean_error += error

mean_error /= len(objpoints)
# Printing mean error
print("Mean reprojection error: {}".format(mean_error))