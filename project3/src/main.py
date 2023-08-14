#Project3
#Lavanya Nagaraju- 110122643
#Hamed Loghmani- 110107453

import math
import cv2
import numpy as np


with open('im1_points.txt', 'r') as f:
    points1 = f.readlines()

with open('im2_points.txt', 'r') as f:
    points2 = f.readlines()

with open('../data/3DCoordinates.txt', 'r') as f:
    points_3d = f.readlines()

points1 = [point.split() for point in points1]  # 2D image points
points2 = [point.split() for point in points2]  # 2D image points
points_3d = [point.split() for point in points_3d]  # 3D world points
im1 = cv2.imread('../data/im1.jpeg', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../data/im2.jpeg', cv2.IMREAD_GRAYSCALE)


def camera_calibration(image_points, world_points):
    A = np.zeros((len(image_points) * 2, 12))
    for i, (image, object) in enumerate(zip(image_points, world_points)):
        X, Y, Z = [float(point) for point in object]
        x, y = [float(point) for point in image]
        A[2 * i, :] = [-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x]
        A[2 * i + 1, :] = [0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y]
    # Perform SVD decomposition
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape((3, 4))

    # Extract camera parameters from V
    camera_params = V[-1, :12]
    camera_params /= camera_params[-1]  # Normalize the parameters

    print(f'Camera parameters: \n {camera_params}')
    # Compute the calibration error
    projection_error = 0.0
    for i, (image, object) in enumerate(zip(image_points, world_points)):
        X = [float(point) for point in object]
        X.append(1)
        x = [float(point) for point in image]
        x.append(1)

        projected_x = np.dot(camera_params.reshape((3, 4)), X)
        projected_x /= projected_x[-1]  # Normalizing
        projection_error += np.linalg.norm(projected_x[:-1] - x[:-1])

    mean_error = projection_error / len(image_points)
    print("Mean reprojection error: {}".format(mean_error))

    return camera_params


def calculate_fundamental_matrix(img1, img2):
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to keep good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract corresponding keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute fundamental matrix using RANSAC
    fundamental_matrix, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)

    print(fundamental_matrix)
    return fundamental_matrix


def calculate_zncc(img1, img2, point1, point2, window_size):
    """
    Calculate Zero Mean Normalized Cross-Correlation (ZNCC) between two pixel points.

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.
        point1 (tuple): Coordinates (row, col) of the first point in img1.
        point2 (tuple): Coordinates (row, col) of the corresponding point in img2.
        window_size (int): Size of the window around the points for ZNCC calculation.

    Returns:
        zncc_score (float): ZNCC score between the patches centered at the two points.
    """
    half_window = window_size // 2

    # Extract patches around the points
    patch1 = img1[point1[0] - half_window:point1[0] + half_window + 1,
             point1[1] - half_window:point1[1] + half_window + 1]

    patch2 = img2[point2[0] - half_window:point2[0] + half_window + 1,
             point2[1] - half_window:point2[1] + half_window + 1]

    # Calculate means of the patches
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)

    # Calculate zero-mean patches
    zero_mean_patch1 = patch1 - mean1
    zero_mean_patch2 = patch2 - mean2

    # Calculate the cross-correlation
    cross_corr = np.sum(zero_mean_patch1 * zero_mean_patch2)

    # Calculate the standard deviations
    std_dev1 = np.sqrt(np.sum(zero_mean_patch1 ** 2))
    std_dev2 = np.sqrt(np.sum(zero_mean_patch2 ** 2))

    # Calculate ZNCC
    zncc_score = cross_corr / (std_dev1 * std_dev2)

    return zncc_score


def drawEpipolarLine(u, v, F, img1, img2):

    p = np.array([[u, v, 1]], dtype=np.float32)

    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints1, desc1 = sift.detectAndCompute(img1, None)
    keypoints2, desc2 = sift.detectAndCompute(img2, None)

    # Create a Brute-Force Matcher
    bf = cv2.BFMatcher()
    # Match descriptors from both images
    matches = bf.knnMatch(desc1, desc2, k=2)
    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in good_matches])
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in good_matches])

    distance_from_keypoints = list()
    for i, pt in enumerate(src_pts):
        distance_from_keypoints.append((math.sqrt((pt[0] - u)**2 + (pt[1] - v)**2), pt, dst_pts[i]))
    distance_from_keypoints.sort(key=lambda x: x[0])
    distance_from_keypoints = distance_from_keypoints[0]

    disparity = distance_from_keypoints[1][0] - distance_from_keypoints[2][0]

    line = cv2.computeCorrespondEpilines(p, 1, F)
    epipolar_line = line.reshape(-1, 3)
    # Calculate epipolar line parameters
    epipolar_line = epipolar_line.flatten()
    a, b, c = epipolar_line

    # Calculate the range of x coordinates for drawing the line
    img1_height, img1_width = img1.shape[:2]
    x1 = 0
    y1 = int(-c / b)
    x2 = img1_width - 1
    y2 = int(-(a * x2 + c) / b)

    img_with_line = cv2.line(img2, [x1, y1], [x2, y2], (0, 255, 0))

    points_right = list()
    ranges = [int(u-disparity), int(u+disparity)]
    for x in range(min(ranges), max(ranges)):
        y = int(-(epipolar_line[0] * x + epipolar_line[2]) / epipolar_line[1])

        try:
            zncc_score = calculate_zncc(img1, img2, (u, v), (x, y), window_size=7)
            points_right.append(((x, y), zncc_score))
        except ValueError:
            continue

    points_right.sort(key=lambda x: x[1], reverse=True)
    best_point_right = points_right[0]

    cv2.circle(img_with_line, (best_point_right[0][0], best_point_right[0][1]), 5, (0, 0, 255), -1)
    cv2.imshow('Image', img_with_line)
    cv2.waitKey(0)
    return best_point_right[0]


def three_dimensional_reconstruction(camera_matrix1, camera_matrix2, p1, p2):
    A = np.zeros((4, 3))

    A[0][0] = p1[0] * camera_matrix1[8] - camera_matrix1[0]
    A[0][1] = p1[0] * camera_matrix1[9] - camera_matrix1[1]
    A[0][2] = p1[0] * camera_matrix1[10] - camera_matrix1[2]

    A[1][0] = p1[1] * camera_matrix1[8] - camera_matrix1[4]
    A[1][1] = p1[1] * camera_matrix1[9] - camera_matrix1[5]
    A[1][2] = p1[1] * camera_matrix1[10] - camera_matrix1[6]

    A[2][0] = p2[0] * camera_matrix2[8] - camera_matrix2[1]
    A[2][1] = p2[0] * camera_matrix2[9] - camera_matrix2[2]
    A[2][2] = p2[0] * camera_matrix2[10] - camera_matrix2[3]

    A[3][0] = p2[0] * camera_matrix2[8] - camera_matrix2[4]
    A[3][1] = p2[0] * camera_matrix2[9] - camera_matrix2[5]
    A[3][2] = p2[0] * camera_matrix2[10] - camera_matrix2[6]

    d = np.zeros((1, 4))
    d[0][0] = p1[0] * camera_matrix1[11] - camera_matrix1[3]
    d[0][1] = p1[1] * camera_matrix1[11] - camera_matrix1[7]
    d[0][2] = p2[0] * camera_matrix2[11] - camera_matrix2[3]
    d[0][3] = p2[1] * camera_matrix2[11] - camera_matrix2[7]

    U, S, Vt = np.linalg.svd(A)
    # Calculate the pseudoinverse of S
    S_pseudo = np.zeros(A.shape).T
    S_pseudo[:S.shape[0], :S.shape[0]] = np.diag(1 / S)
    # Calculate the pseudoinverse of A using SVD
    A_pseudo = np.dot(np.dot(Vt.T, S_pseudo), U.T)

    # Solve for matrix X
    X = np.dot(A_pseudo, d.T)

    return X

M1 = camera_calibration(points1, points_3d)
M2 = camera_calibration(points2, points_3d)
F = calculate_fundamental_matrix(im1, im2)

def click_and_mark(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        #points = (x, y)
        cv2.circle(im1, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image1', im1)# Mark the clicked point with a red circle

        p_ = drawEpipolarLine(x, y, F, im1, im2)
        print('3D Reconstruction: {}'.format(three_dimensional_reconstruction(M1, M2, (x, y), p_)))


cv2.imshow('Image1', im1)
cv2.setMouseCallback('Image1', click_and_mark)

while True:
    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit and save the points
    if key == ord('q'):
        break

