import math

import numpy as np
import cv2
#from zncc import zncc as compute_zncc

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
        u, v = [float(point) for point in image]
        A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, u * X, u * Y, u * Z, u]
        A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v]

    # Perform SVD decomposition
    _, _, V = np.linalg.svd(A)
    P = V[-1].reshape((3, 4))

    # Extract camera parameters from V
    camera_params = V[-1, :12]
    #camera_params /= camera_params[-1]  # Normalize the parameters

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
    # Detect keypoints and extract descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match keypoints using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Select good matches
    good_matches = []
    for match in matches:
        if match.distance < 50:
            good_matches.append(match)

    # Extract corresponding keypoints
    src_pts = np.float32([kp1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Calculate fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(F)
    return F



def compute_zncc(patch1, patch2):
    mean_patch1 = np.mean(patch1)
    mean_patch2 = np.mean(patch2)
    patch1_centered = patch1 - mean_patch1
    patch2_centered = patch2 - mean_patch2
    numerator = np.sum(patch1_centered * patch2_centered)
    denominator = np.sqrt(np.sum(patch1_centered ** 2) * np.sum(patch2_centered ** 2))
    zncc_score = numerator / denominator
    return zncc_score

# def compute_zncc(img1, img2, point1, point2):
#     window_size = 3
#     x1, y1 = point1
#     x2, y2 = point2
#
#     patch1 = img1[y1 - window_size:y1 + window_size + 1, x1 - window_size:x1 + window_size + 1]
#     patch2 = img2[y2 - window_size:y2 + window_size + 1, x2 - window_size:x2 + window_size + 1]
#
#     patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
#     patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)
#
#     zncc = np.mean(patch1 * patch2)
#
#     return zncc
# def compute_zncc(image1, image2, point1, point2, window_size=10):
#     half_window = window_size // 2
#
#     # Extract image patches centered around the provided points
#     window1 = image1[point1[0] - half_window: point1[0] + half_window + 1,
#               point1[1] - half_window: point1[1] + half_window + 1]
#     window2 = image2[point2[0] - half_window: point2[0] + half_window + 1,
#               point2[1] - half_window: point2[1] + half_window + 1]
#
#     if window2 is None or window1 is None:
#         raise Exception
#     # Calculate the means and standard deviations
#     mean_window1 = np.mean(window1)
#     mean_window2 = np.mean(window2)
#     std_window1 = np.std(window1)
#     std_window2 = np.std(window2)
#
#     # Calculate the cross-correlation term
#     cross_corr = np.sum((window1 - mean_window1) * (window2 - mean_window2))
#
#     # Calculate the ZNCC score
#     zncc_score = cross_corr / (std_window1 * std_window2)
#
#     return zncc_score
# def compute_zncc(image1, image2, point1, point2, patch_size=11):
#     """
#     Compute the ZNCC score between two points in two images using specified patch size.
#
#     Args:
#         image1 (numpy.ndarray): First image.
#         image2 (numpy.ndarray): Second image.
#         point1 (tuple): (x, y) coordinates of the point in the first image.
#         point2 (tuple): (x, y) coordinates of the corresponding point in the second image.
#         patch_size (int): Size of the image patches for computing ZNCC.
#
#     Returns:
#         float: ZNCC score between the patches centered around the given points.
#     """
#     x1, y1 = point1
#     x2, y2 = point2
#
#     # Extract patches around the points
#     patch1 = image1[y1 - patch_size // 2:y1 + patch_size // 2 + 1, x1 - patch_size // 2:x1 + patch_size // 2 + 1]
#     patch2 = image2[y2 - patch_size // 2:y2 + patch_size // 2 + 1, x2 - patch_size // 2:x2 + patch_size // 2 + 1]
#
#     # Calculate mean and standard deviation of the patches
#     mean_patch1 = patch1.mean()
#     mean_patch2 = patch2.mean()
#     std_patch1 = patch1.std()
#     std_patch2 = patch2.std()
#
#     # Calculate the ZNCC score
#     zncc_score = ((patch1 - mean_patch1) * (patch2 - mean_patch2)).sum() / (
#                 std_patch1 * std_patch2 * patch_size * patch_size)
#
#     return zncc_score
# Draw the epipolar line for a given point (u, v)
# def drawEpipolarLine(u, v, F):
#     #x1, y1, x2, y2 = computeEpipolarPoints(u, v)
#     p = np.array([[u, v, 1]], dtype=np.float32).T
#
#     # Calculate corresponding epipolar line in Image 2
#     line = F @ p
#     line /= np.sqrt(line[0] ** 2 + line[1] ** 2)
#
#     # Calculate epipolar line parameters
#     a, b, c = line.flatten()
#
#     # Calculate the range of x coordinates for drawing the line
#     img1_height, img1_width = im1.shape[:2]
#     x1 = 0
#     y1 = int(-c / b)
#     x2 = img1_width - 1
#     y2 = int(-(a * x2 + c) / b)
#
#     #canvas2.create_line(x1, y1, x2, y2, fill='green', width=5)
#     img_with_line = cv2.line(im2, (x1,y1), (x2, y2), (0, 255, 0))
#
#     # Show the image with the line
#     # cv2.imshow('Image with Line', img_with_line)
#     # cv2.waitKey(0)
#     # Calculate the corresponding point in Image 2
#     range_min = max(0, y1 - 10)
#     range_max = min(img1_height - 1, y1 + 10)
#
#     best_score = -1
#     best_x = -1
#     best_y = -1
#
#     for i in range(range_min, range_max):
#         x = int(-(b * i + c) / a)
#
#         if x >= 0 and x < img1_width:
#             score = compute_zncc(im1, im2, (int(x), i), (int(x), i))
#             if score > best_score:
#                 best_score = score
#                 best_x = x
#                 best_y = i
#
#     cv2.circle(img_with_line, (best_x, best_y), 5, (0, 0, 255), -1)  # Mark the clicked point with a red circle
#     cv2.imshow('Image', img_with_line)
#     return best_x, best_y


def drawEpipolarLine(u, v, F, img1, img2):
    p = np.array([[u, v, 1]], dtype=np.float32)

    sift = cv2.SIFT_create()

    # Detect keypoints
    keypoints1, desc1 = sift.detectAndCompute(img1, None)
    keypoints2, desc2 = sift.detectAndCompute(img2, None)

    k_d = list(zip(keypoints1, desc1))
    distance_from_keypoints = list()
    for i in k_d:
        distance_from_keypoints.append((math.sqrt((i[0].pt[0] - u)**2 + (i[0].pt[1] - v)**2), i[0], i[1]))
    distance_from_keypoints.sort(key=lambda x: x[0])
    distance_from_keypoints = distance_from_keypoints[:20]

    # Create a Brute-Force Matcher
    bf = cv2.BFMatcher()

    desc = np.array([x[2] for x in distance_from_keypoints])
    # Match descriptors from both images
    matches = bf.knnMatch(desc, desc2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)
    # cv2.imshow('Keypoint Matches', matched_image)
    # cv2.waitKey(0)

    #matched_keypoints1 = [keypoints1[match.queryIdx] for match in good_matches]
    matched_keypoints2 = [keypoints2[match.trainIdx] for match in good_matches]
    
    # Calculate corresponding epipolar line in Image 2
    # line = F @ p
    # line /= np.sqrt(line[0] ** 2 + line[1] ** 2)
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
    # Calculate the corresponding point in Image 2
    # range_min = max(0, y1 - 50)
    # range_max = min(img1_height - 1, y1 + 50)
    #
    # best_score = -1
    # best_x = -1
    # best_y = -1
    #
    # for i in range(range_min, range_max):
    #     x = int(-(b * i + c) / a)
    #
    #     if x >= 0 and x < img1_width:
    #
    #         score = compute_zncc(img1, img2, x, i, best_x, best_y, 1)
    #         if score > best_score:
    #             best_score = score
    #             best_x = x
    #             best_y = i
    #
    #
    patch_size = 3 # Size of the patches to compare
    #best_zncc = -1
    #best_point_right = None
    points_right = list()
    patch_left = img1[u - patch_size:u + patch_size, v - patch_size:v + patch_size]
    for x in range(int(epipolar_line[0]), int(epipolar_line[0]) + img2.shape[1]):
        y = int(-(epipolar_line[0] * x + epipolar_line[2]) / epipolar_line[1])

        if y < patch_size or y >= img2.shape[0] - patch_size or x < patch_size or x >= img2.shape[1] - patch_size:
            continue

        #patch_left = img1[u - patch_size:u + patch_size, v - patch_size:v + patch_size]

        patch_right = img2[x - patch_size:x + patch_size, y - patch_size:y + patch_size]

        try:
            zncc_score = compute_zncc(patch_left, patch_right)
            points_right.append(((x, y), zncc_score))
        except ValueError:
            continue

    points_right.sort(key=lambda x: x[1], reverse=True)
    points_right = points_right[:5]
        # if zncc_score > best_zncc:
        #     best_zncc = zncc_score
        #     best_point_right = np.array([x, y, 1])

    final_result = list()
    for p in points_right:
        for k in matched_keypoints2:
            final_result.append((p[0], math.sqrt((k.pt[0] - p[0][0])**2 + (k.pt[1] - p[0][1])**2)))

    final_result.sort(key= lambda x: x[1])

    if len(final_result) == 0:
        best_point_right = np.array([points_right[0][0][0], points_right[0][0][1], 1])
    else:
        best_point_right = np.array([final_result[0][0][0], final_result[0][0][1], 1])

    cv2.circle(img_with_line, (best_point_right[0], best_point_right[1]), 5, (0, 0, 255), -1)
    cv2.imshow('Image', img_with_line)
    cv2.waitKey(0)
    return best_point_right


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

    print("Matrix X:")
    print(X)
    pass

M1 = camera_calibration(points1, points_3d)
M2 = camera_calibration(points2, points_3d)
F = calculate_fundamental_matrix(im1, im2)
#F, _ = cv2.findFundamentalMat(im1, im2, cv2.FM_LMEDS)

def click_and_mark(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        #points = (x, y)
        cv2.circle(im1, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image1', im1)# Mark the clicked point with a red circle

        p_ = drawEpipolarLine(x, y, F, im1, im2)
        print(three_dimensional_reconstruction(M1, M2, (x, y), p_))


cv2.imshow('Image1', im1)
cv2.setMouseCallback('Image1', click_and_mark)

while True:
    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit and save the points
    if key == ord('q'):
        break
#drawEpipolarLine(113, 120, F)
