import cv2
import numpy as np

# Function to calculate the fundamental matrix using OpenCV
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
        if match.distance < 100:
            good_matches.append(match)

    # Extract corresponding keypoints
    src_pts = np.float32([kp1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

    # Calculate fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return F

# Function to draw epipolar lines and corresponding points
def draw_epipolar_lines(img1, img2, F):
    # Display images
    cv2.namedWindow('Image 1')
    cv2.imshow('Image 1', img1)
    cv2.namedWindow('Image 2')
    cv2.imshow('Image 2', img2)

    while True:
        # Wait for user to click on a pixel in Image 1
        x, y, = 150, 140
        p = np.array([[x, y, 1]], dtype=np.float32).T

        # Calculate corresponding epipolar line in Image 2
        line = F @ p
        line /= np.sqrt(line[0] ** 2 + line[1] ** 2)

        # Calculate epipolar line parameters
        a, b, c = line.flatten()

        # Calculate the range of x coordinates for drawing the line
        img1_height, img1_width = img1.shape[:2]
        x1 = 0
        y1 = int(-c / b)
        x2 = img1_width - 1
        y2 = int(-(a * x2 + c) / b)

        # Draw the epipolar line on Image 1
        cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(img1, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Display updated Image 1
        cv2.imshow('Image 1', img1)

        # Calculate the corresponding point in Image 2
        range_min = max(0, y1 - 10)
        range_max = min(img1_height - 1, y1 + 10)

        best_score = -1
        best_x = -1
        best_y = -1

        for i in range(range_min, range_max):
            x = int(-(b * i + c) / a)

            if x >= 0 and x < img1_width:
                score = calculate_zncc_score(img1, img2, (int(x), i), (int(x), i))
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_y = i

        # Draw the corresponding point on Image 2
        cv2.circle(img2, (best_x, best_y), 3, (0, 255, 0), -1)
        cv2.imshow('Image 2', img2)

        # Wait for ESC key to exit
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()

# Function to calculate ZNCC score
def calculate_zncc_score(img1, img2, point1, point2):
    window_size = 5
    x1, y1 = point1
    x2, y2 = point2

    patch1 = img1[y1 - window_size:y1 + window_size + 1, x1 - window_size:x1 + window_size + 1]
    patch2 = img2[y2 - window_size:y2 + window_size + 1, x2 - window_size:x2 + window_size + 1]

    patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
    patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)

    zncc = np.mean(patch1 * patch2)

    return zncc

# Read the images
img1 = cv2.imread('../data/left.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('../data/right.png', cv2.IMREAD_COLOR)

# Calculate the fundamental matrix
F = calculate_fundamental_matrix(img1, img2)

# Draw epipolar lines and corresponding points
draw_epipolar_lines(img1.copy(), img2.copy(), F)
