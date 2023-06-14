import numpy as np


with open('../data/3D.txt', 'r') as f:
    points_3d = f.readlines()

with open('../data/2D.txt', 'r') as f:
    points_2d = f.readlines()
# Step 1: Gather calibration data

# Step 2: Extract calibration target corners

# Step 3: Generate world points

# Step 4: Build the data matrices
image_points = [point.split() for point in points_2d[1:-1]] # 2D image points
world_points = [point.split() for point in points_3d[1:-1]]  # 3D world points

# Populate image_points and world_points with corresponding points

# Step 5: Perform SVD decomposition
A = np.zeros((len(image_points) * 2, 12))
for i, (image, object) in enumerate(zip(image_points, world_points)):
    X, Y, Z = [float(point) for point in object]
    u, v = [float(point) for point in image]
    A[2 * i] = [-X, -Y, -Z, -1, 0, 0, 0, 0, u * X, u * Y, u * Z, u]
    A[2 * i + 1] = [0, 0, 0, 0, -X, -Y, -Z, -1, v * X, v * Y, v * Z, v]

_, _, V = np.linalg.svd(A)
P = V[-1].reshape((3, 4))

# Step 6: Extract camera parameters
# Extract camera parameters from V
camera_params = V[-1, :12]
camera_params /= camera_params[-1]  # Normalize the parameters

# Compute the calibration error
projection_error = 0.0
for i, (image, object) in enumerate(zip(image_points, world_points)):
    X = [float(point) for point in object]
    X.append(1)
    x = [float(point) for point in image]
    x.append(1)

    projected_x = np.dot(camera_params.reshape((3, 4)), X)
    projected_x /= projected_x[-1]  # Normalize

    projection_error += np.linalg.norm(projected_x[:-1] - x[:-1])

mean_projection_error = projection_error / len(image_points)
