import cv2
import matplotlib.pyplot as plt

# Global variable to store clicked points
points = []
# Mouse callback function
def click_and_mark(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Mark the clicked point with a red circle
        cv2.imshow('Image', img)

# Load the image
img_path = '../data/im1.jpeg'
img = cv2.imread(img_path)

# Create a window and set the mouse callback
cv2.imshow('Image1', img)
cv2.setMouseCallback('Image1', click_and_mark)

while True:
    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit and save the points
    if key == ord('q'):
        break

# Save the points to a text file
output_file = 'im1_points.txt'
with open(output_file, 'w') as file:
    for point in points:
        file.write(f'{point[0]} {point[1]}\n')
cv2.destroyAllWindows()
points = list()

img_path = '../data/im2.jpeg'
img = cv2.imread(img_path)
cv2.imshow('Image2', img)
cv2.setMouseCallback('Image2', click_and_mark)

while True:
    key = cv2.waitKey(1) & 0xFF
    # Press 'q' to quit and save the points
    if key == ord('q'):
        break
cv2.destroyAllWindows()

# Save the points to a text file
output_file = 'im2_points.txt'
with open(output_file, 'w') as file:
    for point in points:
        file.write(f'{point[0]} {point[1]}\n')

# Plot the points using Matplotlib
# plt.figure()
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# for point in points:
#     plt.scatter(point[0], point[1], color='red', marker='+', s=100)
# plt.title('Clicked Points with + Markers')
# plt.axis('off')
# plt.show()
