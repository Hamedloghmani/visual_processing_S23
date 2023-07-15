#Project2
#Lavanya Nagaraju- 110122643
#Hamed Loghmani- 110107453

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2

# Event handler for clicking on the first image canvas
def img1ClickHandler(event):
    if len(imgPoints1) < 10:
        x, y = event.x, event.y
        print("first:", x, y)
        imgPoints1.append([x, y])
        drawPlusSign(canvas1, x, y)  # Draw a plus sign
    print("img1:", imgPoints1)
    if len(imgPoints1) == 10 and len(imgPoints2) == 10:
        epiModeChk.state(['!disabled'])

    if epipolarMode.get():
        x, y = event.x, event.y
        drawPlusSign(canvas1, x, y)  # Draw a plus sign
        imgPoints2.append(list(drawEpipolarLine(x, y)))
        print("img2:", imgPoints2)
# Event handler for clicking on the second image canvas
def img2ClickHandler(event):
    if len(imgPoints2) < 10:
        x, y = event.x, event.y
        print("second:", x, y)
        imgPoints2.append([x, y])
        drawPlusSign(canvas2, x, y)  # Draw a plus sign
    print("img2:", imgPoints2)
    if len(imgPoints1) == 10 and len(imgPoints2) == 10:
        epiModeChk.state(['!disabled'])

# Function to draw a plus sign in the canvas at the specified coordinates (x, y)
def drawPlusSign(canvas, x, y):
    size = 10  # Size of the plus sign
    half_size = size // 2
    canvas.create_line(x - half_size, y, x + half_size, y, fill=dot_color, width=1)  # Horizontal line
    canvas.create_line(x, y - half_size, x, y + half_size, fill=dot_color, width=1)  # Vertical line

# Create the input matrix for computing the fundamental matrix
def createInputMatrix():
    for i in range(len(imgPoints1)):
        p = imgPoints1[i]
        q = imgPoints2[i]
        matrixA.append([q[0] * p[0], q[0] * p[1], q[0], q[1] * p[0], q[1] * p[1], q[1], p[0], p[1], 1])


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

# Compute the epipolar points for a given point (u, v)
#def computeEpipolarPoints(u, v):
    # f1 = matrixF[0]
    # f2 = matrixF[1]
    # f3 = matrixF[2]
    # u1 = -((0 * (v * f2[0] + v * f2[1] + f2[2])) + u * f3[0] + v * f3[1] + f3[2]) / (u * f1[0] + v * f1[1] + f1[2])
    # u2 = -((1080 * (v * f2[0] + v * f2[1] + f2[2])) + u * f3[0] + v * f3[1] + f3[2]) / (u * f1[0] + v * f1[1] + f1[2])
    # return int(u1), 0, int(u2), 1080
    # p = np.array([[u, v, 1]], dtype=np.float32).T
    #
    # # Calculate corresponding epipolar line in Image 2
    # line = matrixF @ p
    # line /= np.sqrt(line[0] ** 2 + line[1] ** 2)
    #
    # # Calculate epipolar line parameters
    # a, b, c = line.flatten()
    #
    # # Calculate the range of x coordinates for drawing the line
    # img1_height, img1_width = img1.shape[:2]
    # x1 = 0
    # y1 = int(-c / b)
    # x2 = img1_width - 1
    # y2 = int(-(a * x2 + c) / b)
    #
    # return x1, y1, x2, y2
# Compute the average value of a pixel neighborhood
def computeAvg(img, u, v, n):
    sum = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            sum += img[u + i][v + j]
    return sum / ((2 * n + 1) ** 2)

# Compute the standard deviation of a pixel neighborhood
def computeStdDeviation(img, u, v, n):
    sum = 0
    avg = computeAvg(img, u, v, n)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            sum += (img[u + i][v + j] - avg) ** 2
    return (sum ** 0.5) / (2 * n + 1)

def compute_zncc(img1, img2, point1, point2):
    window_size = 5
    x1, y1 = point1
    x2, y2 = point2

    patch1 = img1[y1 - window_size:y1 + window_size + 1, x1 - window_size:x1 + window_size + 1]
    patch2 = img2[y2 - window_size:y2 + window_size + 1, x2 - window_size:x2 + window_size + 1]

    patch1 = (patch1 - np.mean(patch1)) / np.std(patch1)
    patch2 = (patch2 - np.mean(patch2)) / np.std(patch2)

    zncc = np.mean(patch1 * patch2)

    return zncc

# Draw the epipolar line for a given point (u, v)
def drawEpipolarLine(u, v):
    #x1, y1, x2, y2 = computeEpipolarPoints(u, v)
    p = np.array([[u, v, 1]], dtype=np.float32).T

    # Calculate corresponding epipolar line in Image 2
    line = matrixF @ p
    line /= np.sqrt(line[0] ** 2 + line[1] ** 2)

    # Calculate epipolar line parameters
    a, b, c = line.flatten()

    # Calculate the range of x coordinates for drawing the line
    img1_height, img1_width = img1.shape[:2]
    x1 = 0
    y1 = int(-c / b)
    x2 = img1_width - 1
    y2 = int(-(a * x2 + c) / b)

    canvas2.create_line(x1, y1, x2, y2, fill='green', width=5)
    # Calculate the corresponding point in Image 2
    range_min = max(0, y1 - 10)
    range_max = min(img1_height - 1, y1 + 10)

    best_score = -1
    best_x = -1
    best_y = -1

    for i in range(range_min, range_max):
        x = int(-(b * i + c) / a)

        if x >= 0 and x < img1_width:
            score = compute_zncc(img1, img2, (int(x), i), (int(x), i))
            if score > best_score:
                best_score = score
                best_x = x
                best_y = i

    drawPlusSign(canvas2, best_x, best_y)
    return best_x, best_y
# Event handler for selecting the first image
def selectImg1():
    imgPoints1.clear()
    imgPoints2.clear()
    epipolarMode.set(False)
    global img1Url
    img1Url = filedialog.askopenfilename(initialdir=imageDir, title="Select 1st image",
                                         filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    image_address.append(img1Url)
    canvas1.img = ImageTk.PhotoImage(Image.open(img1Url))
    canvas1.create_image(0, 0, image=canvas1.img, anchor="nw")

# Event handler for selecting the second image
def selectImg2():
    imgPoints1.clear()
    imgPoints2.clear()
    epipolarMode.set(False)
    global img2Url
    img2Url = filedialog.askopenfilename(initialdir=imageDir, title="Select 2nd image",
                                         filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    image_address.append(img2Url)
    canvas2.img = ImageTk.PhotoImage(Image.open(img2Url))
    canvas2.create_image(0, 0, image=canvas2.img, anchor="nw")

# Get the list of points between two coordinates (x1, y1) and (x2, y2)
def getLinePoints(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if 0 < x < 1921:
            if issteep:
                points.append((y, x))
            else:
                points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
    if rev:
        points.reverse()
    return points

if __name__ == "__main__":
    image_address = list()
    imageDir = "../data/"
    imgPoints1 = []  # List to store points clicked on the first image canvas
    imgPoints2 = []  # List to store points clicked on the second image canvas
    matrixA = []  # Input matrix for computing the fundamental matrix
    dot_color = "#FF0000"
    main_window = Tk()
    main_window.title("Project 2")

    button1 = ttk.Button(main_window, text="Select 1st image", command=selectImg1)
    button2 = ttk.Button(main_window, text="Select 2nd image", command=selectImg2)
    button1.grid(row=0, column=0, padx=10, pady=10)
    button2.grid(row=0, column=1, padx=10, pady=10)

    frame1 = ttk.Frame(main_window)
    frame1.grid(row=1, column=1)
    frame1.config(relief=SOLID)
    frame2 = ttk.Frame(main_window)
    frame2.grid(row=1, column=2)
    frame2.config(relief=SOLID)

    canvas1 = Canvas(frame1, width=500, height=500)
    canvas1.bind('<Button-1>', img1ClickHandler)
    canvas1.pack()
    canvas2 = Canvas(frame2, width=500, height=500)
    canvas2.bind('<Button-1>', img2ClickHandler)
    canvas2.pack()

    frame3 = ttk.Frame(main_window)
    frame3.grid(row=2, column=0)
    frame3.config()

    epipolarMode = BooleanVar()
    epiModeChk = ttk.Checkbutton(frame3, text="Epipolar Mode", variable=epipolarMode)
    epiModeChk.pack()

    frame4 = ttk.Frame(main_window)
    frame4.grid(row=2, column=1)
    frame4.config()

    pixelMode = BooleanVar()
    pixModeChk = ttk.Checkbutton(frame4, text="Pixel Matching Mode", variable=pixelMode)
    pixModeChk.pack()

    img1 = cv2.imread("../data/left.png")
    img2 = cv2.imread("../data/right.png")
    matrixF = calculate_fundamental_matrix(img1, img2)

    main_window.mainloop()

    print("img1:", imgPoints1)
    print("img2:", imgPoints2)
