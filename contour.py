import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import scale
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray


img = imread('CatanBoardStockimage.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

scale_percent = 20
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

resized = cv2.resize(img, dim)

# cv2.imshow('image', resized)
# cv2.waitKey(0)


# attempting to find contours on the board



gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

# Thresh is the binary (black and white) threshold mapping of the grayscale image
# Threshold takes a grayscale and converts things that are close to white to white, and things that are close to black to black
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# We only care about contours, that holds all of the contours that the findContours() returns
# Provide findContours() with a threshold mapping and it uses the boundaries between the black and white to generate an edge
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Remove the first contour from the list, this one is usually the entire screen
cont = contours[1:]

# Iterate through the list of contours and only display the ones that have an area between the two values given
for i in range(0, len(cont)):
    print(cv2.contourArea(cont[i]))
    #need to find the appropriate threshold
    if 1000 > cv2.contourArea(cont[i]) > 10:
        frame = cv2.drawContours(resized, cont, i, (0, 255, 0), 3)  # Draw the contour on the original frame from the camera

# Show the frame captured from the camera
cv2.imshow('frame', frame)
cv2.waitKey(0)



