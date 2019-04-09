# import os
# import random
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
# from skimage.io import imread, imsave
# from skimage.feature import blob_log
from skimage.color import rgb2gray
# from skimage.draw import circle
from sklearn.preprocessing import scale
from math import sqrt

# img_color = imread('CatanBoardStockImage.jpg')
# img = rgb2gray(imread('CatanBoardStockImage.jpg'))

# img = scale(img, axis=0, with_mean=True, with_std=True)

# blobs_log = blob_log(img, min_sigma=35, max_sigma=37, num_sigma=20, threshold=.5)
# blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

# fig, axes = plt.subplots(1,1)
# print('here')
# # plt.subplot(img_color)
# count = 1

# for row in blobs_log:
#     x = int(row[0])
#     y = int(row[1])
#     r = row[2]
#     axes.add_patch(plt.Circle((x,y), r, color='red'))
#     print('Circle: ', count)
#     count += 1

    
# axes.imshow(img_color)
# plt.show()


# Hough Circles attempt
img = cv2.imread('IMG_3043.JPG')
scale_percent = .4
width = int(970)
height = int(850)
dim = (width, height)
img = cv2.resize(img, dim)
imgb = cv2.medianBlur(img,5)
gimg = cv2.cvtColor(imgb,cv2.COLOR_BGR2GRAY)
gimg = cv2.medianBlur(gimg,5)

circles = cv2.HoughCircles(gimg,cv2.HOUGH_GRADIENT,1,20,
                            param1=60,param2=22,minRadius=20,maxRadius=30)
circles = np.uint16(np.around(circles))
# print(circles[0])
count = 0
for i in circles[0, :]:
    
    cv2.circle(img, (i[0], i[1]), i[2], (255,0,0), 2)
    print('Circle: ', count)
    count += 1
cv2.imshow('Detected', img)
cv2.imwrite('CustomImage.jpeg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()