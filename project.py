# import os
# import random
# import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray
# from skimage.draw import circle
from sklearn.preprocessing import scale
from math import sqrt

img_color = imread('CatanBoardStockImage.jpg')
img = rgb2gray(imread('CatanBoardStockImage.jpg'))

img = scale(img, axis=0, with_mean=True, with_std=True)

blobs_log = blob_log(img, min_sigma=30, max_sigma=40, num_sigma=20, threshold=.3)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

fig, axes = plt.subplots(1,1)
print('here')
# plt.subplot(img_color)
count = 1

for row in blobs_log:
    x = int(row[0])
    y = int(row[1])
    r = row[2]
    axes.add_patch(plt.Circle((x,y), r, color='red'))
    print('Circle: ', count)
    count += 1

    
axes.imshow(img_color)
plt.show()

