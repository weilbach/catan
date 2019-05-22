# import os
# import random
import cv2
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread, imsave
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
img = cv2.imread('CatanBoardTemplate.jpg')
print(img.shape)
# scale_percent = .2
width = int(970)
height = int(850)
# dim = (int(img.shape[0] * scale_percent), int(img.shape[1] * scale_percent))
dim = (width, height)
img = cv2.resize(img, dim)
imgb = cv2.medianBlur(img,5)
gimg = cv2.cvtColor(imgb,cv2.COLOR_BGR2GRAY)
# gimg = cv2.medianBlur(gimg,5)

p1 = 60
p2 = 25
mR = 20
maxR = 30



circles = cv2.HoughCircles(gimg,cv2.HOUGH_GRADIENT,1,20,
                            param1=p1,param2=p2,minRadius=mR,maxRadius=maxR)
circles = np.uint16(np.around(circles))
# print(circles[0])
count = 0


for i in circles[0, :]:

    # The code commented below masked out circles to be fed to the CNN
    # we had to lookup how to do this but unfortunately cannot find the
    # stack overflow post we followed to learn this, but it is relatively
    # straight forward
    # circle_img = np.zeros((dim[1], dim[0]), np.uint8)
    # cv2.circle(circle_img, (i[0], i[1]), i[2], 1, thickness=-1)
    # mean_val_test = cv2.mean(gimg, mask=circle_img)
    # masked_data = cv2.bitwise_and(img, img, mask=circle_img)
    # # cv2.imwrite('./numbers/take_2_{}.jpg'.format(count), masked_data)
    # # cv2.imshow('Detected', masked_data)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # count += 1
    cv2.circle(img, (i[0], i[1]), i[2], (0,0,255), 2)
    # print('Circle: ', count)
    # src1_mask = cv2.cvtColor(masked_data, cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 

cv2.imwrite('NumbersStockBoard2.jpg', img)
cv2.imshow('Detected numbers', img)
cv2.waitKey(0)
cv2.destroyAllWindows()












# Legacy for now, an attempt at cropping the numbers
# img_arr = []
# width = 0
# height = 0
# for j in range(masked_data.shape[0]):
#     row = []
#     for k in range(masked_data.shape[1]):
#         if np.any(masked_data[j][k]) != 0:
#             row.append([img[j][k][2], img[j][k][1], img[j][k][0]])
#     if len(row) > 0:
#         if len(row) > width:
#             width = len(row)
#         img_arr.append(row)
#         height += 1

# img_new = np.zeros((height + 30, width + 30, 3), np.uint8)

# for j in range(height):
#     for k in range(width):
#         if k >= len(img_arr[j]):
#             img_new[j][k] = [0,0,0]
#         else:
#             img_new[j][k] = img_arr[j][k]
# # img_arr = np.array(img_arr)
# # print(img_arr.shape)
# count += 1
# imsave('./numbers/take2_{}.jpeg'.format(count), img_new)
# # cv2.imwrite('./numbers/take2_{}'.format(count), img_arr)
    

