import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import scale
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray
from shapedetector import ShapeDetector
import imutils

def contours():

        img = imread('CatanBoardStockImage.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # scale_percent = 20
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        width = 970
        height = 850
        dim = (width, height)

        resized = cv2.resize(img, dim)
        ratio = img.shape[0] / float(resized.shape[0])

        # cv2.imshow('image', resized)
        # cv2.waitKey(0)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

        # attempting to find contours on the board

        #this is all experimenting with shape detection
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        # cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # sd = ShapeDetector()

        # # loop over the contours
        # for c in cnts:
        # 	# compute the center of the contour, then detect the name of the
        # 	# shape using only the contour
        # 	M = cv2.moments(c)
        # 	cX = int((M["m10"] / M["m00"]) * ratio)
        # 	cY = int((M["m01"] / M["m00"]) * ratio)
        # 	shape = sd.detect(c)

        # 	# multiply the contour (x, y)-coordinates by the resize ratio,
        # 	# then draw the contours and the name of the shape on the image
        # 	c = c.astype("float")
        # 	c *= ratio
        # 	c = c.astype("int")
        # 	cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
        # 	cv2.putText(resized, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        # 		0.5, (255, 255, 255), 2)

        # 	# show the output image
        # 	cv2.imshow("Image", resized)
        # 	cv2.waitKey(0)

        #end shape detection experimenting

        #this is experimenting with edge detection 

        edges = cv2.Canny(resized, 100, 200)
        # cv2.imshow('edges', edges)
        # cv2.waitKey(0)

        # end edge experimentation

        # Thresh is the binary (black and white) threshold mapping of the grayscale image
        # Threshold takes a grayscale and converts things that are close to white to white, and things that are close to black to black
        # this is where contour code begins
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # We only care about contours, that holds all of the contours that the findContours() returns
        # Provide findContours() with a threshold mapping and it uses the boundaries between the black and white to generate an edge
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Remove the first contour from the list, this one is usually the entire screen
        cont = contours[1:]

        # Iterate through the list of contours and only display the ones that have an area between the two values given
        for i in range(0, len(cont)):
                # print(cv2.contourArea(cont[i]))
                #need to find the appropriate threshold
                if 50000 > cv2.contourArea(cont[i]) > 1000:
                                frame = cv2.drawContours(resized, cont, i, (255, 0, 0), 3)  # Draw the contour on the original frame from the camera

                # Show the frame captured from the camera
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

        # this is where contour code ends except the return obviously

        return cont 


if __name__ == '__main__':
	contours()



