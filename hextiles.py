import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import scale
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray
from shapedetector import ShapeDetector
import imutils
from math import sqrt, isnan
import queue
from skimage import img_as_float, img_as_int
import random


def angle(pt1, center, pt3, printCos=False):
        p12 = pt1 - center
        p23 = pt3 - center

        cosine_angle = np.dot(p12, p23) / (np.linalg.norm(p12) * np.linalg.norm(p23))
        if printCos:
                print(cosine_angle)

        angle = np.arccos(cosine_angle)
        if(isnan(angle)):
                return -1
        return np.degrees(angle)
        
# Made a blank template for a catan board
def make_template():
        img = cv2.imread('CatanBoardTemplate.jpg')
        resize = cv2.resize(img, (970,850))
        white_board = np.zeros((850, 970, 3))
        print(resize.shape)
        print(white_board.shape)
        for y in range(resize.shape[0]):
                for x in range(resize.shape[1]):
                        if resize[y,x][0] > 250 and resize[y,x][1] < 2 and resize[y,x][2] < 2:
                                white_board[y,x] = (255,0,0)
        
        # cv2.imshow('test', white_board)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('CatanTemplateBlank.jpg', white_board)



# legacy distance function, replaced by Professor Fouhey's code
def distance(pt1, pt2):
        x,y = pt1
        x1,y1 = pt2

        distance = np.sqrt( np.square(x1-x)   + np.square(y1 - y))
        return distance


def thresholding():
        # Load, resize, and blur the image
        img = cv2.imread('CatanBoardStock2.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = 970
        height = 850
        dim = (width, height)
        resized = cv2.resize(img, dim)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
        gray = cv2.medianBlur(gray, 5)


        ###
        # Between the three # symbols, are various takes on what we tried to pass to the find contours
        # function, only use one at a time if you would like to run this code.
        # Threshhold the image, this code was from the cv2 examples on image detection with thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # This is just a seperate thresholding attempt, doesn't work
        # ret, thresh = cv2.threshold(gray, 140, 180, cv2.THRESH_BINARY_INV)

        # Find edges in the image to pass to find contours instead (we tried a variety of parameters, but none
        # gave the results we wanted)
        # edges = cv2.Canny(gray, 0, 200)
        # cv2.imshow('edges', edges)

        # Finally we tried color thresholding. 
        # code below is from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
        # lower_bound = np.array([80,80,80])
        # upper_bound = np.array([170,170,170])
        # mask = cv2.inRange(resized, lower_bound, upper_bound)
        # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # test = resized & mask_rgb
        # thresh = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        ###


        # find the contours of the image
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



        # Credit to this github for the contour inspiration, but this
        # code is reworked to fit our needs https://github.com/mikeStr8s/CV-Settlers

        # Remove the first contour from the list, this one is usually the entire screen
        cont = contours[:]

        # Iterate through the list of contours and only display the ones that have an area between the two values given
        for i in range(0, len(cont)):


                # We initially tried to limit the contour by area, as seen in the github linked above
                # but we scrapped this idea so that our code could work across multiple images
                # if 50000 > cv2.contourArea(cont[i]) > 10000:

                # Instead we used cv2 approxPolyDp to attempt to automatically find contours
                approx = cv2.approxPolyDP(cont[i], cv2.arcLength(cont[i], True) * .02, True)
                
                # initially we tried only drawing contours with six points, but this had very poor results
                if len(approx) == 6:
                        maxCosine = 0
                # if 1000000 > cv2.contourArea(cont[i]) > 10:
                frame = cv2.drawContours(resized, cont, i, (255, 0, 0), 3)  # Draw the contour on the original frame from the camera
                        # frame = cv2.drawContours(resized, cont, i , (0,0,255), 3)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        # # #         # Show the frame captured from the camera



def find_true_corners():
        # this code attempted to find corners of hexes by the distance they should be from other corners,
        # we didn't mention this in our report as this approach was short lived and not succesful
        dst = cv2.cornerHarris(gray, 5, 3, 0.04)
        corners = gray[dst>.05 * dst.max()]
        true_corners = []
        for index in range(corners[0].shape[0]):
                x,y = corners[0][index], corners[1][index]
                true_corner = True
                for index2 in range(corners[0].shape[0]):
                        if index != index2:
                                x1,y1 = corners[0][index2], corners[1][index2]
                                dist = distance((x,y), (x1,y1))
                                if dist < 1:
                                        true_corner = False
                if true_corner:
                        true_corners.append((x,y))


def corner_DFS():
        img = cv2.imread('CatanBoardStock2.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = 970
        height = 850
        dim = (width, height)
        resized = cv2.resize(img, dim)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
        gray = cv2.medianBlur(gray, 5)


        # find corners
        dst = cv2.cornerHarris(gray, 5, 3, 0.04)
        dst = cv2.dilate(dst, None)
        
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)   
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        
        # take the centroids of those corners
        centroids = np.float64(centroids)
        centroids_for_points = np.uint64(centroids)

        # need centroids for angle calculations to be floats, but corners to be ints
        centroids = np.float64(corners)
        centroids_for_points = np.uint64(corners)
        count = 0

        # Calculate distances between every point (credit to Professor Fouhey for this code)
        centroidsN = np.sum(centroids**2,axis=1,keepdims=True)
        D = centroidsN + centroidsN.T - 2*np.dot(centroids, centroids.T)
        D = D **(1/2)

        connections = {}
        # Updates the list of connections that we have
        def updateConnections(index1, index2, index3):
                if index2 not in connections:
                        connections[index2] = set()
                
                connections[index2].add(index1)
                connections[index2].add(index3)

                if index1 not in connections:
                        connections[index1] = set()
                
                connections[index1].add(index2)
                
                if index3 not in connections:
                        connections[index3] = set()
                
                connections[index3].add(index2)
        
                
        # Loops through every set of three points, if each point is within our distance threshold and angle
        # threshold, update the connections
        for index1 in range(centroids.shape[0]):
                pt1 = centroids[index1]
                for index2 in range(centroids.shape[0]):
                        pt2 = centroids[index2]
                        # prev 81 90
                        if index1 != index2 and 81 < D[index1][index2] < 90:
                                for index3 in range(centroids.shape[0]):
                                        if index3 != index1 and index3 != index2 and 81 < D[index2][index3] < 90:
                                                pt3 = centroids[index3]
                                                ang = angle(np.array(pt1), np.array(pt2), np.array(pt3))
                                                
                                                if index1 == 89 and index2 == 41 and index3 == 18:
                                                        print(pt1, pt2, pt3)
                                                        print('Angle:', angle(pt1, pt2, pt3))
                                                
                                                pt1 = centroids_for_points[index1]
                                                pt2 = centroids_for_points[index2]
                                                pt3 = centroids_for_points[index3]
                                                

                                                # if index1 == 18 and index2 == 41 and index3 == 89:
                                                # #         # cv2.circle(resized, (pt1[0], pt1[1]), 2, (255,0,0), 2)
                                                # #         # cv2.circle(resized, (pt2[0], pt2[1]), 2, (255,0,0), 2)
                                                # #         # cv2.circle(resized, (pt3[0], pt3[1]), 2, (255,0,0), 2)
                                                        
                                                #         print('Angle:', angle(pt1, pt2, pt3))
                                                pt1 = centroids[index1]
                                                pt2 = centroids[index2]
                                                
                                                
                                                if 117 < ang < 125:
                                                        updateConnections(index1, index2, index3)
                                                        # error checking
                                                        # cv2.line(resized, (pt1[0], pt1[1]), (pt2[0], pt2[1]),(255,0,0), 2)
                                                        # cv2.line(resized, (pt2[0], pt2[1]), (pt3[0], pt3[1]), (255,0,0), 2)
                                                        # cv2.circle(resized, (pt1[0], pt1[1]), 2, (255,0,0), 2)
                                                        # cv2.circle(resized, (pt2[0], pt2[1]), 2, (255,0,0), 2)
                                                        # cv2.circle(resized, (pt3[0], pt3[1]), 2, (255,0,0), 2)

        
        hexagon = set()
        stack = []
        dp = {}
        connections = {}

        # cv2.imshow('dst', resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        """
        This is all debugging/error checking
        try:
                print(connections[18])
        except:
                print('Not: ', 213)
                pass
        try:
                print(connections[41])
        except:
                print('Not: ', 183)
                pass
        print(connections[89])
        print(connections[113])
        print(connections[88])
        print(connections[39])

        cv2.circle(resized, (centroids_for_points[182][0], centroids_for_points[182][1]), 2, (255,0,0), 2)
        cv2.circle(resized, (centroids_for_points[213][0], centroids_for_points[213][1]), 2, (0,255,0), 2)
        cv2.circle(resized, (centroids_for_points[270][0], centroids_for_points[270][1]), 2, (0,0,255), 2)
        cv2.circle(resized, (centroids_for_points[289][0], centroids_for_points[289][1]), 2, (0,0,0), 2)
        cv2.circle(resized, (centroids_for_points[269][0], centroids_for_points[269][1]), 2, (255,255,255), 2)
        cv2.circle(resized, (centroids_for_points[212][0], centroids_for_points[212][1]), 2, (255,0,255), 2)
        """
        hexagons = []

        # retrace takes in a start point, and a dictionary of connections, and remakes
        # our detected hexagon, calculating the interior angles
        # and the sum of the interior angles
        def retrace(dp, point):
                print('here')
                potential = []
                start_point = point
                current = dp[start_point]
                potential.append(start_point)
                
                while current != start_point:
                        potential.append(current)
                        current = dp[current]
                
                angle_sum = 0
                hexagon = set(potential)
                if len(potential) != 6 or hexagon in hexagons:
                        return
                
                for i in range(4):
                        pt1 = centroids[potential[i]]
                        pt2 = centroids[potential[i + 1]]
                        pt3 = centroids[potential[i + 2]]
                        angle_sum += angle(pt1, pt2, pt3)
                        if  angle(pt1, pt2, pt3) > 140 or angle(pt1, pt2, pt3) < 100:
                                return
                
                pt1 = centroids[potential[4]]
                pt2 = centroids[potential[5]]
                pt3 = centroids[potential[0]]
                if  angle(pt1, pt2, pt3) > 140 or angle(pt1, pt2, pt3) < 100:
                                return
                angle_sum += angle(pt1, pt2, pt3)

                pt1 = centroids[potential[5]]
                pt2 = centroids[potential[0]]
                pt3 = centroids[potential[1]]
                if  angle(pt1, pt2, pt3) > 140 or angle(pt1, pt2, pt3) < 100:
                                return
                angle_sum += angle(pt1, pt2, pt3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                for i in potential:
                        point_ = centroids_for_points[i]
                        cv2.putText(resized, 'corner_{}'.format(i),(point_[0], point_[1]), font, .5, (0,0,0), 1, cv2.LINE_AA)
                        
                

                print(angle_sum, potential)
                if 700 < angle_sum < 740:
                        hexagons.append(hexagon)
                        start_point = point
                        current = dp[start_point]
                        
                        pt1 = centroids_for_points[start_point]
                        pt2 = centroids_for_points[current]
                        cv2.line(resized, (pt1[0], pt1[1]), (pt2[0], pt2[1]),(255,0,0), 2)
                        cv2.line(white_board, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255,0,0), 2)

                        while current != start_point:
                                pt1 = centroids_for_points[current]
                                pt2 = centroids_for_points[dp[current]]
                                cv2.line(resized, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255,0,0), 2)
                                cv2.line(white_board, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255,0,0), 2)
                                current = dp[current]



        # The below code is where the real magic happens,
        # we take every point that has a connection, and do a depth
        # first search, attemping to get back to the start point
        # if we can make it back to the start point in six steps,
        # then those potential points are passed to the retrace function and
        # checked to see if they make a hexagon
        for start_point in connections.keys():
                visited = set()
                
                # start_point = 223

                # print('Start point:', start_point)
                # start_point = 18
                for conn in connections[start_point]:
                        stack = []
                        dp = {}
                        iterations = {}
                        visited = set()
                        stack.append(conn)
                        iterations[conn] = 1
                        visited.add(conn)
                        dp[conn] = start_point
                
                        visited.add(start_point)
                        # print('***********************************New direction***********************************')
                        # if conn == 39 or conn == 41:
                        while stack:
                                # print(stack)
                                current = stack.pop()
                                # print(current)
                                # print(iterations)
                                if current in connections:
                                        for point in connections[current]:
                                                if current in iterations and iterations[current] == 5 and point == start_point:
                                                        dp[start_point] = current
                                                        retrace(dp, point)
                                                        iterations[current] = 0
                                        for point in connections[current]:
                                                if point not in visited:
                                                        if iterations[current] + 1 < 6:
                                                                stack.append(point)
                                                                dp[point] = current
                                                                iterations[point] = iterations[current] + 1
                                                                visited.add(point)

                
        cv2.imshow('hexagons', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def legacy_code():
        # This function just contains legacy code, stuff we used for error checking and experimenting


        # corner checking
        # print(connections)

        # pt1 = centroids[88]
        # pt2 = centroids[111]
        # pt3 = centroids[181]
        # pt4 = centroids[212]
        # pt5 = centroids[182]
        # pt6 = centroids[113]
        

        # ang = angle(pt1, pt2, pt3)
        # print(ang)
        # dist1 = D[39][18]
        # dist2 = D[18][41]
        # print(dist1, dist2)


        # pt1 = centroids_for_points[88]
        # pt2 = centroids_for_points[111]
        # pt3 = centroids_for_points[181]
        # pt4 = centroids_for_points[212]
        # pt5 = centroids_for_points[182]
        # pt6 = centroids_for_points[113]

        # cv2.line(resized, (pt1[0], pt1[1]), (pt2[0], pt2[1]),(255,0,0), 2)
        # cv2.line(resized, (pt2[0], pt2[1]), (pt3[0], pt3[1]), (255,0,0), 2)
        # cv2.line(resized, (pt3[0], pt3[1]), (pt4[0], pt4[1]), (255,0,0), 2)
        # cv2.line(resized, (pt4[0], pt4[1]), (pt5[0], pt5[1]), (255,0,0), 2)
        # cv2.line(resized, (pt5[0], pt5[1]), (pt6[0], pt6[1]), (255,0,0), 2)
        # cv2.line(resized, (pt6[0], pt6[1]), (pt1[0], pt1[1]), (255,0,0), 2)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # for index, point in enumerate(centroids_for_points):
        #         cv2.circle(resized, (point[0], point[1]), 2, (0,0,255), 2)
                # if index == 96 and index != 85:
                #         cv2.putText(resized, 'corner_{}'.format(index),(point[0], point[1]), font, .5, (0,0,0), 1, cv2.LINE_AA)


        
        # resized[centroids] = [255,0,0]
        # print(resized[centroids[0][1], centroids[0][0]])
        # resized[dst > .01 * dst.max()] = [255,0,0]
        # cv2.imsave('isolated-tiles.jpg', resized)
        # cv2.imshow('dst', white_board)
        # cv2.imwrite('CornerDetection.jpg', resized)
        # cv2.imshow('dst', resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # an attempt to draw lines through the board to mathmatically find hex tiles
        # (eventually repurposed for template matching)
        # # y,x or row,col
        # upper_left = (float('inf'), float('inf'))
        # upper_right = (float('inf'), float('-inf'))
        # mid_left = (float('inf'), float('inf'))
        # mid_right = (float('inf'), float('-inf'))
        # bottom_left = ( float('-inf'), float('inf'))
        # bottom_right = (float('-inf'), float('-inf'))


        # for index in range(corners[0].shape[0]):
        #         y,x = corners[0][index], corners[1][index]

        #         if y < upper_left[0] and x < upper_left[1]:
        #                 upper_left = (y,x)
        #         if upper_right[0] == float('inf'):
        #                 if y < upper_right[0] and x > upper_right[1]:
        #                         upper_right = (y,x)
        #         else:
        #                 if abs(y - upper_right[0]) < 3 and x > upper_right[1]:
        #                         upper_right = (y,x)
                

        #         if x < mid_left[1]:
        #                 mid_left = (y,x)
        #         if x > mid_right[1]:
        #                 mid_right = (y,x)
                
                
        #         if y > bottom_left[0]:
        #                 bottom_left = (y,x)
                
        #         if y > bottom_right[0] and abs(upper_right[1] - x) < 10:
        #                 bottom_right = (y,x)


        # print(upper_left)
        # print(upper_right)
        # print(mid_left)
        # print(mid_right)
        # print(bottom_left)
        # print(bottom_right)
                
        # cv2.circle(resized, (upper_left[1], upper_left[0]), 2, (255,0,0), 2)
        # cv2.circle(resized, (upper_right[1], upper_right[0]), 2, (255,0,0), 2)
        # cv2.circle(resized, (mid_left[1], mid_left[0]), 2, (255,0,0), 2)
        # cv2.circle(resized, (mid_right[1], mid_right[0]), 2, (255,0,0), 2)
        # cv2.circle(resized, (bottom_left[1], bottom_left[0]), 2, (255,0,0), 2)
        # cv2.circle(resized, (bottom_right[1], bottom_right[0]), 2, (255,0,0), 2)

        # def drawLine(pt1, pt2):
        #         cv2.line(resized, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (255,0,0), 2)
        

        # drawLine(upper_left, bottom_right)
        # drawLine(upper_right, bottom_left)
        # drawLine(mid_left, mid_right)

        
        # resized[dst>.01 * dst.max()] = [255,0,0]

        # cv2.imshow('dst', resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def template_matching():


        img = cv2.imread('CatanBoardStock2.jpg')
        # cv2.imshow('test', img)
        # cv2.waitKey(0)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = 970
        height = 850

        dim = (width, height)

        resized = cv2.resize(img, dim)
        ratio = img.shape[0] / float(resized.shape[0])
        white_board = np.zeros(dim)

        # white_board = cv2.resize(white_board, dim)
        # cv2.imwrite('template.jpg', white_board)
        # cv2.imshow('image', white_board)
        # cv2.waitKey(0)
        # return
        



        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
        gray = cv2.medianBlur(gray, 5)


        # this function finds the most extreme six points,
        # should be the upper left, upper right, mid left, mid right,
        # bottom left and bottom right corners
        def find_extremes(gray):
                
                dst = cv2.cornerHarris(gray, 10, 3, .04)
                # print(dst.shape)
                dst = cv2.dilate(dst, None)
                
                ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
                dst = np.uint8(dst)
                ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                
                centroids = np.float64(centroids)
                corners = np.uint64(centroids)

                # for corner in corners:
                #         cv2.circle(resized, (corner[0], corner[1]), 2, (255,0,0), 2)

                upper_left = (float('inf'), float('inf'))
                upper_right = (float('inf'), float('-inf'))
                mid_left = (float('inf'), float('inf'))
                mid_right = (float('inf'), float('-inf'))
                bottom_left = ( float('-inf'), float('inf'))
                bottom_right = (float('-inf'), float('-inf'))

                # print(len(corners))
                for index in range(len(corners)):
                        y,x = corners[index][1], corners[index][0]

                        if y < upper_left[0] and x < upper_left[1]:
                                upper_left = (y,x)
                        if upper_right[0] == float('inf'):
                                if y < upper_right[0] and x > upper_right[1]:
                                        upper_right = (y,x)
                        else:
                                if abs(y - upper_left[0]) < 3 and x > upper_right[1]:
                                        upper_right = (y,x)
                        

                        if x < mid_left[1]:
                                mid_left = (y,x)
                        if x > mid_right[1]:
                                mid_right = (y,x)
                        
                        
                        if y > bottom_left[0]:
                                bottom_left = (y,x)
                        
                        if y > bottom_right[0] and abs(upper_right[1] - x) < 10:
                                bottom_right = (y,x)
                return [upper_left, upper_right, mid_left, mid_right, bottom_left, bottom_right]


        # template matching start
        img = cv2.imread('CatanBoardStock2.jpg')
        width = 970
        height = 850

        dim = (width, height)

        resized = cv2.resize(img, dim)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
        gray = cv2.medianBlur(gray, 5)

        extremes = find_extremes(gray)

        # for point in extremes:
        #         cv2.circle(resized, (point[1], point[0]), 2, (255,0,0), 2)

        img_temp = cv2.imread('CatanTemplateBlank.jpg')
        width = 970
        height = 850

        dim = (width, height)

        resized_temp = cv2.resize(img_temp, dim)

        gray_temp = cv2.cvtColor(resized_temp, cv2.COLOR_BGR2GRAY) 
        gray_temp = cv2.medianBlur(gray_temp, 5)

        extremes_temp = find_extremes(gray_temp)
        matches = []
        # after finding the extremes for the template and the board to detect, place them in a matched array
        for index in range(len(extremes)):
                matches.append((extremes_temp[index], extremes[index]))

        
        # Get a homography
        max_inliers = []
        best_h = None
        
        # Use RANSAC to calculate a homography
        # This code is from P3
        for i in range(1000):
                points = random.sample(matches, 4)
                points = np.array(points)
                points = np.float64(points)
                
                inliers = []
                points_x_y = []
                p_mat = []
                for i in points:
                        x,y,x_dest, y_dest = i[0][1], i[0][0], i[1][1], i[1][0]
                        
                        p_mat.append([-x, -y, -1, 0, 0, 0, x * x_dest, y * x_dest, x_dest])
                        p_mat.append([0,0,0, -x, -y, -1, x * y_dest, y * y_dest, y_dest])
                        points_x_y.append([x,y,x_dest,y_dest])

                p_mat = np.float32(np.array(p_mat))
                # Find the homography
                u, s, vh = np.linalg.svd(p_mat)
                vh = np.array(vh)
                h = vh[-1]
                h = np.float32(h)
                h = h / h[8]
                h = np.reshape(h, (3,3))
                for match in matches:
                        point = (match[0][1], match[0][0])
                        point2 = (match[1][1], match[1][0])
                        # point = np.array(point)
                        # point = np.float64(point)
                        # point2 = np.array(point2)
                        # point2 = np.float64(point2)
                        
                        original = np.transpose(np.array([point[0], point[1], 1]))
                        original = np.float64(original)
                        estimate = np.dot(h, original)
                        estimate = estimate / estimate[2]
                        estimate = np.array([estimate[0], estimate[1], 1])

                        d = np.sqrt(np.square(point2[0] - estimate[0]) + np.square(point2[1] - estimate[1]))

                        if d < 5:
                                inliers.append(match)
        
                if len(inliers) > len(max_inliers):
                        max_inliers = inliers
                        best_h = h

        
        print(best_h)
        left = img_as_float(resized)
        right = img_as_float(resized_temp)

        width = left.shape[1]
        height = left.shape[0]


        top_left = np.matmul(best_h, np.transpose(np.array([0,0,1])))
        top_right = np.matmul(best_h, np.transpose(np.array([width, 0, 1])))
        bot_right = np.matmul(best_h, np.transpose(np.array([0,height,1])))
        bot_left = np.matmul(best_h, np.transpose(np.array([width, height,1])))

        min_height = min(top_left[1], top_right[1], bot_right[1], bot_left[1])
        min_width = min(top_left[0], top_right[0], bot_left[0], bot_right[0])
        max_height = max(top_left[1], top_right[1], bot_right[1], bot_left[1])
        max_width = max(top_left[0], top_right[0], bot_left[0], bot_right[0])

        print(min_height, min_width, max_height, max_width)
        offset_homography = np.array([[1,0, -min_width], [0,1,-min_height], [0,0,1]])


        best_h = np.float32(best_h)
        left_warped = cv2.warpPerspective(left, np.dot(offset_homography, best_h), (int(max_width - min_width), int(max_height - min_height)))
        right_warped = cv2.warpPerspective(right, offset_homography, (int(right.shape[1] - min_width), int(right.shape[0] - min_height)))

        output = np.zeros((right_warped.shape), np.float32)
        
        
        # print(left_warped[10][10])
        for i in range(right_warped.shape[0]):
                for j in range(right_warped.shape[1]):
                        if i >= len(left_warped) or j >= len(left_warped[i]):
                                output[i][j] = right_warped[i][j]
                        # elif np.array_equal(left_warped[i][j],np.array([0,0,0])) and not np.array_equal(right_warped[i][j],np.array([0,0,0])):
                        #         output[i][j] = right_warped[i][j]
                        # elif not np.array_equal(left_warped[i][j],np.array([0,0,0])) and np.array_equal(right_warped[i][j],np.array([0,0,0])):
                        #         output[i][j] = left_warped[i][j]
                        # else:
                        #         output[i][j] = left_warped[i,j]
                        # repurposed P3 code, only want the template image where it is blue,
                        # otherwise the board image is fine
                        elif right_warped[i][j][0] > .5:
                                output[i][j] = [right_warped[i][j][2], right_warped[i][j][1], right_warped[i][j][0]]
                                # print(right_warped[i][j])
                        else:
                                output[i][j] = [left_warped[i][j][2], left_warped[i][j][1], left_warped[i][j][0]]
                        
                        # print(right_warped[i][j])

        
        # imsave('output.jpg', output)
        cv2.imshow('test', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




        


if __name__ == '__main__':
        template_matching()



