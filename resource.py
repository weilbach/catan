import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import scale
from skimage.io import imread, imsave
from skimage.feature import blob_log
from skimage.color import rgb2gray
from contour import contours

def get_resources(resized, gray):

    cont = contours()

    final = np.zeros(resized.shape, np.uint8)  # Unused, ignore

    # Mask acts as a hole in a black overlay allowing only what is inside the hole to be seen
    mask = np.zeros(gray.shape, np.uint8)

    hex_colors = []  # List of average colors inside each contour
    hexes = []  # List of contour centriods

    for i in range(0, len(cont)):
        #  If a contour is within specified area threshold
        if 1000 > cv2.contourArea(cont[i]) > 100:
            mask[...] = 0
            cv2.drawContours(mask, cont, i, 255, -1)  # Lay mask over the contour
            mean_val = cv2.mean(resized, mask)  # Gather the average color of the exposed camera resized
            x, y, w, h = cv2.boundingRect(cont[i])
            cx = int(x + w/2)
            cy = int(y + h/2)
            hexes.append((cx,cy))  # Add centroid
            hex_colors.append(mean_val)  # Add mean color
            cv2.drawContours(resized, cont, i, mean_val, -1)  # Graphically display mean color on screen for user

    cv2.imshow('colored contours', resized)
    cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    return hex_colors, hexes

def show_resources(image, hex_colors, hexes):

    resource_dict = {(90,125,125,0): 'wood', (110,170,160,0): 'sheep', (80,120,150,0):  'brick', (130,140,150,0): 'stone', (100,150,190,0): 'wheat', (180,215,235,0): 'desert'}

    resource_type = []  # List of each resource associated with the valid contours
    closest_color = ''

    # Itterate through the provided list of colors found inside the contours
    for a in hex_colors:
        min_dist = 1234567890  # Number to represent the distance between two colors
        f = np.array(a)  # Change RGB tuple 'a' to a numpy array 'f'

        # Itterate through possible resources
        for b in resource_dict.keys():
            r = np.array(b)  # Change RGB tuple 'b' to a numpy array 'r'
            dist = np.linalg.norm(f-r)  # Calculate the euclidean distance between the two numpy arrays

            # Find the smallest distance between the found colors and the colors that represent resources
            if dist < min_dist:
                min_dist = dist
                closest_color = resource_dict[b]

        # Add resource that was closest to the color in the contour
        resource_type.append(closest_color)

        print(len(resource_type))
        for index, i in enumerate(hexes):
            #the current issue is that we only have one resource type
            cv2.putText(image, resource_type[index], (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('frame', image)
        cv2.waitKey(0)
        return resource_type


if __name__ == '__main__':
    
    img = imread('CatanBoardStockimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    scale_percent = 20
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim)


    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hex_colors, hexes = get_resources(resized, gray)
    show_resources(resized, hex_colors, hexes)