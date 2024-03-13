#Created on 05.10.2016
#Modified on 23.12.2020


import numpy as np
import cv2
import matplotlib.pyplot as plt


# do not import more modules!


def drawCircle(img, x, y):

    #Draw a circle at circle of radius 5px at (x, y) stroke 2 px
    #This helps you to visually check your methods.
    #:param img: a 2d nd-array
    #param y:
    #:param x:
    #:return: img with circle at desired position

    cv2.circle(img, (x, y), 5, 255, 2)
    return img


def binarizeAndSmooth(img) -> np.ndarray:

   # First Binarize using threshold of 115, then smooth with gauss kernel (5, 5)
   # :param img: greyscale image in range [0, 255]
    #:return: preprocessed image

    # making the picture black and white

    th, im_th = cv2.threshold(img, 115, 255, cv2.THRESH_BINARY)

    # using gaussian filter
    new = cv2.GaussianBlur(im_th, (5, 5), 0)



    return new


def drawLargestContour(img) -> np.ndarray:

   # find the largest contour and return a new image showing this contour drawn with cv2 (stroke 2)
   # :param img: preprocessed image (mostly b&w)
   # :return: contour image

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # create an empty image for contours
    img_contours = np.zeros(img.shape)
    # draw the contours on the empty image
    contourImg = cv2.drawContours(img_contours, contour, -1, (255, 255, 255), 2)


    return contourImg


def getFingerContourIntersections(contour_img, x) -> np.ndarray:

   # Run along a column at position x, and return the 6 intersecting y-values with the finger contours.
   # (For help check Palmprint_Algnment_Helper.pdf section 2b)
   # :param contour_img:
   # :param x: position of the image column to run along
   # :return: y-values in np.ndarray in shape (6,)

    # creating zero image in the same size of img
    empty = np.zeros(contour_img.shape)
    # draw vertical line
    line = cv2.line(empty, pt1=(x, 0), pt2=(x, 240), color=(255, 255, 255), thickness=1)

    # calculating where line and contours are meeting
    overlap = np.logical_and(contour_img, line)
    # only true remain
    overlap = np.array(overlap)
    width = overlap.shape[0]
    list = []

    height = overlap.shape[1]
    for i in range(1, width - 3):
        for j in range(1, height):
            # check if it's a new counter
            if overlap[i + 1][j] == True and overlap[i][j] == False:
                list.append(i + 1)

    list = list[:6]

    res_array = np.array(list)

    return res_array


def findKPoints(img, y1, x1, y2, x2) -> tuple:
   
    #given two points and the contour image, find the intersection point k
    #:param img: binarized contour image (255 == contour)
    #:param y1: y-coordinate of point
    #:param x1: x-coordinate of point
    #:param y2: y-coordinate of point
   # :param x2: x-coordinate of point
    #:return: intersection point k as a tuple (ky, kx)
    

    theta = np.arctan2(y1 - y2, x1 - x2)
    endpt_x = int(x1 - img.shape[0] * np.cos(theta))
    endpt_y = int(y1 - img.shape[0] * np.sin(theta))

    line_img = np.zeros_like(img)
    cv2.line(line_img, pt1=(x1, y1), pt2=(endpt_x, endpt_y), color=(255, 255, 255), thickness=1)

    overlap = np.logical_and(img, line_img)
    overlap = np.array(overlap)
    width = overlap.shape[0]

    k = (0, 0)

    height = overlap.shape[1]
    for i in range(1, width - 3):
        for j in range(1, height):
            # check if it's a new counter
            if overlap[i + 1][j] == True and overlap[i][j] == False:
                k = (i, j)

    return k


def getCoordinateTransform(k1, k2, k3) -> np.ndarray:
    
   # Get a transform matrix to map points from old to new coordinate system defined by k1-3
   # Hint: Use cv2 for this.
   # :param k1: point in (y, x) order
   # :param k2: point in (y, x) order
   # :param k3: point in (y, x) order
   # :return: 2x3 matrix rotation around origin by angle

    y_axis = np.zeros((240, 320, 3))
    cv2.line(y_axis, pt1=(k1[1], k1[0]), pt2=(k3[1], k3[0]), color=(255, 0, 0), thickness=2)

    l_y = k3[0] - k1[0]
    l_x = k3[1] - k1[1]

    point = (k2[1] + l_y * 320, k2[0] - (k3[1] - k1[1]) * 320)

    x_axis = np.zeros((240, 320, 3))
    cv2.line(x_axis, pt1=(k2[1], k2[0]), pt2=point, color=(255, 255, 0), thickness=1)

    center = (0, 0)

    for i in range(240):
        for j in range(320):
            if y_axis[i][j][0] == 255.0:
                if x_axis[i][j][0] == 255.0:
                    center = i, j

    angle = np.arctan(l_x/l_y)*180/np.pi

    matrix = cv2.getRotationMatrix2D(center, -angle, 1)

    return matrix


def palmPrintAlignment(img):
    '''
    Transform a given image like in the paper using the helper functions above when possible
    :param img: greyscale image
    :return: transformed image
    '''

    # TODO threshold and blur
    blured = binarizeAndSmooth(img)

    # TODO find and draw largest contour in image
    contour = drawLargestContour(blured)

    # TODO choose two suitable columns and find 6 intersections with the finger's contour
    (y11, y12, y13, y14, y15, y16) = getFingerContourIntersections(contour, 10)
    (y21, y22, y23, y24, y25, y26) = getFingerContourIntersections(contour, 18)

    # TODO compute middle points from these contour intersections
    hole1_y1 = int((y12-y11)/2) + y11
    hole1_y2 = int((y22-y21)/2) + y21
    hole2_y1 = int((y14-y13)/2) + y13
    hole2_y2 = int((y24-y23)/2) + y23
    hole3_y1 = int((y16-y15)/2) + y15
    hole3_y2 = int((y26-y25)/2) + y25

    # TODO extrapolate line to find k1-3
    k1 = findKPoints(img, hole1_y1, 10, hole1_y2, 18)
    k2 = findKPoints(img, hole2_y1, 10, hole2_y2, 18)
    k3 = findKPoints(img, hole3_y1, 10, hole3_y2, 18)

    # TODO calculate Rotation matrix from coordinate system spanned by k1-3
    matrix = getCoordinateTransform(k1, k2, k3)

    # TODO rotate the image around new origin
    rotated = cv2.warpAffine(img, matrix, (320, 240))

    return rotated