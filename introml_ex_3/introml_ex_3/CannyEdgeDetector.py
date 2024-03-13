import numpy as np
import matplotlib.pyplot as plt
import scipy
#from scipy.ndimage import convolve
from convo import make_gauss_kernel
import cv2


def normalize(k):
    '''
    normalize chai gauss filter ko lagi banako taki code dherai lamo nahosh vanera in gauss filter
    Helper function to be used in gaussFilter in order to normalize the gauss kernel
    The normalization ensures that the average greylevel of the image remains the same when we blur the image
    with this kernel.
    The shape method returns a tuple representing the dimensions i.e. (rows & columns)
    '''
    total_sum = np.sum(k)
    k_height = k.shape[0]
    k_width = k.shape[1]
    for i in range(0, k_height):
        for j in range(0, k_width):
            k[i, j] = k[i, j] / total_sum
    return k

def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    kernel = make_gauss_kernel(ksize, sigma)
    #k_normalized = normalize(kernel)
    convolved= scipy.ndimage.convolve(img_in, kernel)
    return (kernel, convolved.astype(int))#cast filtered image back to int


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...
    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    Sobel Operator is a specific type of 2D derivative mask which is efficient in detecting 
    the edges in an image. We will use following two masks:
    """
    kernel_vert = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])
    kernel_horiz = np.array([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]])

    #convolve function flips the kernel, thats why we have to flip the horizontal kernel. Vertical flip, however is not required
    kernel_horiz_flip = np.flip(kernel_horiz)
    filtered_vert = scipy.ndimage.convolve(img_in, kernel_vert) 
    filtered_horiz = scipy.ndimage.convolve(img_in, kernel_horiz_flip) 
    return filtered_vert.astype(np.int32), filtered_horiz.astype(int) #cast filtered image back to int


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    mag_gradient = np.sqrt(np.square(gx) + np.square(gy))
    direction_grad_theta = np.arctan2(gy, gx)
    return mag_gradient.astype(int), direction_grad_theta #cast filtered image back to int


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle_degree = np.rad2deg(angle)
    while angle_degree > 180:
        angle_degree -= 180

    if 0 <= angle_degree <= 45:
        d = 45 - angle_degree
        if d > 22.5:
            angle_degree = 0
        else:
            angle_degree = 45

    if 45 < angle_degree <= 90:
        d = 90 - angle_degree
        if d > 22.5:
            angle_degree = 45
        else:
            angle_degree = 90

    if 90 < angle_degree <= 135:
        d = 135 - angle_degree
        if d > 22.5:
            angle_degree = 90
        else:
            angle_degree = 135

    if 135 < angle_degree <= 180:
        d = 180 - angle_degree
        if d > 22.5:
            angle_degree = 135
        else:
            angle_degree = 0

    return angle_degree

def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    The shape method returns a tuple representing the dimensions i.e. (rows & columns)
    """
    # TODO Hint: For 2.3.1 and 2 use the helper method above
    h = g.shape[0]
    w = g.shape[1]
    max_suppress = np.zeros(shape=(h, w))

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            converted_angle = convertAngle(theta[i, j])
            if converted_angle == 0:
                local_maxima = g[i, j]
                if local_maxima >= g[i, j + 1] and local_maxima >= g[i, j - 1]:
                    local_maxima = g[i, j]
                    '''
                    4  5  6
                    1* 5" 4*
                    5  6  0
                    '''
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima
            elif converted_angle == 45:
                local_maxima = g[i, j]
                if local_maxima >= g[i + 1, j - 1] and local_maxima >= g[i - 1, j + 1]:
                    local_maxima = g[i, j]
                    '''
                    4  5  6*
                    1  5" 4
                    5* 6  0
                    '''
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima
            elif converted_angle == 90:
                local_maxima = g[i, j]
                if local_maxima >= g[i + 1, j] and local_maxima >= g[i - 1, j]:
                    local_maxima = g[i, j]
                    '''
                    4  5* 6
                    1  5" 4
                    5  6* 0
                    '''
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima
            elif converted_angle == 135:
                local_maxima = g[i, j]
                if local_maxima >= g[i + 1, j + 1] and local_maxima >= g[i - 1, j - 1]:
                    local_maxima = g[i, j]
                    '''
                    4* 5  6
                    1  5" 4
                    5  6  0*
                    '''
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima

    '''
    4* 5* 6*
    1* 5" 4*
    5* 6* 0*
    '''
    return max_suppress

def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    h = max_sup.shape[0]
    w = max_sup.shape[1]
    classified_pixel = np.zeros(shape=(h, w))

    for i in range(0, h):
        for j in range(0, w):
            if max_sup[i, j] <= t_low:
                classified_pixel[i, j] = 0
            elif max_sup[i, j] > t_low and max_sup[i, j] <= t_high:
                classified_pixel[i, j] = 1
            else:
                classified_pixel[i, j] = 2

    classified_pixel = np.pad(classified_pixel, (1, 1), 'constant', constant_values=(0, 0))#padding the matrix with zeroes to deal with border pixels
    pad_h = classified_pixel.shape[0]
    pad_w = classified_pixel.shape[1]
    for x in range(1, pad_h - 1):
        for y in range(1, pad_w - 1):
            if classified_pixel[x, y] == 2:
                classified_pixel[x, y] = 255
                if classified_pixel[x, y + 1] == 1:
                    classified_pixel[x, y + 1] = 255
                if classified_pixel[x, y - 1] == 1:
                    classified_pixel[x, y - 1] = 255
                if classified_pixel[x - 1, y] == 1:
                    classified_pixel[x - 1, y] = 255
                if classified_pixel[x + 1, y] == 1:
                    classified_pixel[x + 1, y] = 255
                if classified_pixel[x - 1, y + 1] == 1:
                    classified_pixel[x - 1, y + 1] = 255
                if classified_pixel[x + 1, y + 1] == 1:
                    classified_pixel[x + 1, y + 1] = 255
                if classified_pixel[x + 1, y - 1] == 1:
                    classified_pixel[x + 1, y - 1] = 255
                if classified_pixel[x - 1, y - 1] == 1:
                    classified_pixel[x - 1, y - 1] = 255

    hysteris_img = np.zeros(shape=(h, w))
    for m in range(1, pad_h - 1):
        for n in range(1, pad_w - 1):
            hysteris_img[m - 1, n - 1] = classified_pixel[m, n]
    return hysteris_img

def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)
    return result


if __name__ == '__main__':
    img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
    canny(img)