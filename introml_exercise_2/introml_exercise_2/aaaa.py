# Implement the histogram equalization in this file
import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
#a
img = cv2.imread('introml_exercise_2\introml_exercise_2\hello.png', cv2.IMREAD_GRAYSCALE)
print(img)
#b
img_flattened = img.flatten() # image from shape=nxd to len=n*d
print(img_flattened)
print(img_flattened.size)
hist = np.zeros(256)
print(hist)

for i in img_flattened: # img flatten ko each pixel ma janxa ra print(i) garda kheri pixel number auxa. print(i) ma jada kheri pahila 
    #0 index ma janxa ani tya ko value print garxa. hist[i]+=1 garda np.zeroes gareko hist konumber ma janxa. suppose jaba i ko index 0
    #hunxa yadi tyo index ma img-flatten ma 147 pixel number xa vane hist[i]+=1 garda hist ko tyai number ko pixel ma janxa ani
    #existing value ma 1 add garxa
    hist[i] += 1
print(hist)
#print(hist.size)

initial_conv = np.where((img_flattened <= 155), img_flattened, 255)
final_conv = np.where((initial_conv > 155), initial_conv, 0)
mage = final_conv.reshape(img.shape)   
print(final_conv)
print(mage)
'''
'''
img_flattened[img_flattened<=155] = 0
img_flattened[img_flattened>155] = 255
image = img_flattened.reshape(img.shape)
print(image)'''
'''p0_list = [hist[i] for i in range(155+1)]
p0 = sum(p0_list) --<219835---215613
p1_list = [hist[i] for i in range(155+1, len(hist))]
p1 = sum(p1_list)
p0 = np.sum(hist[:155+1])
p1 = np.sum(hist[155+1:])
print(p0)
print(p1)

p0 = np.sum(hist[:84+1])
print(p0)

mu0_init=0
mu1_init=1
for i in range (155+1):
    mu0_init = (hist[i]*i)+mu0_init
mu0= 1/10*mu0_init
for j in range (155+1, len(hist)):
    mu1_init = (hist[j]*j)+mu1_init
mu1= 1/11*mu1_init
print(len(hist))
print(mu0)
print(mu1)


p0 = 10
p1 = 11


mu0_list = [i*hist[i] for i in range(155+1)]
mu0 = 1/p0 * sum(mu0_list)
#mo is 3019860.0
#3236644.4545454546

mu1_list = [i*hist[i] for i in range(155+1, len(hist))]
mu1 = 1/p1 * sum(mu1_list)
p = hist/sum(hist)
q=sum(p)
print('mo is', mu0)
print(mu1)
print(q)
print(len(hist))

array = [10,2,153]
for i in array:
    print(i)


import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
Laura = arr.flatten()
print(Laura)
Johanna = arr.size
Sujit = np.zeros(9)

for i in range(Johanna):
    Sujit[i] = Laura[Johanna-1-i]
vincent = Sujit.reshape(arr.shape)
print(vincent)
print(Sujit)
#5
a = 3/5
print(a)
'''
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy
#from scipy.ndimage import convolve
from convo import make_gauss_kernel

def normalize(k):# Helper function to be used in gaussFilter in order to normalize the gauss kernel
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
    A Gaussian Filter is a low pass filter used for reducing noise (high frequency components) and blurring regions of an image. 
    The filter is implemented as an Odd sized Symmetric Kernel (DIP version of a Matrix) which is passed through each pixel of 
    the Region of Interest to get the desired effect. The kernel is not hard towards drastic color changed (edges) due to it the
    pixels towards the center of the kernel having more weightage towards the final value then the periphery.
    In the process of using Gaussian Filter on an image we firstly define the size of the Kernel/Matrix that would be used for 
    demising the image. The sizes are generally odd numbers, i.e. the overall results can be computed on the central pixel. 
    Also the Kernels are symmetric & therefore have the same number of rows and column. The values inside the kernel are 
    computed by the Gaussian function
    Normalization is a good technique to use when you do not know the distribution of your data or when you know the 
    distribution is not Gaussian (a bell curve).
    """
    kernel = make_gauss_kernel(ksize, sigma)
    k_normalized = normalize(kernel)
    convolved= scipy.ndimage.convolve(img_in, k_normalized)
    return (k_normalized, convolved.astype(np.int32))#cast filtered image back to int


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
    return filtered_vert.astype(np.int32), filtered_horiz.astype(np.int32) #cast filtered image back to int


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    mag_gradient = np.sqrt(np.square(gx) + np.square(gy))
    direction_grad = np.arctan2(gy, gx)
    return mag_gradient.astype(np.int32), direction_grad #cast filtered image back to int


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
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima
            elif converted_angle == 45:
                local_maxima = g[i, j]
                if local_maxima >= g[i + 1, j - 1] and local_max >= g[i - 1, j + 1]:
                    local_maxima = g[i, j]
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima
            elif converted_angle == 90:
                local_max = g[i, j]
                if local_maxima >= g[i + 1, j] and local_max >= g[i - 1, j]:
                    local_maxima = g[i, j]
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima
            elif converted_angle == 135:
                local_maxima = g[i, j]
                if local_maxima >= g[i + 1, j + 1] and local_max >= g[i - 1, j - 1]:
                    local_maxima = g[i, j]
                else:
                    local_maxima = 0
                max_suppress[i, j] = local_maxima

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
    categories = np.zeros(shape=(h, w))

    for i in range(0, h):
        for j in range(0, w):
            if max_sup[i, j] <= t_low:
                categories[i, j] = 0
            elif max_sup[i, j] > t_low and max_sup[i, j] <= t_high:
                categories[i, j] = 1
            else:
                categories[i, j] = 2

    categories = np.pad(categories, (1, 1), 'constant', constant_values=(0, 0))#PADDING THE MATRIX WITH ZEROES TO DEAL WITH BORDER PIXELS
    pad_h = categories.shape[0]
    pad_w = categories.shape[1]
    for x in range(1, pad_h - 1):
        for y in range(1, pad_w - 1):
            if categories[x, y] == 2:
                categories[x, y] = 255
                # RIGHT
                if categories[x, y + 1] == 1:
                    categories[x, y + 1] = 255
                # LEFT
                if categories[x, y - 1] == 1:
                    categories[x, y - 1] = 255
                # UP
                if categories[x - 1, y] == 1:
                    categories[x - 1, y] = 255
                # DOWN
                if categories[x + 1, y] == 1:
                    categories[x + 1, y] = 255
                # UPRIGHT
                if categories[x - 1, y + 1] == 1:
                    categories[x - 1, y + 1] = 255
                # DOWNRIGHT
                if categories[x + 1, y + 1] == 1:
                    categories[x + 1, y + 1] = 255
                # DOWNLEFT
                if categories[x + 1, y - 1] == 1:
                    categories[x + 1, y - 1] = 255
                # UPLEFT
                if categories[x - 1, y - 1] == 1:
                    categories[x - 1, y - 1] = 255

    # REMOVE THE ZERO PADDING
    hysteris_img = np.zeros(shape=(h, w))
    for m in range(1, pad_h - 1):
        for n in range(1, pad_w - 1):
            hysteris_img[m - 1, n - 1] = categories[m, n]

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
'''
angle_degree = 200
while angle_degree > 180:
        angle_degree -= 180
        
print(angle_degree)