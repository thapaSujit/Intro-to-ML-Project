import numpy as np
#
# NO OTHER IMPORTS ALLOWED
#

def create_greyscale_histogram(img):
    '''
    returns a histogram of the given image
    :param img: 2D image in greyscale [0, 255]
    :return: np.ndarray (256,) with absolute counts for each possible pixel value
    '''
    # TODO
    img_flattened_array = img.flatten()  #We can use flatten method to get a copy of array collapsed into one dimension  
    histogram = np.zeros(256) 
    ''' An array initialized with all zeros:
    [[ 0.  0.  0.  0.]
    [ 0.  0.  0.  0.]
    [ 0.  0.  0.  0.]]
    # img flatten ko each pixel ma janxa ra print(i) garda kheri pixel number auxa. print(i) ma jada kheri pahila 
    #0 index ma janxa ani tya ko value print garxa. hist[i]+=1 garda np.zeroes gareko hist konumber ma janxa. suppose jaba i ko index 0
    #hunxa yadi tyo index ma img-flatten ma 147 pixel number xa vane hist[i]+=1 garda hist ko tyai number ko pixel ma janxa ani
    #existing value ma 1 add garxa
    '''
    for i in img_flattened_array:
        histogram[i] += 1
    return histogram

def binarize_threshold(img, t):
    '''
    Image binarization is the process of taking a grayscale image and converting it to black-and-white, essentially reducing the 
    information contained within the image from 256 shades of gray to 2: black and white, a binary image
    binarize an image with a given threshold
    :param img: 2D image as ndarray
    :param t: int threshold value
    :return: np.ndarray binarized image with values in {0, 255}
    '''
    # TODO
    img_flattened_array = img.flatten() 
    initial_conv = np.where((img_flattened_array <= t), img_flattened_array, 255) #yadi condition true vayo vane img-flatten-array if not 255
    final_conv = np.where((initial_conv > t), initial_conv, 0)
    image = final_conv.reshape(img.shape)
    return image

def p_helper(hist, theta: int):
    '''
    Compute p0 and p1 using the histogram and the current theta,
    do not take care of border cases in here
    :param hist:
    :param theta: current theta
    :return: p0, p1
    p(x) lai pixel at index x manera p(X) calculate gareko formula ko basis ma. sab p(x) jodeko
    '''
    for i in range(0,255):
        p0 = np.sum(hist[:theta+1])
        p1 = np.sum(hist[theta+1:])
    return p0, p1

def mu_helper(hist, theta, p0, p1):
    '''
    Compute mu0 and m1
    :param hist: histogram
    :param theta: current theta
    :param p0:
    :param p1:
    :return: mu0, mu1
''' 
    mu0_init=0
    mu1_init=0
    if p0 == 0: #RuntimeWarning: invalid value encountered in double_scalars
        p0 = 0.0000001
    for i in range (theta+1):
        mu0_init = (hist[i]*i)+mu0_init
    mu0= mu0_init/p0
    if p1 == 0:
        p1 = 0.0000001
    for i in range (theta+1, len(hist)):
        mu1_init = (hist[i]*i)+mu1_init
    mu1= mu1_init/p1

    return mu0, mu1

def calculate_otsu_threshold(hist):
    '''
    calculates theta according to otsus method

    :param hist: 1D array
    :return: threshold (int)
    For every bin assume that this bin is the threshold for binarization and
    calculate for each threshold:

    '''
    # TODO change the histogram, so that it visualizes the probability distribution of the pixels
    # --> sum(hist) = 1
    
    pd = hist/sum(hist)
    highestClass_variance = 0
    # TODO loop through all possible thetas
    for theta in range(len(pd)):
        # TODO compute p0 and p1 using the helper function
        p0,p1 = p_helper(pd, theta)
        # TODO compute mu and m1 using the helper function
        mu0, mu1 = mu_helper(pd, theta, p0, p1)
        # TODO compute variance
        class_variance = p0*p1*(mu1-mu0)**2
        # TODO update the threshold
        if class_variance > highestClass_variance:
            highestClass_variance=class_variance
            otsu_theta = theta
    return otsu_theta
    
def otsu(img):
    '''
    calculates a binarized image using the otsu method.
    Hint: reuse the other methods
    :param image: grayscale image values in range [0, 255]
    :return: np.ndarray binarized image with values {0, 255}
    '''
    histogram = create_greyscale_histogram(img)
    threshold = calculate_otsu_threshold(histogram)
    binarized_image = binarize_threshold(img, threshold)
    return binarized_image
    
