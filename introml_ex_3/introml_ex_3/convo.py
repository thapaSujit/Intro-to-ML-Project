from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
from scipy.signal import gaussian


def make_gauss_kernel(ksize, sigma):
    center = (int)(ksize / 2)
    kernel = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = (1 / (2 * np.pi * sigma)) * np.exp(-(diff ** 2) / (2 * sigma ** 2))
        # implement the Gaussian kernel here
    return kernel / np.sum(kernel)


def slow_convolve(arr, k):
    newImage = np.zeros(arr.shape)
    flipKernel = np.fliplr(np.flipud(k))
    capU = k.shape[0]
    capV = k.shape[1]
    padImage = np.pad(arr, ((capU // 2, capU // 2), (capV // 2, capV // 2)), 'constant')
    row = arr.shape[0]
    column = arr.shape[1]
    sum = 0

    for i in range(row):
        for j in range(column):
            sum = 0

            for u in range(0, capU):
                for v in range(0, capV):
                    sum += flipKernel[u, v] * padImage[i + u, j + v]

            newImage[i, j] = sum

    return newImage

if __name__ == '__main__':
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    im = np.array(Image.open('introml_ex_3\introml_ex_3\input1.jpg'))
    k = make_gauss_kernel(3, 2)  # todo: find better parameters
    r = im[:, :, 0]
    g = im[:, :, 1]
    b = im[:, :, 2]
    new_r = slow_convolve(r, k)
    new_g = slow_convolve(g, k)
    new_b = slow_convolve(b, k)
    rgb = np.dstack((new_r, new_g, new_b))
    result = im + (im - rgb)
    clip_arr = result.clip(0, 255)
    result_arr = clip_arr.astype(np.uint8)
    im = Image.fromarray(result_arr)
    im.save("im1_new.jpeg")

    # TODO: chose the image you prefer

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
