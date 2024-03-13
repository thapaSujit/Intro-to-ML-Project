'''
Created on 04.10.2016

@author: Daniel Stromer
@modified by Charly, Christian, Max (23.12.2020)
'''
import numpy as np
#import matplotlib.pyplot as plt
# do not import more modules!


def calculate_R_Distance(Rx, Ry):
    '''
    calculate similarities of Ring features
    :param Rx: Ring features of Person X
    :param Ry: Ring features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    lengthOfRx = len(Rx)
    sum = 0
    DistanceHelper = a = np.empty(lengthOfRx, dtype=object)
    for i in range(lengthOfRx):
        DistanceHelper[i] = abs(Rx[i] - Ry[i])
        sum += DistanceHelper[i]
    distance = sum/lengthOfRx
    return distance


def calculate_Theta_Distance(Thetax, Thetay):
    '''
    calculate similarities of Fan features
    :param Thetax: Fan features of Person X
    :param Thetay: Fan features of Person Y
    :return: Similiarity index of the two feature vectors
    '''
    def helper_Formula(theta_set_value):
        l = 0
        sum_of_theta = np.sum(theta_set_value)

        for i in range(theta_set_value.shape[0]):
            l += (theta_set_value[i] - (1 / theta_set_value.shape[0]) * sum_of_theta) ** 2

        return l, sum_of_theta

    l_xx, sum_xx = helper_Formula(Thetax)
    l_yy, sum_yy = helper_Formula(Thetay)

    l_xy = 0
    theta_shape = Thetax.shape[0]
    for j in range(theta_shape):
        l_xy += (Thetax[j] - (1 / theta_shape * sum_xx)) * (Thetay[j] - (1 / theta_shape * sum_yy))

    return (1 - (l_xy ** 2) / (l_xx * l_yy)) * 100