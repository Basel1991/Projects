# -*- coding: utf-8 -*-
"""
Spyder Editor

This script is to implement Active Shapes Models algorithm from scratch

Created on Dun Nov  11 17:17:25 2018

@author: Basel Alyafi


"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from align import (align, transform)

from functions import (principle_comp, img_nearest_edges)
from initialization import DynamicShape
import sys


#reading the normalized and aligned shapes
data_set = np.loadtxt('PreProcessed.txt', delimiter=',');
(variables, observations) = np.shape(data_set)

# the initial state of the initialization
norm_range  = 18
max_iter    = 140
image_name  = "hand/images/0005.jpg"
scale       = 1000;
theta       = -0.1582
translate_x = 316
translate_y = 361

# error = np.linalg.norm(Y-transform(x))


num_points = int(variables / 2);
x = data_set[0:int(variables / 2), :]

y = data_set[int(variables / 2):, :]

lambdas, eigen_vec = principle_comp(data_set, .99)

im0 = plt.imread(image_name)
mean_shape = np.mean(data_set, axis=1)

### RGB to gray v
gray_img = rgb2gray(im0);

fig, ax = plt.subplots()

ax.imshow(gray_img, cmap='gray'), ax.set_title('Initializing image ' + image_name[-8:-4]), ax.set_xlabel('Left drag for translation, wheel for scale, horizontal right drag for angle', fontdict= dict(color='red', fontsize='14'))
plot, = ax.plot(mean_shape[0:num_points], mean_shape[num_points:], 'r-', label='mean model')

# initialization
dynamic_hand = DynamicShape(mean_shape, fig, plot, dict(scale=scale, theta=theta, translate_x=translate_x,
                                                        translate_y=translate_y))
dynamic_hand.connect()
plt.show()

param={}
scale, theta, translate_x, translate_y = dynamic_hand.get_parameters()
param["scale"], param["theta"], param["translate_x"], param["translate_y"] = dynamic_hand.get_parameters()

# pkill -f 'active-shapes'

count = 0
b = np.zeros(len(lambdas))

while count < max_iter:
    x = mean_shape + np.matmul(eigen_vec, b)

    # align the model with the image
    # improved on 19-Nov-2018 at 08.00
    # X = transform(x, scale, theta, translate_x, translate_y)
    X = transform(x, param["scale"], param["theta"], param["translate_x"], param["translate_y"])

    # find new image points
    Y = img_nearest_edges(X, gray_img, norm_range)
    plt.title('iteration' + str(count))

    # align the model with the new found image points
    _, param = align(Y, x)

    # bring image points to model space
    y = transform(Y, 1 / param["scale"], -param["theta"], -param["translate_x"], -param["translate_y"])

    # it  is not recommended to use the projection, it was tested and results were not improved
    # y= y/np.dot(np.squeeze(y), np.squeeze(mean_shape))
    # calculate the new parameters (deformation)
    b = np.matmul(eigen_vec.T, np.squeeze(y) - np.squeeze(mean_shape))

    count = count + 1;
    print('iter', count)

    # put some constraints on b
    for i in range(len(b)):
        if b[i] > 3 * np.sqrt(lambdas[i]):
            b[i] = 3 * np.sqrt(lambdas[i])
            print('positive excess')
        elif b[i] < -3 * np.sqrt(lambdas[i]):
            b[i] = -3 * np.sqrt(lambdas[i])
            print('negative excess')


print('intialization: scale %1.4f, theta %1.4f, translate_x %1.4f, translate_y %1.4f'%(scale, theta, translate_x, translate_y))

plt.figure()
plt.imshow(im0, cmap='gray'), plt.title('Result image' + image_name[-8:-4])
plt.plot(X[0:num_points], X[num_points:], 'g-', label='mean model', linewidth=2)
plt.text(30,30,'scale = %d \t theta = %1.3f \n translate_x = %1.3f \t translate_y = %1.3f'%(param["scale"], param["theta"], param["translate_x"], param["translate_y"]))
plt.show()
