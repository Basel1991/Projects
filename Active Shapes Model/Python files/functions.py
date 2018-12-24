# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 07:17:25 2018

@author: Basel Alyafi
"""
import numpy as np
from numpy import cov
from scipy import ndimage
from scipy.linalg import eig
from skimage.filters import (sobel, gaussian)
from skimage.feature import canny
import matplotlib.pyplot as plt


def principle_comp(dataSet, varianceRatio):
    """
    This function takes a matrix  d X N and applies the Principle Component
    Analysis .The eigen values and eigen vectors of the covariance matrix
    (d X d) are returned so the variance ratio is satisfied.

    Parameters
    ----------
    dataset : a matrix d X N
        N observations, d features for each observation.

    varianceRatio : a value <=1
        the ratio required for the returned sorted eigen values
        to the summation of all eigen values.

    returns
    -------
    (lambdas, eigen_vecs) : lambdas are the highest k real eigen values that
       satisfy the variance ratio required, eigen_vecs are the corresponding
       real eigen vectors.

    """
    variables, observations = np.shape(dataSet);
    covariance_mat = cov(dataSet)
    lambdas, V = eig(covariance_mat)
    idx = np.argsort(np.real(lambdas))
    idx = np.flip(idx)
    sortedLambdas, sortedV = (lambdas[idx], V[:, idx]);

    lambdas = np.real(sortedLambdas[0]);
    eigen_vecs = np.real(sortedV[:, 0]);
    variance = np.real(lambdas)/np.sum(np.real(sortedLambdas));
    i = 1;
    if varianceRatio==1:
        return sortedLambdas, sortedV
    else:
        while variance < varianceRatio:
            lambdas = np.append(lambdas, np.real(sortedLambdas[i]));
            eigen_vecs = np.concatenate((eigen_vecs, np.real(sortedV[:, i])));
            variance = np.sum(np.real(lambdas))/np.sum(np.real(sortedLambdas));
            i += 1;
        eigen_vecs = np.reshape(eigen_vecs,(i, variables) ).T
        return lambdas, eigen_vecs


def img_nearest_edges(img_model, gray_img, norm_range):
    """
    finds the nearest edge along the norm of every point of the model
    img_shape = img_nearest_edges(img_model, img_grad, norm_range)

    Parameters
    ----------
    img_model : 2n integer vector (the first n values are the x coordinates,
        then the n y coordinates)
        the shpae to find image edges around its points.

    gray_img : 2D gray scale image
        the image to find edges inside

    norm_range : integer
        the one-way range along which to look for image edges.

    returns
    -------
    img_shape : 2n integer vector (the first n values are the x coordinates,
        then the n y coordinates)
        the maximum edges (one for each model point) found around image model points

    """
    grad = (sobel(sobel(gaussian(gray_img))))  #gaussian

    num_points = int(len(img_model)/2)
    norms = np.zeros((4*norm_range+2, num_points))
    #### loop over all the points
    for index in range(num_points):
        ### a special case: the first point
        if index == 0:
            differ_x = img_model[1] - img_model[0]
            differ_y = img_model[num_points + 1] - img_model[num_points]
            norm = np.linalg.norm([differ_x, differ_y])
            range_x = (-differ_y / norm) * np.linspace(-norm_range, norm_range, 2 * norm_range + 1)
            range_y = (differ_x / norm) * np.linspace(-norm_range, norm_range, 2 * norm_range + 1)
        #### a second special case: the second point
        elif index == num_points-1:
            differ_x = img_model[index] - img_model[index-1]
            differ_y = img_model[num_points + index] - img_model[num_points + index - 1]
            norm = np.linalg.norm([differ_x, differ_y])
            range_x = (-differ_y / norm) * np.linspace(-norm_range, norm_range, 2 * norm_range + 1)
            range_y = (differ_x / norm) * np.linspace(-norm_range, norm_range, 2 * norm_range + 1)
        #### the ordinary case: a point within the first and last points
        else:
            differ_x = img_model[index + 1] - img_model[index - 1]
            differ_y = img_model[num_points + index + 1] - img_model[num_points + index - 1]
            norm = np.linalg.norm([differ_x, differ_y])
            range_x = (-differ_y / norm) * np.linspace(-norm_range, norm_range, 2 * norm_range + 1)
            range_y = (differ_x / norm) * np.linspace(-norm_range, norm_range, 2 * norm_range + 1)

        norm = np.concatenate((img_model[index] + range_x, img_model[index + num_points] + range_y))
#        norms = np.column_stack((norms ,norm))
        norms[:,index] = norm

    ### getting the integer (x,y) indices
    norms = (np.floor(norms)).astype(np.int)

    norms_x = norms[0:2 * norm_range + 1,:]
    norms_y = norms[2 * norm_range + 1:,:]

    ## to be sure indices are inside the image
    norms_x = np.clip(norms_x, a_min=0, a_max = grad.shape[1]-1)  #was 1
    norms_y = np.clip(norms_y, a_min = 0, a_max = grad.shape[0]-1) # was 0

    ## showing results
    plt.clf()
    plt.imshow(grad, cmap='gray')
    plt.plot(img_model[0:num_points], img_model[num_points:], 'r-')
    plt.title('gradient image')

#    ### getting the maximum gradient over each norm
    edges_indices = np.argmax(grad[norms_y, norms_x], axis=0)
    img_shape = np.concatenate((norms_x[edges_indices, range(num_points)], norms_y[edges_indices, range(num_points)]))
    plt.plot(norms_x[edges_indices, range(num_points)], norms_y[edges_indices, range(num_points)],'co')
#
    plt.plot(norms_x, norms_y, 'y-')
    plt.legend(["mean model", "norms maxima", "norms"])
    plt.pause(.01)

    return img_shape