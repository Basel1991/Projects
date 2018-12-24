# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:07:33 2018

@author: Basel Alyafi
"""
import numpy as np


def align(fixed_shape, mov_shape):
    """
    This function finds the scale and rotation needed to align two
    shapes.
    (s, theta) = align(fixed_shape, mov_shape)

    Parameters
    ----------
    fixed_shape : 2n integer vector (the first n values are the x coordinates, then the n y coordinates)
        The fixed shape

    mov_shape : 2n integer vector (the first n values are the x coordinates, then the n y coordinates)
        The moving shape

    Returns
    -------
    (aligned_shape, param) : float tuple
        The aligned shape,
        param: dictionary of {s, theta, Tx, Ty}
        where: s is the scale, theta is the rotation angle, Tx is translation over
        x axis, Ty is the translation over y axis.


    """

    #### centralizing shapes
    centralized_fixed = centralize(fixed_shape)
    centralized_mov = centralize(mov_shape)

    #### extracting x and y coordinates
    x_fixed = centralized_fixed[:int(len(centralized_fixed)/2)]  # was mov_shape
    y_fixed = centralized_fixed[int(len(centralized_fixed)/2):]   # was mov_shape

    x_mov = centralized_mov[:int(len(centralized_mov)/2)]
    y_mov = centralized_mov[int(len(centralized_mov)/2):]

    ### computing transformation parameters
    mov_norm = np.linalg.norm(centralized_mov)

    a = np.dot(np.squeeze(centralized_fixed), np.squeeze(centralized_mov))/np.square(mov_norm)

    b = np.sum(x_mov * y_fixed - x_fixed * y_mov) / np.square(mov_norm)

    s = np.sqrt(np.square(a) + np.square(b))

    theta = np.arctan(b/a)

    ### translation computation
    x_fixed = fixed_shape[:int(len(fixed_shape)/2)]
    y_fixed = fixed_shape[int(len(fixed_shape)/2):]

    x_mov = mov_shape[:int(len(mov_shape)/2)]
    y_mov = mov_shape[int(len(mov_shape)/2):]

    tx = np.mean(x_fixed - x_mov)
    ty = np.mean(y_fixed - y_mov)

    aligned_shape = transform(mov_shape, scale=s, theta=theta, translate_x=tx, translate_y=ty)
    param = dict()
    param["scale"] = s
    param["theta"] = theta
    param["translate_x"] = tx
    param["translate_y"] = ty
    return aligned_shape, param


def transform(data_points, scale, theta, translate_x=0, translate_y=0):
    """
    This function applies the transformation specified on data points.
    transformedData = Transform(dataPoints, scale, theta)

    Parameters
    ----------
    data_points : 2n integer vector (the first n values are the x coordinates,then the n y coordinates).
        The shape to be transformed

    scale : a float number
        The scaling factor

    theta : a float number
        The rotation angle

    translate_x : integer, default=0
        Translation over x axis

    translate_y : integer, default=0
        Translation over y axis

    Returns
    -------
    transformed_data : 2n integer vector
        The result of applying the transformation over the dataPoints.


    """
    x_ = np.squeeze(data_points[0:int(len(data_points)/2)])
    y_ = np.squeeze(data_points[int(len(data_points)/2):])

    T = np.array([[scale*np.cos(theta), -scale*np.sin(theta) ],[scale*np.sin(theta), scale*np.cos(theta)]]);

    ### scale and rotation first
    transformed_data = np.matmul(T, np.vstack([x_, y_]))

    ### translation comes now
    transformed_data = np.add(transformed_data, np.vstack([translate_x, translate_y]))

    return np.reshape(transformed_data, (-1, 1))


def centralize(shape):

    """
     this function aligns the centroid of the shape at (0,0).
        shifted_shape = normalize(shape)
    Parameters
    ----------
    shape : a 2n vector

    Returns
    -------
    shifted_shape : the shifted to the reference point (0,0)
    """
    x = shape[:int(len(shape)/2)]

    y = shape[int(len(shape)/2):]

    ### getting both shapes at the same centroid
    x_mean = np.mean(x)

    y_mean = np.mean(y)

    shifted_x = x - x_mean

    shifted_y = y - y_mean

    shifted_shape = np.concatenate(([shifted_x, shifted_y]));

    return shifted_shape