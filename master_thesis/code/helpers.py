"""
author: Basel Alyafi
year: 2019
Erasmus Mundus in Medical Imaging and Applications 2017-2019
Master Thesis
"""
import argparse
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2
import seaborn
import sklearn.preprocessing as skl_preprocess
import os

import torch
from skimage import io, transform
import warnings
import seaborn as sea
import pandas as pd
import matplotlib.pyplot as plt
import imageio as imio
from torch import nn

from Gans.DCGAN import Generator
warnings.filterwarnings(action='ignore')

def extract_roi(gt_folder, dcm_folder, end_with, roi_folder): #DOC OK
    """
    this function is used to extract region of interest given a gray-scale image and a binary groundtruth.

    Params:
    -------
    gt_folder: string
        the real_path to the ground truth images (binary images with a white rectangle over the region of interest
    dcm_folder: string
        the real_path to the Dicom images
    end_with: string
        the difference at the end of corresponding groundtruth and image files names + extension
        example: image1.dcm and image1_GT.png
        end_with should be in this case '_GT.png'
    roi_folder: string
        the path where the extracted roi files should be saved

    Returns:
    -------
    void
    """

    # read roi files names
    gt_names= glob.glob(gt_folder + '*' + end_with)

    print('number of ground truth images found', len(gt_names))

    # find corresponding images
    for idx, _ in enumerate(gt_names):

        print('playing with', gt_names[idx])

        # remove tha part related to ground truth images names only
        gt_names[idx]= gt_names[idx].rstrip(end_with)

        # extract the last part of the name after the last slash, then find the image recursively (all children folders)
        last_part = gt_names[idx].rsplit(sep='/', maxsplit=1)[1]

        # find the image name
        file_name = glob.glob(dcm_folder +'**/' + last_part + '*.dcm', recursive=True)

        # read the dicom file
        dcm_file = pydicom.dcmread(file_name[-1])

        # extract the matrix (image only without header info)
        dcm_arr = dcm_file.pixel_array

        # create the mask
        thresh_img = dcm_arr > 0

        # show images and corresponding masks
        plt.figure()
        plt.imshow(thresh_img, cmap='gray')
        plt.figure()
        plt.show()
        plt.imshow(dcm_arr, cmap='gray')

        # read the image using open cv to find components
        roi_img = cv2.imread(gt_names[idx] + end_with, cv2.IMREAD_UNCHANGED)

        img_contours, contours = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print('contours shape', np.shape(img_contours), 'contours \n', img_contours)

        cv2.waitKey(0)

        # find the rectangle start and stop (rows and columns )
        col_start   = img_contours[0][0][0][0]
        col_end     = img_contours[0][2][0][0]
        row_start   = img_contours[0][0][0][1]
        row_end     = img_contours[0][2][0][1]

        # extract the patch
        roi_overlay = dcm_arr[row_start:row_end, col_start:col_end]

        # save the roi image
        plt.imsave(roi_folder + last_part  + '_roi.png', roi_overlay, cmap='gray')  #str(idx)
        print('index', idx, last_part + ' saved successfully')


def preprocess(src_path, dst_path, method, end_with='', must_include='', max_dims=None, size=None): #DOC OK
    """
    this function reads a folder of images, processes them, and saves the processed images in the specified real_path.

    Params
    ------

    src_path: string
        the real_path where the images are stored
    dst_path: string
        the real_path where to save the processed images
    method: string
        the preprocessing method, options: '01rescale', 'gray2rgb', 'resize'
    end_with: string
        the last part of images names including the extension
    must_include: string
        any file name must include this to be considered
    max_dims: list or tuple of two values
        the maximum accepted dimensions (hight, width), if an image found to be larger than these dimensions, it will be skipped
    size: tuple
        the output size

    Returns
    ------
    void
    """

    # get files names
    imgs_names = glob.glob(os.path.join(src_path, '*' + end_with), recursive=True)
    shapes = []
    maxes, mins = [], []
    print('total number', len(imgs_names))

    # loop over all images
    for idx, path in enumerate(imgs_names):

        img = io.imread(path) #read image
        shapes.append(np.shape(img))    # append the shape for statistical purposes

        # get the file name only, it should work on Linux ('\') as well as Win systems ('\')
        _, last_part = str.rsplit(path, '/', maxsplit=1) if path.find('/') != -1 else str.rsplit(path, '\\', maxsplit=1)

        if method=='01rescale':

            # min max rescaling for the image intensities
            processed_img = skl_preprocess.minmax_scale(np.ravel(img))

            # reshaping from a vector to the original shape, range [0,255], dtype: uint8
            processed_img = np.reshape(processed_img * 255, np.shape(img))
            processed_img = np.round(processed_img).astype(np.uint8)

            maxes.append(np.amax(np.reshape(img, (-1, 1)))) # was rescaled_
            mins.append(np.amin(np.reshape(img, (-1, 1)))) # was rescaled_

            print('shape', np.shape(processed_img), 'type', type(processed_img[0, 0]), 'max', maxes[-1], 'min', mins[-1], 'idx', idx)

        elif method=='gray2rgb':    #from gray scale to RGB
            processed_img = np.stack([img,img,img], axis=2).astype(np.uint8)

            print('\n', '-'*40 +'shape', np.shape(processed_img), 'type', type(processed_img[0, 0, 0]), 'idx', idx)

        elif method=='resize':  #rescale dimensions to a predefined size
            if size is None:
                raise ValueError('size should be set in order to resize images')
            processed_img = transform.resize(img, output_shape=size, order=1, anti_aliasing=True, preserve_range=True) #resize to the target dimensions
            processed_img = np.round(processed_img).astype(np.uint8) #keep datatype as unisgned integer 8 bits

            print('\n', '-'*40 +'shape', np.shape(processed_img), 'type', type(processed_img[0, 0]), 'idx', idx)

        else:
            raise ValueError('this method is not supported, see function documentation for options please' )

        # create all folders needed if they don't exist
        if not (os.path.exists(dst_path)):
            os.makedirs(dst_path, exist_ok=True)

        if (max_dims is not None): #this is useful when there is a limit where images larger than some size are to be ignored
            if~(((shapes[-1][0] < max_dims[0]) | (shapes[-1][1] < max_dims[1])) & (last_part.find(must_include) != -1)):
                continue

        # save_model the processed image
        io.imsave(os.path.join(dst_path, last_part), processed_img)

        # debugging
        new = io.imread(os.path.join(dst_path, last_part))

        print('value: new, processed, org', new[0, 0], processed_img[0, 0], img[0,0])
        print('type: new, processed, org', type(new[0, 0]), type(processed_img[0, 0]), type(img[0,0]))

        print('max: new, processed, org ', np.max(new), np.max(processed_img), np.max(img))
        print('min: new, processed, org', np.min(new), np.min(processed_img), np.min(img))

# statistics about shape
    rows, cols = 0,0
    for shape in shapes:
        row, col = shape[0], shape[1]
        rows += row
        cols += col
    print('\naverage shape', (rows/len(imgs_names), cols/len(imgs_names)))
    print('average max, min', np.mean(maxes), np.mean(mins))


def get_name(abs_file_path, split_at=''): #DOC OK
    """
    this function is used to return the part of the absolute file name between the folders names and split_at.
    It works on lists of files paths as well.

    Params:
    -------
    abs_file_path: string
        the real_path(s) to the file(s).
    split_at: string
        the part where the file real_path should be split.

    Returns:
    -------
    string, the string(s) between the absolute real_path and the split_at.

    Example:
    >>real_path= 'E:\\user\\Documents\\img1_GT.png'
    >>split_at = '_GT.png'
    >>print(get_name(real_path, split_at))
    img1

    """
    # remove all folders names
    gt_names = list(map(lambda gt_abs_name: os.path.split(gt_abs_name)[1], abs_file_path))

    # take the first part of the split version
    split_gt_names = list(map(lambda split_name: split_name.rsplit(split_at)[0], gt_names))
    return split_gt_names

def run_generator(model, batch_size, save_path, RGB): #DOC OK
    """
    to run a generator to generate images and save them.

    Params
    ------
    model: nn.Module
        the model to run
    batch_size: int
        number of images to generate
    save_path: string
        where to save the generated images
    RGB: bool
        if True images will be saved in RGB format, otherwise grayscale will be used

    Returns
    -------
    void
    """
    # detect if there is a GPU, otherwise use cpu instead
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model input
    fixed_noise = torch.randn(batch_size, model.nz, 1, 1, device=device)
    model.to(device)

    # Testing mode
    mode = model.training
    model.eval()
    model.apply(apply_dropout)

    with torch.no_grad():
        output = model(fixed_noise).detach().cpu()

    #back to original training state
    model.train(mode=mode)

    # create the path if does not exist.
    if not(os.path.exists(save_path)):
        os.makedirs(save_path, exist_ok=True)

    print('Running Generator...')
    for i in range(batch_size):

        #rescale intensities from [-1,1] to [0,1]
        img = np.transpose(output[i], [1, 2, 0]) / 2 + 0.5
        # img = np.squeeze(img)
        img = np.array(255 * img).round().astype(np.uint8)

        # if one channel, squeeze to 2d
        if img.shape[2]==1:
            img = img.squeeze(axis=2)

        # if gray but RGB required
        if RGB and len(img.shape)==2:
            img = np.stack([img,img,img], axis=-1)

        #save the image
        io.imsave(save_path + '{}.png'.format(i), img)
    print('Finished Generating Images')

def apply_dropout(layer):#DOC OK
    """
    This function is used to activate dropout layers during training

    Params:
    -------
    layer: torch.nn.Module
        the layer for which the dropout to be activated

    Returns:
    --------
    void
    """
    classname = layer.__class__.__name__
    if classname.find('Dropout') != -1:
        layer.train()

def wasserstein_loss(D_fake, D_real=None, is_D=True):#DOC OK
    """
    This function is to calculate the Wasserstein loss for training GANs
    Params:
    -------

    D_fake: torch.Tensor
         discriminator output for fake batch
    D_real: torch.Tensor
        discriminator output for real batch
    is_D: boolean
        if true, this function returns D loss, otherwise, it returns G loss
    Returns:
        w_loss: torch.Tensor
            either Wasserstein D_loss or Wasserstein G_loss


    """
    if is_D:
        loss = -(torch.mean(D_real) - torch.mean(D_fake))
    else:
        loss = -torch.mean(D_fake)
    return loss


def args_parser(**names_types):#DOC OK
    """
    This function is to enable the user to enter the parameters in the console easily and in any order.

    Params:
    ------
    variable_name = variable_type
        the key arguments should be passed as mentioned above, first the variable name then the variable type.
    Returns:
    --------
    parser, the parser of all arguments.

    Example:
    --------
    parser = args_parser(name=str, age=int)
    args = parser.parse_args()
    print( args.name, args.age)

    -- on Console
    python3 script.py --name 'Basel' --age 28
    >> output:
    Basel 28
    """
    parser = argparse.ArgumentParser()

    for var_name, var_type in zip(names_types.keys(), names_types.values()):
        parser.add_argument('--'+var_name, type=var_type)

    return parser

def plot_fig(data, title, label, x_label, y_label, save_path=None, axis='ON'):
    """
    This function is used to plot figures.

    Params:
    ------
    data: list
        the list of values to plot
    title: str
        figure title
    label: str
        the label of the plot for legend uses
    x_label: str
        the x-axis label
    y_label: str
        the y-axis label
    save_path: str
        the path where to save the figure.
    axis: bool
        if true, axis will be shown.

    Returns:
    -------
    void
    """

    plt.plot(data, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis(axis)

    if save_path:
        plt.savefig(save_path)
