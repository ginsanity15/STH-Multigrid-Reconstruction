#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 14:17:01 2017

@author: Bo Gao

This function serves to display the multi-grid image
"""

import numpy as np
from matplotlib import pyplot as plt
import odl

def ExpandMatrix(matrix, scale):
    '''
    A function that is capable of expanding the size of input matrix,
    which in our case serves to expand the coarse sinogram
    matrix: Projection data of Cone Beam CT
    scale: A numpy vector contains two elements
    '''
    L1, L2 = matrix.shape
    length = scale[0]*np.ones([1,L1]).reshape([L1]).astype(int)
    width = scale[1]*np.ones([1,L2]).reshape([L2]).astype(int)
    matrix_n = np.repeat(matrix,length,axis=0)
    matrix_n = np.repeat(matrix_n,width,axis=1)
    return matrix_n


def Display_multigrid(coarse_image, roi, fold, roi_min, roi_max, fine_cell, **kwargs):
    '''
    Explanation of each parameters
    
    Use keyword arguments to give users freedom on which axis should be visualized
    as well as the relative positon of the visualized slice
    Options for keyword arguments:
    axis: Can be choosen among 'X', 'Y', 'Z'
    
    pos: input the target position
    '''
    
    c_l, c_w, c_h = coarse_image.shape()
    f_l, f_w, f_h = roi.shape()
    length_l = fine_cell * fold * c_l
    length_w = fine_cell * fold * c_w
    length_h = fine_cell * fold * c_h
    
    axis = kwargs.pop('axis', None)
    pos = kwargs.pop('pos', None)
    if axis is None or axis == 'Z':
        if pos >= fine_cell*fold*c_h/2 or pos <= -fine_cell*fold*c_h/2:
            raise ValueError('Selected slice out of reconstruction range')
    elif axis == 'Y':
        if pos >= fine_cell*fold*c_w/2 or pos <= -fine_cell*fold*c_w/2:
            raise ValueError('Selected slice out of reconstruction range')
    elif axis == 'X':
        if pos >= fine_cell*fold*c_l/2 or pos <= -fine_cell*fold*c_l/2:
            raise ValueError('Selected slice out of reconstruction range')
       
    crossline = kwargs.pop('crossline', None)
    
    
    roi_ori = (np.array(roi_min) + np.array(roi_max))/2
    roi_ori_index = np.array([c_l*fold/2, c_w*fold/2, c_h*fold/2]) + (roi_ori)/(fine_cell)
    
    # I got no better idea than making a discussion here
    if axis is None or axis == 'Z':
        half_c = np.int(c_h/2)
        half_f = np.int(f_h/2)
        if pos is not None:
            half_c += pos/(fine_cell*fold)
            half_f += pos/fine_cell
        coarse = coarse_image[:,:,half_c]
        fine = roi[:,:,half_f]
        coarse_ext_image = ExpandMatrix(coarse, [fold, fold])
        # Indices of ROI on the expanded coarse image
        index_x_min = np.int(roi_ori_index[0]) - np.int(f_l/2)
        index_x_max = np.int(roi_ori_index[0]) + np.int(f_l/2)
        index_y_min = np.int(roi_ori_index[1]) - np.int(f_w/2)
        index_y_max = np.int(roi_ori_index[1]) + np.int(f_w/2)
        coarse_ext_image[index_x_min:index_x_max,index_y_min:index_y_max] = fine
        fine_space = odl.uniform_discr([-length_l,-length_w], [length_l, length_w], 
                                       [fold*c_l, fold*c_w])
    
    if axis is not None and axis != 'Z':
        if axis == 'X':
            half_c = np.int(c_l/2)
            half_f = np.int(f_l/2)
            if pos is not None:
                half_c += pos/(fine_cell*fold)
                half_f += pos/fine_cell
            coarse = coarse_image[half_c,:,:]
            fine = roi[half_f,:,:]
            coarse_ext_image = ExpandMatrix(coarse, [fold, fold])
            # Indices of ROI on the expanded coarse image
            index_x_min = np.int(roi_ori_index[1]) - np.int(f_w/2)
            index_x_max = np.int(roi_ori_index[1]) + np.int(f_w/2)
            index_y_min = np.int(roi_ori_index[2]) - np.int(f_h/2)
            index_y_max = np.int(roi_ori_index[2]) + np.int(f_h/2)
            coarse_ext_image[index_x_min:index_x_max,index_y_min:index_y_max] = fine
            fine_space = odl.uniform_discr([-length_w,-length_h], [length_w, length_h], 
                                           [fold*c_w, fold*c_h])
            
        if axis == 'Y':
            half_c = np.int(c_w/2)
            half_f = np.int(f_w/2)
            if pos is not None:
                half_c += pos/(fine_cell*fold)
                half_f += pos/fine_cell
            coarse = coarse_image[:,half_c,:]
            fine = roi[:,half_f,:]
            coarse_ext_image = ExpandMatrix(coarse, [fold, fold])
            # Indices of ROI on the expanded coarse image
            index_x_min = np.int(roi_ori_index[0]) - np.int(f_l/2)
            index_x_max = np.int(roi_ori_index[0]) + np.int(f_l/2)
            index_y_min = np.int(roi_ori_index[2]) - np.int(f_h/2)
            index_y_max = np.int(roi_ori_index[2]) + np.int(f_h/2)
            coarse_ext_image[index_x_min:index_x_max,index_y_min:index_y_max] = fine
            fine_space = odl.uniform_discr([-length_l,-length_h], [length_l, length_h], 
                                           [fold*c_l, fold*c_h])
            
    if crossline is not None:    
        plt.plot(coarse_ext_image[roi_ori_index[0],:])
    
    recon = fine_space.element(coarse_ext_image)
    recon.show()

# For now, it seems there is no way to display reconstruciton image with as
# many ROIs as we like in one single functions, this is a serious issue
def Display_multigrid_2roi(coarse_image, roi, **kwargs):
    '''
    Explanation here
    '''
    return 'To be defined'
    