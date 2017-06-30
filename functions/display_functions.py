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


def Display_multigrid(reco, ratio, roi_min, roi_max, fine_cell, **kwargs):
    '''
    Explanation of each parameters
    
    Use keyword arguments to give users freedom on which axis should be visualized
    as well as the relative positon of the visualized slice
    
    reco: reconstruction (reco[0]: coarse, reco[1]: fine)
    ratio: ratio in discretization between fine/coarse
    roi_min: physical coord of fine ROI (min point)
    roi_max: physical coord of fine ROI (max point)
    fine_cell: cell size in fine grid
    
    Optional keyword arguments:
    axis: Can be choosen among 'X', 'Y', 'Z' 
    pos: input the relative position of the slice that users want to visualize
    clim: select the intensity range in the displayed image
    '''
    
    coarse_image = reco[0].asarray()
    roi = reco[1].asarray()
    c_l, c_w, c_h = coarse_image.shape
    f_l, f_w, f_h = roi.shape
    ratio = np.int(ratio)
    length_l = fine_cell * ratio * c_l
    length_w = fine_cell * ratio * c_w
    length_h = fine_cell * ratio * c_h
    
    axis = kwargs.pop('axis', None)
    pos = kwargs.pop('pos', None)
    clim = kwargs.pop('clim', None)
    if pos is not None:
        if axis is None or axis == 'Z':
            if pos >= fine_cell*ratio*c_h/2 or pos <= -fine_cell*ratio*c_h/2:
                raise ValueError('Selected slice out of reconstruction range')
        elif axis == 'Y':
            if pos >= fine_cell*ratio*c_w/2 or pos <= -fine_cell*ratio*c_w/2:
                raise ValueError('Selected slice out of reconstruction range')
        elif axis == 'X':
            if pos >= fine_cell*ratio*c_l/2 or pos <= -fine_cell*ratio*c_l/2:
                raise ValueError('Selected slice out of reconstruction range')
       
    crossline = kwargs.pop('crossline', None)
    
    # Catch unexpected keyword arguments
    if kwargs:
        raise TypeError('unexpected keyword argument: {}'.format(kwargs))
    
    roi_ori = (np.array(roi_min) + np.array(roi_max))/2
    roi_ori_index = np.array([c_l*ratio/2, c_w*ratio/2, c_h*ratio/2]) + (roi_ori)/(fine_cell)
    
    # Define slice axis (if such keyword arg is given)
    if axis is None or axis == 'Z':
        half_c = np.int(c_h/2)
        half_f = np.int(f_h/2)
        if pos is not None:
            half_c += np.int(pos/(fine_cell*ratio))
            half_f += np.int(pos/fine_cell)
        coarse = coarse_image[:,:,half_c]
        coarse_ext_image = ExpandMatrix(coarse, [ratio, ratio])
        # 'If' statement below serves to determine if ROI is included in the displayed image 
        if half_f >= 0 and half_f < f_h:
            fine = roi[:,:,half_f].reshape([f_l, f_w])
            # Indices of ROI on the expanded coarse image
            index_x_min = np.int(roi_ori_index[0]) - np.int(f_l/2)
            index_x_max = np.int(roi_ori_index[0]) + np.int(f_l/2)
            index_y_min = np.int(roi_ori_index[1]) - np.int(f_w/2)
            index_y_max = np.int(roi_ori_index[1]) + np.int(f_w/2)
            # Discussion below focuses on the size of ROI and coarse reconstruction space. It may sound 
            # strightforward that the coarse reconstruction space will always be larger than ROI. However,
            # Due to approximation on discretization grid, sometimes ROI will cover a space larger than 
            # the coarse reconstruction space
            # Most common case: Coarse reconstruction space is larger than ROI
            if c_l*ratio >= f_l and c_w*ratio >= f_w:
                coarse_ext_image[index_x_min:index_x_max,index_y_min:index_y_max] = fine
                fine_space = odl.uniform_discr([-length_l/2,-length_w/2], [length_l/2, length_w/2], 
                                               [ratio*c_l, ratio*c_w])
            
            # Edge and corner 1: ROI is larger than coarse reconstruction space along X direction
            if c_l*ratio <= f_l and c_w*ratio >= f_w:
                index_x_min = np.int(f_l/2 - c_l*ratio/2)
                index_x_max = np.int(f_l/2 + c_l*ratio/2)
                coarse_ext_image[:, index_y_min:index_y_max] = fine[index_x_min:index_x_max, :]
                fine_space = odl.uniform_discr([roi_min[0],-length_w/2], [roi_max[0],length_w/2], 
                                               [index_x_max-index_x_min,ratio*c_w])

            # Edge and corner 2: ROI is larger than coarse reconstruction space along Y direction
            if c_w*ratio <= f_w and c_l*ratio >= f_l:
                index_y_min = np.int(f_w/2 - c_w*ratio/2)
                index_y_max = np.int(f_w/2 + c_w*ratio/2)
                coarse_ext_image[index_x_min:index_x_max,:] = fine[:, index_y_min:index_y_max]
                fine_space = odl.uniform_discr([-length_l/2, roi_min[1]], [length_l/2, roi_max[1]], 
                                               [ratio*c_l, index_y_max-index_y_min])
             
            # Edge and corner 3: ROI is larger than coarse reconstruction space along X and Y direction
            if c_h*ratio <= f_h and c_w*ratio <= f_w:
                index_x_min = np.int(f_l/2 - c_l*ratio/2)
                index_x_max = np.int(f_l/2 + c_l*ratio/2)
                index_y_min = np.int(f_w/2 - c_w*ratio/2)
                index_y_max = np.int(f_w/2 + c_w*ratio/2)
                coarse_ext_image = fine[index_x_min:index_x_max,index_y_min:index_y_max]
                fine_space = odl.uniform_discr([roi_min[0], roi_min[1]], [roi_max[0], roi_max[1]], 
                                               [index_x_max-index_x_min, index_y_max-index_y_min])
                
        # Code below will be executed when there is no need to add ROI to the displayed image
        else:
            fine_space = odl.uniform_discr([-length_l/2,-length_w/2], [length_l/2, length_w/2], 
                                           [ratio*c_l, ratio*c_w])
    
    if axis is not None and axis != 'Z':
        if axis == 'X':
            half_c = np.int(c_l/2)
            half_f = np.int(f_l/2)
            if pos is not None:
                half_c += np.int(pos/(fine_cell*ratio))
                half_f += np.int(pos/fine_cell)
            coarse = coarse_image[half_c,:,:].reshape([c_w, c_h])
            coarse_ext_image = ExpandMatrix(coarse, [ratio, ratio])
            # 'If' statement below serves to determine if ROI is included in the displayed image
            if half_f >=0 and half_f < f_l:
                fine = roi[half_f,:,:].reshape([f_w, f_h])
                # Indices of ROI on the expanded coarse image
                index_x_min = np.int(roi_ori_index[1]) - np.int(f_w/2)
                index_x_max = np.int(roi_ori_index[1]) + np.int(f_w/2)
                index_y_min = np.int(roi_ori_index[2]) - np.int(f_h/2)
                index_y_max = np.int(roi_ori_index[2]) + np.int(f_h/2)
                # Please refers to line 95-98 for purpose of this discussion
                # Most common case: Coarse reconstruction space is larger than ROI
                if c_h*ratio >= f_h and c_w*ratio >= f_w:
                    coarse_ext_image[index_x_min:index_x_max,index_y_min:index_y_max] = fine
                    fine_space = odl.uniform_discr([-length_w/2,-length_h/2], [length_w/2, length_h/2], 
                                                   [ratio*c_w, ratio*c_h])
            
                # Edge and corner 1: ROI is larger than coarse reconstruction space along Z direction
                if c_h*ratio <= f_h and c_w*ratio >= f_w:
                    index_y_min = np.int(f_h/2 - c_h*ratio/2)
                    index_y_max = np.int(f_h/2 + c_h*ratio/2)
                    coarse_ext_image[index_x_min:index_x_max,:] = fine[:,index_y_min:index_y_max]
                    fine_space = odl.uniform_discr([-length_w/2,roi_min[2]], [length_w/2, roi_max[2]], 
                                                   [ratio*c_w, index_y_max-index_y_min])

                # Edge and corner 2: ROI is larger than coarse reconstruction space along Y direction
                if c_w*ratio <= f_w and c_h*ratio >= f_h:
                    index_x_min = np.int(f_w/2 - c_w*ratio/2)
                    index_x_max = np.int(f_w/2 + c_w*ratio/2)
                    coarse_ext_image[:,index_y_min:index_y_max] = fine[index_x_min:index_x_max,:]
                    fine_space = odl.uniform_discr([roi_min[1], -length_h/2], [roi_max[1], length_h/2], 
                                                   [index_x_max-index_x_min, ratio*c_h])
                
                # Edge and corner 3: ROI is larger than coarse reconstruction space along Y and Z direction
                if c_h*ratio <= f_h and c_w*ratio <= f_w:
                    index_y_min = np.int(f_h/2 - c_h*ratio/2)
                    index_y_max = np.int(f_h/2 + c_h*ratio/2)
                    index_x_min = np.int(f_w/2 - c_w*ratio/2)
                    index_x_max = np.int(f_w/2 + c_w*ratio/2)
                    coarse_ext_image = fine[index_x_min:index_x_max,index_y_min:index_y_max]
                    fine_space = odl.uniform_discr([roi_min[1], roi_min[2]], [roi_max[1], roi_max[2]], 
                                                   [index_x_max-index_x_min, index_y_max-index_y_min])
            # Code below will be executed when there is no need to add ROI to the displayed image
            else:
                fine_space = odl.uniform_discr([-length_w/2,-length_h/2], [length_w/2, length_h/2], 
                                               [ratio*c_w, ratio*c_h])
                
            
        if axis == 'Y':
            half_c = np.int(c_w/2)
            half_f = np.int(f_w/2)
            if pos is not None:
                half_c += np.int(pos/(fine_cell*ratio))
                half_f += np.int(pos/fine_cell)
            coarse = coarse_image[:,half_c,:].reshape([c_l, c_h])
            coarse_ext_image = ExpandMatrix(coarse, [ratio, ratio])
            # 'If' statement below serves to determine if ROI is included in the displayed image 
            if half_f >= 0 and half_f < f_w:
                fine = roi[:,half_f,:].reshape([f_l, f_h])
                # Indices of ROI on the expanded coarse image
                index_x_min = np.int(roi_ori_index[0]) - np.int(f_l/2)
                index_x_max = np.int(roi_ori_index[0]) + np.int(f_l/2)
                index_y_min = np.int(roi_ori_index[2]) - np.int(f_h/2)
                index_y_max = np.int(roi_ori_index[2]) + np.int(f_h/2)
                # Please refers to line 95-98 for purpose of this discussion
                # Most common case: Coarse reconstruction space is larger than ROI
                if c_l*ratio >= f_l and c_h*ratio >= f_h:
                    coarse_ext_image[index_x_min:index_x_max,index_y_min:index_y_max] = fine
                    fine_space = odl.uniform_discr([-length_l/2,-length_h/2], [length_l/2, length_h/2], 
                                                   [ratio*c_l, ratio*c_h])
            
                # Edge and corner 1: ROI is larger than coarse reconstruction space along X direction
                if c_l*ratio <= f_l and c_h*ratio >= f_h:
                    index_x_min = np.int(f_l/2 - c_l*ratio/2)
                    index_x_max = np.int(f_l/2 + c_l*ratio/2)
                    coarse_ext_image[:,index_y_min:index_y_max] = fine[index_x_min:index_x_max, :]
                    fine_space = odl.uniform_discr([roi_min[0], -length_h/2], [roi_max[0],length_h/2], 
                                                   [index_x_max-index_x_min, ratio*c_h])

                # Edge and corner 2: ROI is larger than coarse reconstruction space along Z direction
                if c_h*ratio <= f_h and c_l*ratio >= f_l:
                    index_y_min = np.int(f_h/2 - c_h*ratio/2)
                    index_y_max = np.int(f_h/2 + c_h*ratio/2)
                    coarse_ext_image[index_x_min:index_x_max, :] = fine[:, index_y_min:index_y_max]
                    fine_space = odl.uniform_discr([-length_l/2, roi_min[2]], [length_l/2, roi_max[2]], 
                                                   [ratio*c_l, index_y_max-index_y_min])
                
                # Edge and corner 3: ROI is larger than coarse reconstruction space along X and Z direction
                if c_h*ratio <= f_h and c_l*ratio <= f_l:
                    index_y_min = np.int(f_h/2 - c_h*ratio/2)
                    index_y_max = np.int(f_h/2 + c_h*ratio/2)
                    index_x_min = np.int(f_l/2 - c_l*ratio/2)
                    index_x_max = np.int(f_l/2 + c_l*ratio/2)
                    coarse_ext_image = fine[index_x_min:index_x_max,index_y_min:index_y_max]
                    fine_space = odl.uniform_discr([roi_min[0], roi_min[2]], [roi_max[0], roi_max[2]], 
                                                   [index_x_max-index_x_min, index_y_max-index_y_min])
            
            # Code below will be executed when there is no need to add ROI to the displayed image
            else:
                fine_space = odl.uniform_discr([-length_l/2,-length_h/2], [length_l/2, length_h/2], 
                                               [ratio*c_l, ratio*c_h])
            
            
    if crossline is not None:
        plt.figure(1)
        if axis != 'X':
            plt.plot(coarse_ext_image[roi_ori_index[0],:])
        else:
            plt.plot(coarse_ext_image[roi_ori_index[1],:])
    
    recon = fine_space.element(coarse_ext_image)
    if clim is not None:
        recon.show(clim = clim)
    else:
        recon.show()
    