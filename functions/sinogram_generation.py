#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:54:24 2017

@author: Bo Gao
"""

import glob
import os
import numpy as np
import dicom

def sino_gene(DICOM_path, **kwargs):
    '''
    This function serves to generate sinogram for multi-grid reconstruction
    Basically, it just combine projection images stored in Open CT format into
    a 3D matrix. 
    Based on the optional input key argument, it can also truncate the input
    projection dataset that is not corresponding to the reconstruction space 
    with the hope of minimizing the burden on memory.
    
    Parameters
    --------------------------
    DICOM_path: the directory where projection images at each angle are stored,
    in open CT format
    
    Keyword Arguments
    --------------------------
    roi_min, roi_max: 
    To better understand the notion of roi_min and roi_max, please refer to 
    figure given below
    
                        ----------------------------* <-- here is max_pt
                        -                           -
                        -                           -
                        -                           -
                        -      RECONSTRUCTION       -
                        -           SPACE           -
                        -           (ROI)           -
                        -                           -
                        -                           -
                        -                           -
                        -                           -
    here is min_pt -->  *----------------------------
    
    LightField:
    
    Log: After log calibration, information stored in img is attenuation coefficient
    '''
    
    # A flag to check if addition input parameters are enough to perform projection
    # image truncating
    flag = 0
    
    # Read in if the user would like to perform projection image truncating
    roi_min = kwargs.pop('roi_min', None)
    if roi_min is not None:
        flag += 1
        a = np.array(roi_min)
           
    roi_max = kwargs.pop('roi_max', None)
    if roi_max is not None:
        flag += 1
        a1 = np.array(roi_max)
    
    # These two variables need to be given in order to perform projection image processing
    LightFieldPath = kwargs.pop('LightFieldPath', None)
    if LightFieldPath is not None:
        LF_ds = dicom.read_file(LightFieldPath)
        LightField = LF_ds.pixel_array
    Log = kwargs.pop('Log', None)
    
    # Check if the reconstruction space is centered in the origin point
    if roi_max[0] + roi_min[0] != 0 or roi_max[1] + roi_min[1] != 0 \
    or roi_max[0] + roi_min[0] != 0:
        raise ValueError('Expect the reconstruction space to be centered at (0,0,0), \
                         got{}'.format((a+a1)/2))
    
    # No checking procedure for these three variables, as I believe no one is dumb
    # enough to input a matrix when they are supposed to input a constant
    if (flag != 0 and flag != 2):
        raise ValueError('Detect the attempt to perform projection image truncating, \
                         while the input paramter is not enough, got {}'.format(flag) + \
                         ' while expect 2')
    
    # TODO: Is there another way to read arbitrary file from one folder in Python?
    # Read in one DICOM file (arbitrarily), and read in the number of pixels included
    # in one horizontal and vertical line across the detector
    i = 0
    for filename in glob.glob(os.path.join(DICOM_path, '*.dcm')):
        if(i == 0):
            ds = dicom.read_file(filename)
        i += 1

    # Calculate the range on projection image that correspond to the region
    # that users want to reconstruct
    L1 = np.int(ds.Rows/2)
    L2 = np.int(ds.Columns/2)
    center_1 = L1
    center_2 = L2
    src = ds.DistanceSourceToDetector
    det = ds.DistanceSourceToPatient 
    l1 = ds.ReconstructionPixelSpacing[0]
    l2 = ds.ReconstructionPixelSpacing[1]
    
    ProjectionGeometry = ds.TypeofProjectionGeometry
    number = ds.NumberOfFramesInRotation
    
    if (flag == 2):
        # When flag equals 2, truncating needs to be done in projection image, 
        # however, truncation will vary with the type of projection geometry
        if ProjectionGeometry == 'CONEBEAM':
            # Calculate the range of reconstruction space at extreme cases
            L1_90_1 = (src+det)/(src - roi_max[0]) * roi_max[1] * 1/l1 + 5
            L1_90_2 = (src+det)/(src - roi_max[1]) * roi_max[0] * 1/l1 + 5
            
            # To Ensure we can cover the projection corresponds to the targeted
            # reconstruction space (as well as ease the mathematical computation)
            # we find the projection that corresponds to a square
            Len = max(roi_max[0], roi_max[1])
            L1_45 = (src+det)/(src) * np.sqrt(2) * Len * 1/l1 + 5
            
            L1 = max(L1_45, L1_90_1, L1_90_2)
            #L1 = (src+det)/(src - (roi_max[0]**2+roi_max[1]**2)/src) * np.sqrt(\
            #     roi_max[0]**2+roi_max[1]**2 - (roi_max[0]**2+roi_max[1]**2)**2/src**2) \
            #     * 1/l1 + 5
            
            L2 = (src+det)/(src-np.sqrt(roi_max[0]**2+roi_max[1]**2))*roi_max[2]*1/l2 + 5
            # Ensure the truncated region will not be larger than the collected projection image
            L1 = np.int(np.clip(L1,1,ds.Columns/2))
            L2 = np.int(np.clip(L2,1,ds.Rows/2))
            
            sinogram = np.zeros([number, 2*L1, 2*L2])
            # I would like to use this, but this command cannot guarantee to read in files in the right order
            #for filename in glob.glob(os.path.join(DICOM_path, '*.dcm')):
            for i in range(number):
                filename = DICOM_path + '/projection_image' + str(i+1) + '.dcm'
                ds = dicom.read_file(filename)
                img = ds.pixel_array
                if LightFieldPath is not None:
                    img = img/LightField
                    img[np.isnan(img)] = 1
                    img[np.isinf(img)] = 1
                if Log is not None:
                    img = -np.log(img)
                    img[img<0] = 0
                
                sinogram[i,:,:] = img[center_1-L1:center_1+L1, center_2-L2:center_2+L2]
                print(str(i) + ' projection images have been included')
    
        if ProjectionGeometry == 'FANBEAM':
            # Calculate the range of reconstruction space at extreme cases
            L1_90_1 = (src+det)/(src - roi_max[0]) * roi_max[1]
            L1_90_2 = (src+det)/(src - roi_max[1]) * roi_max[0]
            
            # To Ensure we can cover the projection corresponds to the targeted
            # reconstruction space (as well as ease the mathematical computation)
            # we find the projection that corresponds to a square
            Len = max(roi_max[0], roi_max[1])
            L1_45 = (src+det)/(src) * np.sqrt(2) * Len * 1/l1 + 5
            
            L1 = max(L1_45, L1_90_1, L1_90_2)
            L1 = np.int(np.clip(L1,1,ds.Columns/2))
            # L1 = ((src+det)/src) * np.sqrt(roi_max[0]**2 + roi_max[1]**2)/l1
            sinogram = np.zeros([number, 2*L1])
            #for filename in glob.glob(os.path.join(DICOM_path, '*.dcm')):
            for i in range(number):
                filename = DICOM_path + '/projection_image' + str(i+1) + '.dcm'
                ds = dicom.read_file(filename)
                img = ds.pixel_array
                if LightFieldPath is not None:
                    img = img/LightField
                    img[np.isnan(img)] = 1
                    img[np.isinf(img)] = 1
                if Log is not None:
                    img = -np.log(img)
                    img[img<0] = 0
                sinogram[i,:] = img[center_1-L1:center_1+L1]
        
        if ProjectionGeometry == 'PARALLEL2D':
            L1 = np.sqrt(roi_max[0]**2 + roi_max[1]**2)/l1
            L1 = np.int(np.clip(L1,1,ds.Columns/2))
            sinogram = np.zeros([number, 2*L1])
            i = 0
            #for filename in glob.glob(os.path.join(DICOM_path, '*.dcm')):
            for i in range(number):
                filename = DICOM_path + '/projection_image' + str(i+1) + '.dcm'
                ds = dicom.read_file(filename)
                img = ds.pixel_array
                if LightFieldPath is not None:
                    img = img/LightField
                    img[np.isnan(img)] = 1
                    img[np.isinf(img)] = 1
                if Log is not None:
                    img = -np.log(img)
                    img[img<0] = 0
                sinogram[i,:] = img[center_1-L1:center_1+L1]
            
        if ProjectionGeometry == 'PARALLEL3D':
            L1 = np.sqrt(roi_max[0]**2 + roi_max[1]**2)/l1
            L2 = roi_max[2]/l2
            L1 = np.int(np.clip(L1,1,ds.Columns/2))
            L2 = np.int(np.clip(L2,1,ds.Rows/2))
            
            sinogram = np.zeros([number, 2*L1, 2*L2])
            i = 0
            #for filename in glob.glob(os.path.join(DICOM_path, '*.dcm')):
            for i in range(number):
                filename = DICOM_path + '/projection_image' + str(i+1) + '.dcm'
                ds = dicom.read_file(filename)
                img = ds.pixel_array
                if LightFieldPath is not None:
                    img = img/LightField
                    img[np.isnan(img)] = 1
                    img[np.isinf(img)] = 1
                if Log is not None:
                    img = -np.log(img)
                    img[img<0] = 0
                sinogram[i,:,:] = img[center_1-L1:center_1+L1, center_2-L2:center_2+L2]

    else:
        # No truncating is performed in this case
        sinogram = np.zeros([number, 2*L1, 2*L2])
        
        #for filename in glob.glob(os.path.join(DICOM_path, '*.dcm')):
        for i in range(number):
            filename = DICOM_path + '/projection_image' + str(i+1) + '.dcm'
            ds = dicom.read_file(filename)
            img = ds.pixel_array
            img = -np.log(img/LightField)
            img[img<0] = 0
            img[np.isnan(img)] = 0
            img[np.isinf(img)] = 0
            sinogram[i,:,:] = img
            
    return sinogram, ds
    

