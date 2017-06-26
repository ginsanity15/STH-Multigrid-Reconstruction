#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 23:49:55 2017

@author: Bo Gao

This file includes functions to store the reconstruction on disc

Input:
    Reconstruction result, storing name (Directory)
    
Output:
    Reconstruction data in .txt file or Open CT file (DICOM)
    
"""
import pickle
import dicom

def store_as_dcm(reco, filename, templatename, **kwargs):
    '''
    It is encouraged to store reconstruction image also as OPEN CT format (DICOM),
    even though users can hardly observe any information of interest when opening 
    the stored image with DICOM reader. 
    Storing reconstruction result with DICOM format give users option of containing
    relevant information about the reconstructed image in the header file, which will
    be useful when trying to display the image with functions developed in this repository
    '''
    ds = dicom.read_file(templatename)
    ds.pixel_array = reco.asarray()
    
    #TODO: Think about which tags in header file can be used here
    
    #TODOTOMORROW: Copy Forward and Backward PD, CG to functions folder, cause
    # we will give them the ability to calculate convergence curve
    
    ds.save_as(filename)

# If there is one sub-operator that could determine how many subspaces there 
# are in 'reco', this procedure can be further simplified and the functionality
# can be futher extended   
def store_as_txt(reco, filename_c, filename_f, **kwargs):
    '''
    Add some explanations here
    '''
    curve = kwargs.pop('curve', None)
    time = kwargs.pop('time', None)
    minimization = kwargs.pop('minimization', None)
    
    f = open(filename_c+'.txt','wb')
    coarse_image = reco[0].asarray()
    pickle.dump(coarse_image,f)
    f = open(filename_f+'.txt','wb')
    roi1 = reco[1].asarray()
    pickle.dump(roi1,f)
    
    if curve is not None:
        filename_curve = filename_c + '_curve.txt'
        f = open(filename_curve,'wb')
        pickle.dump(curve,f)
        
    if curve is not None:
        filename_time = filename_c + '_time.txt'
        f = open(filename_time,'wb')
        pickle.dump(time,f)
        
    if curve is not None:
        filename_minimization = filename_c + '_minimization.txt'
        f = open(filename_minimization,'wb')
        pickle.dump(minimization,f)

