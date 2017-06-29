#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 15:33:15 2017

@author: Bo Gao

The following is the utility script used to convert binary output data form the 
KTH-STH microCT to odl-compatible OpenCT data.

Input data:

Output data:

Requirements:
    pydicom (import dicom)
    ...
    
"""

import numpy as np
import odl
import odl_multigrid as multigrid
import pickle
import sys
sys.path.insert(0, '/home/davlars/STH-Multigrid-Reconstruction/functions')

import display_functions as df
import sinogram_generation as sg
# %%
# Given the path that stores all those projection images in DICOM format, users 
# may need to modify this based on the directory they store the dataset
DICOM_path = '/home/davlars/microCT/projections/'

# Path to the Light Field image
Light_Field = '/home/davlars/microCT/LF/Light_Field.dcm'

# Directory for storing the .txt file that includes information of the reconstructed image 
output_store_path = '/home/davlars/STH-Multigrid-Reconstruction/output/'

# Define the reconstruction space, these two points should be the opposite of each other
# Decreasing the size of these two indices can increase the reconstruction speed
min_pt = [-20,-20,-1]
max_pt = [20, 20, 1]

# TODO: write a function to truncate projection image to include ROI only and 
# output the combined sinogram as well as one DICOM file (arbitrarily, we are 
# only interested in the identical information stored in header file)
sino, ds = sg.sino_gene(DICOM_path,
                        roi_min=min_pt,
                        roi_max=max_pt,
                        LightFieldPath = Light_Field,
                        Log=1)

# These three numbers corresponds to the number of projection image as well as
# the size of each projection image
num, L1, L2 = sino.shape

# ODL only support reconstruction of projection data collected counter clockwise,
# read in information from projection image's header part, if the projection image
# is collected clockwise, convert its order to counter clockwise
rot_dir = ds.RotationDirection
if rot_dir == 'CW':
    sino_first = sino[0:1,:,:]
    sino_rest = sino[1:num,:,:]
    sino_rest = np.flipud(sino_rest)
    sino = np.concatenate([sino_first, sino_rest])

# Read out geometry information about the micro-CT scanner from the header of
# the DICOM file
src = ds.DistanceSourceToPatient
src = np.float(src)
det = ds.RadialPosition
det = np.float(det)

# Read in information on where each projeciton is collected 
initial = ds.StartAngle
initial = np.float(initial)
end = ds.ScanArc
end = np.float(end)

# Number of pixels along each row and column on projection image
length = ds.Rows
length = np.int(length)
width = ds.Columns
width = np.int(width)

# Check is the detector has performed binning (e.g. combine intensity on four pixels
# and output as one) on the output projection 
Binning_scale = ds.DetectorBinning
Bin_scale = 1/float(Binning_scale[0])

# cell refers to the size of pixel on detector
cell = ds.DetectorElementTransverseSpacing/length * int(Bin_scale)
cell = np.float(cell)

# pixel_space refers to the size of pixel on reconstruction space
pixel_space = ds.ReconstructionPixelSpacing
pixel_space = np.float(pixel_space[0])

# Below sets the max number of pixels included in one direction on coarse grid
# This can be given arbitrarily, however, through testing, it is not recommend
# to set a value lower than 50, even when this number equals 50, an obvious
# difference on intensity can be observed at ROI and backgound
coarse_length = 200
coarse_length_x = np.int(coarse_length * max_pt[0]/max(max_pt))
coarse_length_y = np.int(coarse_length * max_pt[1]/max(max_pt))
coarse_length_z = np.int(coarse_length * max_pt[2]/max(max_pt))

# Below sets the max number of pixels included in one direction on fine grid
# This number is determined by the range of reconstruction space as well as the 
# size of each pxiel on the reconstruction space
fine_length_x = np.int((max_pt[0] - min_pt[0])/pixel_space)
fine_length_y = np.int((max_pt[1] - min_pt[1])/pixel_space)
fine_length_z = np.int((max_pt[2] - min_pt[2])/pixel_space)

# Define space to store background image and ROI
filename_c = output_store_path + 'CG_coarse_space_' + str(coarse_length) + '.txt'
filename_f = output_store_path + 'CG_coarse_space_' + str(coarse_length) + '_fine.txt'

# Define the reconstruction space (both coarse grid and fine grid) depends on the 
# setting give above
coarse_discr = odl.uniform_discr(min_pt, max_pt, [coarse_length_x, coarse_length_y, coarse_length_z])
fine_discr = odl.uniform_discr(min_pt, max_pt, [fine_length_x, fine_length_y, fine_length_z])

# Define ROI here
insert_min_pt1 = [-5, 0, -.5]
insert_max_pt1 = [5, 10, .5]

# Pre-process the ROI to ensure the edge effect will be minimized
index1 = np.floor((np.array(insert_min_pt1) - np.array(min_pt))/coarse_discr.cell_sides)
insert_min_pt1 = np.array(min_pt) + coarse_discr.cell_sides*index1
index2 = np.floor((np.array(max_pt) - np.array(insert_max_pt1))/coarse_discr.cell_sides)
insert_max_pt1 = np.array(max_pt) - coarse_discr.cell_sides*index2

# Define the detector
det_min_pt = -np.array([L1/2, L2/2])*cell
det_max_pt = -det_min_pt
detector_partition = odl.uniform_partition(det_min_pt, det_max_pt, [L1, L2])

# Define the angle that each projeciton image is collected
angle_partition = odl.uniform_partition(0, end - initial, num)

# Geometry of the projection (Parallel beam, Fan Flat beam or Cone Flat Beam)
geometry = odl.tomo.CircularConeFlatGeometry(angle_partition, detector_partition, 
                                             src, det)

# Mask out ROI on the coarse grid
coarse_mask1 = multigrid.operators.MaskingOperator(coarse_discr, insert_min_pt1, insert_max_pt1)

# Define the forward operator of the masked coarse grid
coarse_ray_trafo = odl.tomo.RayTransform(coarse_discr, geometry,impl='astra_cuda')
masked_coarse_ray_trafo = coarse_ray_trafo * coarse_mask1
      
# Define insert discretization using the fine cell sizes but the insert
# min and max points   
insert_discr1 = odl.uniform_discr_fromdiscr(
                fine_discr, min_pt=insert_min_pt1, max_pt=insert_max_pt1,
                cell_sides=fine_discr.cell_sides)
    
# Ray trafo on the insert discretization only
insert_ray_trafo1 = odl.tomo.RayTransform(insert_discr1, geometry,impl='astra_cuda')
    
# Forward operator = sum of masked coarse ray trafo and insert ray trafo
sum_ray_trafo = odl.ReductionOperator(masked_coarse_ray_trafo, insert_ray_trafo1)
    
# Make phantom in the product space
pspace = sum_ray_trafo.domain

reco = sum_ray_trafo.domain.zero()
data = sum_ray_trafo.range.element(sino*1000)
    
# %% Reconstruction
callback = odl.solvers.CallbackShow()
odl.solvers.conjugate_gradient_normal(sum_ray_trafo, reco, data,
                                      niter=10, callback=callback)
      
# %% Storing the image
f = open(filename_c,'wb')
coarse_image = reco[0].asarray()
pickle.dump(coarse_image,f)
f = open(filename_f,'wb')
roi = reco[1].asarray()
pickle.dump(roi,f)
    
# %% Display multi-grid image
df.Display_multigrid(coarse_image, roi)