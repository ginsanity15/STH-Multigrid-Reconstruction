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
sys.path.insert(0, '/Users/starbury/odl/STH-Multigrid-Reconstruction/functions')

import display_function as df
import sinogram_generation as sg
# %%
# Given the path that stores all those projection images in DICOM format, users 
# may need to modify this based on the directory they store the dataset
DICOM_path = '/Users/starbury/odl/STH-Multigrid-Reconstruction/Data'

# Directory for storing the .txt file that includes information of the reconstructed image 
output_store_path = '/home/davlars/Bo/real/Data_LC_512/TV/'

# Path to the Light Field image
Light_Field_Path = ''

# Define the reconstruction space, these two points should be the opposite of each other
# Decreasing the size of these two indices can increase the reconstruction speed
min_pt = [-20,-20,-1]
max_pt = [20, 20, 1]

# TODO: write a function to truncate projection image to include ROI only and 
# output the combined sinogram as well as one DICOM file (arbitrarily, we are 
# only interested in the identical information stored in header file)
sino, ds = sg.sino_gene(DICOM_path, min_pt, max_pt, Light_Field_Path, Log = 1)

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
insert_grad = 2.8*odl.Gradient(insert_discr1, pad_mode='order1')
        
# Differentiable part, build as ||. - g||^2 o P
data_func = odl.solvers.L2NormSquared(sum_ray_trafo.range).translated(data) * sum_ray_trafo
reg_param_1 = 2e-1
reg_func_1 = reg_param_1 * (odl.solvers.L2NormSquared(coarse_discr) *
                            odl.ComponentProjection(pspace, 0))
smooth_func = data_func + reg_func_1

# Non-differentiable part composed with linear operators
reg_param = 2e-1
nonsmooth_func = reg_param * odl.solvers.L1Norm(insert_grad.range)
        
# Assemble into lists (arbitrary number can be given)
comp_proj_1 = odl.ComponentProjection(pspace, 1)
lin_ops = [insert_grad * comp_proj_1]
nonsmooth_funcs = [nonsmooth_func]

box_constr = odl.solvers.IndicatorBox(pspace, 0, 40) # According to reconstruction result of FBP
f = box_constr

# eta^-1 is the Lipschitz constant of the smooth functional gradient
# xstart can be anything, now let's initialize it with Shepp-Logan Phantom
phantom_c = odl.phantom.shepp_logan(coarse_discr, modified=True)
phantom_f = odl.phantom.shepp_logan(insert_discr1, modified=True)
phantom = pspace.element([phantom_c, phantom_f])
ray_trafo_norm = 1.1 * odl.power_method_opnorm(sum_ray_trafo,
                                               xstart=phantom, maxiter=2)
print('norm of the ray transform: {}'.format(ray_trafo_norm))
eta = 1 / (2 * ray_trafo_norm ** 2 + 2 * reg_param_1)
print('eta = {}'.format(eta))
grad_norm = 1.1 * odl.power_method_opnorm(insert_grad,
                                          xstart=phantom_f,
                                          maxiter=4)
print('norm of the gradient: {}'.format(grad_norm))
        
# tau and sigma are like step sizes
sigma = 4e-4
tau = 1.0 * sigma
# Here we check the convergence criterion for the forward-backward solver
# 1. This is required such that the square root is well-defined
print('tau * sigma * grad_norm ** 2 = {}, should be <= 1'
      ''.format(tau * sigma * grad_norm ** 2))
assert tau * sigma * grad_norm ** 2 <= 1
# 2. This is the actual convergence criterion
check_value = (2 * eta * min(1 / tau, 1 / sigma) *
               np.sqrt(1 - tau * sigma * grad_norm ** 2))
print('check_value = {}, must be > 1 for convergence'.format(check_value))
convergence_criterion = check_value > 1
assert convergence_criterion

callback = (odl.solvers.CallbackPrintIteration())# &
#            odl.solvers.CallbackShow())
reco = pspace.zero()  # starting point
curve, time_slot, minimization = odl.solvers.forward_backward_pd(
            reco, f=f, g=nonsmooth_funcs, L=lin_ops, h=smooth_func,
            tau=tau, sigma=[sigma], niter=150, callback=callback, RayTrafo = sum_ray_trafo,
            Sinogram = data, timeslot = 1, minimization = 1)
      
# %% Storing the image
f = open(filename_c,'wb')
coarse_image = reco[0].asarray()
pickle.dump(coarse_image,f)
f = open(filename_f,'wb')
roi = reco[1].asarray()
pickle.dump(roi,f)
    
# %% Display multi-grid image
df.Display_multigrid(coarse_image, roi)