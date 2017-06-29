#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:02:10 2017

@author: starbury

This is a custom solver for multi-grid reconstruction, which serves to reconstruct
the coarse grid with Conjugate Gradient Normal while the fine grid with Forward-Backward
Prime-Dual. 

Input parameters:
    op: Forward Operator
    x: Initial Guess of reconstruction space (both the coarse grid and the fine grid),
    which is not recommended to be set to zero.
    rhs: Projection data
"""

import odl
import numpy as np

def CG_FB_mixed(op, x, rhs, niters, callback = None):
    insert_grad = 2.8*odl.Gradient(op[1].domain, pad_mode='order1')
        
    # Differentiable part only of ROI, build as ||. - g||^2 o P
    data = rhs - op[0](x[0])
    data_func = odl.solvers.L2NormSquared(op[1].range).translated(data) * op[1]
    reg_param_1 = 2e-1
    reg_func_1 = reg_param_1 * (odl.solvers.L2NormSquared(op[1].domain) *
                                odl.ComponentProjection(op[1].domain, 0))
    const_coarse = odl.sum(rhs - op[1](x[1]))
    smooth_func = data_func + reg_func_1 + const_coarse

    # Non-differentiable part composed with linear operators
    reg_param = 2e-1
    nonsmooth_func = reg_param * odl.solvers.L1Norm(insert_grad.range)
    
    # Assemble into lists (arbitrary number can be given)
    comp_proj_1 = odl.ComponentProjection(op.domain, 1)
    lin_ops = [insert_grad * comp_proj_1]
    nonsmooth_funcs = [nonsmooth_func]
    
    box_constr = odl.solvers.IndicatorBox(op.domain, 0, 40) # According to reconstruction result of FBP
    f = box_constr
    
    # eta^-1 is the Lipschitz constant of the smooth functional gradient
    # xstart can be anything, now let's initialize it with Shepp-Logan Phantom
    
    ray_trafo_norm = 1.1 * odl.power_method_opnorm(op[1],xstart=phantom, maxiter=2)
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

    
    reco = op.domain.zero()  # starting point
    
    for i in range(niters):
        # In each iteration, update the coarse grid with CG for one iteration
        odl.solvers.conjugate_gradient_normal(op[0], x[0], rhs,
                                              niter=1, callback=None)
        odl.solvers.forward_backward_pd(reco, f=f, g=nonsmooth_funcs, L=lin_ops, 
                                        h=smooth_func, tau=tau, sigma=[sigma], 
                                        niter=10, callback=None)
        
        if callback is not None:
            callback(x)
            
            
            
            
            
# %%
# Here we can write a simple test example
phantom_c = odl.phantom.shepp_logan(coarse_discr, modified=True)
phantom_f = odl.phantom.shepp_logan(insert_discr1, modified=True)
x = pspace.element([phantom_c, phantom_f])

callback = (odl.solvers.CallbackPrintIteration())# &
    #            odl.solvers.CallbackShow())
        