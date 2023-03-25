# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 18:23:16 2022
This code include functions to generate subnetwork deployment
@author: Abode Daniel, Ramoni Adeogun, AAU [2022]

References:
D. Abode, R. Adeogun, and G. Berardinelli, “Power control for 6g industrial wireless subnetworks: A graph neural network approach,”
2022. [Online]. Available: https://arxiv.org/abs/2212.14051  
"""

import numpy as np
from scipy.spatial.distance import cdist

def randMin(Npoints,mindist,deploy_param):
    x = deploy_param.rng_value.rand(10000,1)
    keeperX = np.zeros([Npoints,1],dtype=np.float64)
    keeperX[0] = x[0]
    counter = 1
    k = 1
    while counter < Npoints:
        thisX = x[k]
        minDistance = np.min(np.abs(thisX - keeperX))
        if minDistance >= mindist:
            keeperX[counter] = thisX
            counter += 1
        k += 1
    return keeperX

def create_layout(deploy_param):
    N = deploy_param.num_of_subnetworks
    bound = deploy_param.deploy_length - 2*deploy_param.subnet_radius
    # XLoc = deploy_param.subnet_radius+deploy_param.rng_value.uniform(low=0, high=bound, size=[N,1])
    # YLoc = deploy_param.subnet_radius+deploy_param.rng_value.uniform(low=0, high=bound, size=[N,1])
    X = np.zeros([deploy_param.num_of_subnetworks,1],dtype=np.float64)
    Y = np.zeros([deploy_param.num_of_subnetworks,1],dtype=np.float64)
    dist_2 = deploy_param.minDistance**2
    loop_terminate = 1
    nValid = 0
    while nValid < deploy_param.num_of_subnetworks and loop_terminate < 1e6:
        newX = bound*(deploy_param.rng_value.uniform()-0.5)
        newY = bound*(deploy_param.rng_value.uniform()-0.5)
        if all(np.greater(((X[0:nValid] - newX)**2 + (Y[0:nValid] - newY)**2),dist_2)):
            X[nValid] = newX
            Y[nValid] = newY
            nValid = nValid+1
        loop_terminate = loop_terminate+1
    if nValid < deploy_param.num_of_subnetworks:
        print("Invalid number of subnetworks for deploy size")
        exit
    #Location of the access points
    X = X+deploy_param.deploy_length/2
    Y = Y+deploy_param.deploy_length/2
    gwLoc = np.concatenate((X, Y), axis=1)
    #cellRange = deploy_param.subnet_radius - deploy_param.minD
    dist_rand = deploy_param.rng_value.uniform(low=deploy_param.minD, high=deploy_param.subnet_radius, size=[N,1])
    angN = deploy_param.rng_value.uniform(low=0, high=2*np.pi, size=[N,1])
    D_XLoc = X + dist_rand*np.cos(angN)
    D_YLoc = Y + dist_rand*np.sin(angN)
    dvLoc = np.concatenate((D_XLoc, D_YLoc), axis=1)
    dist = cdist(gwLoc,dvLoc)
    #print('gwloc ',gwLoc)
    #print(dist)
    return dist

def compute_power(deploy_param, dist):
    N = deploy_param.num_of_subnetworks
    S = deploy_param.sigmaS*deploy_param.rng_value.randn(N,N)
    S_linear = 10**(S/10)
    h = (1/np.sqrt(2))*(deploy_param.rng_value.randn(N,N)+1j*deploy_param.rng_value.randn(N,N))
    #power_PC = np.repeat(power_PC,N,axis=1)
    power = deploy_param.transmit_power*(4*np.pi/deploy_param.lambdA)**(-2)\
        *(np.power(dist,-1*deploy_param.plExponent))\
        *S_linear*np.power(np.abs(h),2)
    return power, S

def generate_samples(deploy_param,number_of_snapshots):
    N = deploy_param.num_of_subnetworks
    distance_ = np.zeros([number_of_snapshots,N,N])
    Channel_gain = np.zeros([number_of_snapshots,N,N])
    for k in range(number_of_snapshots):
        dist = create_layout(deploy_param)
        Channel_gain[k,:,:],S = compute_power(deploy_param,dist)
        distance_[k,:,:] = dist
    return Channel_gain, distance_
    