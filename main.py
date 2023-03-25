# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:03:24 2022

@author: Daniel Abode

This code calls the functions for the power control GNN algorithm, including generation of data set, training and testing

References:
D. Abode, R. Adeogun, and G. Berardinelli, “Power control for 6g industrial wireless subnetworks: A graph neural network approach,”
2022. [Online]. Available: https://arxiv.org/abs/2212.14051
"""
import numpy as np                        
import matplotlib.pyplot as plt            
import torch
from torch_geometric.loader import DataLoader
import subnetwork_generator
import Sum_rate_power_allocator as lr_H_mat_code


class init_parameters:
    def __init__(self,S_sd,rng, num_of_subn):
        self.num_of_subnetworks = num_of_subn
        self.deploy_length = 20    # the length and breadth of the factory area (m)  
        self.subnet_radius = 2      # the radius of the subnetwork cell (m)
        self.minD = 0.5               #minimum distance from device to controller(access point) (m)
        self.minDistance = self.subnet_radius    #minimum controller to controller distance (m)
        self.sigmaS = S_sd        #shadowing standard deviation
        self.transmit_power = 1      #normalized transmit power mW
        self.rng_value = np.random.RandomState(rng)
        self.bandwidth = 5e6         #bandwidth (Hz)
        self.frequency = 6e9          #Operating frequency (Hz)
        self.lambdA = 3e8/6e9        
        self.plExponent = 2.7        #path loss exponent
        


f_metric_ = ['hH','dD', 'hD']   #hH, dD, hD #The type of graph attribute hH - use full channel gain, dD - use only distance information, hD - use desired link channel gain and interfering link distances

def run_experiment(f_metric, train_num_subn, test_num_subn, trainsh_sd, testsh_sd, device):
    #f_metric options - hH, dD, hD #The type of graph attribute hH - use full channel gain, dD - use only distance information, hD - use desired link channel gain and interfering link distances  
    #train_num_subn - number of subnetworks in training deployment - choose between values 20, 25, 10, for larger values, there is a need to increase self.deploy_length
    #test_num_subn  - number of subnetworks in testing deployments - choose between values 20, 25, 10, for larger values, there is a need to increase self.deploy_length
    #trainsh_sd - the shadowing standard deviation of the training deployments - reasonable values between 4 and 10
    #testsh_sd - the shadowing standard deviation of the testing environment - reasonable values between 4 and 10
    #device - Choose appropriate device - 'cuda', 'cpu'
    
    
    train_config = init_parameters(trainsh_sd,0,train_num_subn)
    val_config = init_parameters(trainsh_sd,1,train_num_subn)
    training_snapshots = 10000
    validation_snapshots = 5000
    
    
    print('#### Generating training and validation dataset ####')
    training_powers, t_dist= subnetwork_generator.generate_samples(train_config, training_snapshots)
    validation_powers, v_dist= subnetwork_generator.generate_samples(val_config, validation_snapshots)
       
    if f_metric == 'hH':
        training_features = training_powers
        validation_features = validation_powers
    elif f_metric == 'hD':
        training_features = lr_H_mat_code.create_features(t_dist, training_powers)
        validation_features = lr_H_mat_code.create_features(v_dist, validation_powers)
    elif f_metric == 'dD':
        training_features = t_dist
        validation_features = v_dist
    
    num_of_subnetworks = train_config.num_of_subnetworks
    
    bandwidth = train_config.bandwidth
    Noise_power = np.power(10,((-174+10+10*np.log10(bandwidth))/10))
    
    norm_train_features, norm_validation_features = lr_H_mat_code.normalize_data(training_features, validation_features)
    training_Graph_list = lr_H_mat_code.create_graph_list(norm_train_features, training_powers)
    validation_Graph_list = lr_H_mat_code.create_graph_list(norm_validation_features, validation_powers)
    train_loader = DataLoader(training_Graph_list, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_Graph_list, batch_size=64, shuffle=False)
    
    #Training
    print('#### Training Model ####')
    model2 = lr_H_mat_code.PCGNN().to('cuda')
    model_name = f_metric + 'sh_sd' + str(trainsh_sd)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    loss_, losst_ = lr_H_mat_code.trainmodel(model_name,model2,scheduler, train_loader, validation_loader,optimizer,num_of_subnetworks, Noise_power, device)
    
    print('min loss = ', np.min(np.array(loss_)))
    print('max loss = ', np.max(np.array(loss_)))
    print(np.max(np.array(loss_)) - np.min(np.array(loss_)))
    plt.figure(str(trainsh_sd))
    plt.plot(-np.array(loss_), 'r', -np.array(losst_), 'b')
    plt.legend(('Validation','training'))
    plt.ylabel('sum log rate')
    plt.xlabel('epochs')
    plt.title('sum log rate)')
      
    
    ##Testing
    print('#### Testing Model ####')
    testing_snapshots = 50000
    
    test_config = init_parameters(testsh_sd,1,test_num_subn)
    num_of_subnetworks = test_config.num_of_subnetworks
    print('#### Generating test dataset ####')
    test_powers, te_dist = subnetwork_generator.generate_samples(test_config, testing_snapshots)
    test_features = test_powers
    
    if f_metric == 'hH':
        test_features = test_powers
    elif f_metric == 'hD':
        test_features = lr_H_mat_code.create_features(te_dist, test_powers)
    elif f_metric == 'dD':
        test_features = te_dist
       
    
    norm_train_features, norm_test_features = lr_H_mat_code.normalize_data(training_features, test_features)
    test_Graph_list = lr_H_mat_code.create_graph_list(norm_test_features, test_powers)
    test_loader = DataLoader(test_Graph_list, batch_size=100, shuffle=False)
    
    
    model_name_ = model_name
    GNN_sum_rate, GNN_capacities, GNN_weights, GNN_powers = lr_H_mat_code.GNN_test(model_name_, test_loader, num_of_subnetworks, Noise_power, device)
    
    test_loader = DataLoader(test_Graph_list, batch_size=testing_snapshots, shuffle=False)
    for data in test_loader:
        break
    
    
    
    weights_ones = torch.ones((test_features.shape[0],test_features.shape[1]))
    capacities_ones, Uniform_pow = lr_H_mat_code.mycapacity(weights_ones, data, data.num_graphs,num_of_subnetworks, Noise_power)
    
    av_ones_cap = (torch.sum(torch.sum(capacities_ones,1))/(testing_snapshots*num_of_subnetworks)).item()
    av_GNN_cap = (torch.sum(GNN_sum_rate)/(testing_snapshots*num_of_subnetworks)).item()
    print('Average spectral efficiency achieved by PCGNN = ', av_GNN_cap)
    print('Average spectral efficiency achieved by Max power = ', av_ones_cap)
    print('PCGNN gain (%)=', ((av_GNN_cap-av_ones_cap)/av_ones_cap)*100)
    
    
    ##transmit powers
    plt.figure(str(trainsh_sd)+str(testsh_sd)+str(test_num_subn)+str(0))
    x,y = lr_H_mat_code.generate_cdf(10*np.log10(GNN_weights + 1e-18), 1000)
    plt.plot(x,y, label = "GNN"+f_metric)
     
    
    plt.title('Transmit power: :'+str(test_num_subn)+str(train_num_subn)+str(trainsh_sd)+str(testsh_sd))
    plt.legend()
    plt.grid(which='both')
    plt.xlabel('Transmit Power dBm')
    plt.ylabel('cdf')
    
    
    ## individual capacites
    plt.figure(str(trainsh_sd)+str(testsh_sd)+str(test_num_subn)+str(1))
    x,y = lr_H_mat_code.generate_cdf(GNN_capacities,1000)
    plt.plot(x,y, label = "GNN"+f_metric)    
    x,y = lr_H_mat_code.generate_cdf(capacities_ones,1000)
    plt.plot(x,y, label = "Uniform Power")    
    plt.title('Individual Subnetworks Spectral Efficiency :'+str(test_num_subn)+str(train_num_subn)+str(trainsh_sd)+str(testsh_sd))
    plt.legend()
    plt.grid(which='both')
    plt.ylabel('cdf')
    plt.xlabel('SE (b/s/Hz)')
    
    
    ## individual capacities outage
    plt.title('Individual Subnetworks Outage Spectral Efficiency:'+str(test_num_subn)+str(train_num_subn)+str(trainsh_sd)+str(testsh_sd))
    plt.figure(str(trainsh_sd)+str(testsh_sd)+str(test_num_subn)+str(2))
    x,y = lr_H_mat_code.generate_cdf(GNN_capacities,50000)
    plt.plot(x[0:3000],y[0:3000])
    x,y = lr_H_mat_code.generate_cdf(capacities_ones,50000)
    plt.plot(x[0:3000],y[0:3000])
    plt.grid(which='both')
    plt.xlim([0,0.5])
    plt.ylim([0,0.3])
      
    ## sum capacites
    plt.figure(str(trainsh_sd)+str(testsh_sd)+str(test_num_subn)+str(3))
    x,y = lr_H_mat_code.generate_cdf(GNN_sum_rate/num_of_subnetworks,1000)
    plt.plot(x,y, label = "GNN"+f_metric)
    x,y = lr_H_mat_code.generate_cdf(torch.sum(capacities_ones,1)/num_of_subnetworks,1000)
    plt.plot(x,y, label = "Uniform Power")
     
    
    #Average             
    plt.title('Sum Subnetworks SE: test subnetworks ='+str(test_num_subn)+str(train_num_subn)+str(trainsh_sd)+str(testsh_sd))
    plt.legend()
    plt.grid(which='both')
    plt.ylabel('cdf')
    plt.xlabel('SE (b/s/Hz)')
    
      
       
       
       
       
       
