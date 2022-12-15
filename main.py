# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:03:24 2022

@author: Daniel Abode
"""
import numpy as np                        
import matplotlib.pyplot as plt            
import torch
from torch_geometric.loader import DataLoader
import subnetwork_generator
import tikzplotlib
import Sum_rate_power_allocator as lr_H_mat_code
import time
import pandas as pd


class init_parameters:
    def __init__(self,S_sd,rng, num_of_subn):
        self.num_of_subnetworks = num_of_subn
        self.deploy_length = 20    # m
        self.subnet_radius = 2      # m
        self.minD = 0.5               #device to controller m
        self.minDistance = self.subnet_radius    #controller to controller distance m
        self.sigmaS = S_sd        #shadowing standard deviation
        self.transmit_power_dBm = 0  # P_max for OLPC dBm 
        self.transmit_power = 1      #normalized transmit power mW
        self.rng_value = np.random.RandomState(rng)
        self.bandwidth = 5e6         #Hz
        self.P0 = -50                #dBm
        self.frequency = 6e9          #dBm
        self.lambdA = 3e8/6e9        
        self.plExponent = 2.7        #path loss exponent
        


f_metric_ = ['hH','dD', 'hD']   #hH, dD, hD
l_metric = 'sum rate' 
d = 0

for f_metric in f_metric_:
    for trainsh_sd in [7]:  #specify the training shadowing standard deviation
        
       if d == 0:
           train_config = init_parameters(trainsh_sd,0,20)
           val_config = init_parameters(trainsh_sd,1,20)
           training_snapshots = 10000
           validation_snapshots = 5000
       
           training_powers, t_OLPC_powers, t_dist= subnetwork_generator.generate_samples(train_config, training_snapshots)
           validation_powers, v_OLPC_powers, v_dist= subnetwork_generator.generate_samples(val_config, validation_snapshots)
       
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
       
       model2 = lr_H_mat_code.PCGNN().to('cuda')
       model_name = f_metric + l_metric  + 'sh_sd' + str(trainsh_sd)
       optimizer = torch.optim.Adam(model2.parameters(), lr=0.0001)
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
       
       loss_, losst_ = lr_H_mat_code.trainmodel(model_name,model2,scheduler, train_loader, validation_loader,optimizer,num_of_subnetworks, Noise_power)
       
       print('min loss = ', np.min(np.array(loss_)))
       print('max loss = ', np.max(np.array(loss_)))
       print(np.max(np.array(loss_)) - np.min(np.array(loss_)))
       plt.figure(str(trainsh_sd))
       plt.plot(-np.array(loss_), 'r', -np.array(losst_), 'b')
       plt.legend(('Validation','training'))
       plt.ylabel('sum log rate')
       plt.xlabel('epochs')
       plt.title('sum log rate)')
       plt.savefig('loss')
       loss_plt_name = "tikz files/"+f_metric + l_metric + str(trainsh_sd) + str(time.time()) + ".tex"
       tikzplotlib.save(loss_plt_name)
       
       
       
       ##Testing for different environment
       testsh_sd_ = [7]  #Specify the test shadowing standard deviation
       test_num_subn = [20]
       df_av = pd.DataFrame(columns=['GNN','OLPC','Uniform'])
       df_md = pd.DataFrame(columns=['GNN','OLPC','Uniform'])
       df_ot = pd.DataFrame(columns=['GNN','OLPC','Uniform'])
       
       for testsh_sd in testsh_sd_:
           for num_subn in test_num_subn:
               if d == 0:
                   testing_snapshots = 50000
                   test_config = init_parameters(testsh_sd,1,num_subn)
                   num_of_subnetworks = test_config.num_of_subnetworks
                   test_powers, te_OLPC_powers, te_dist = subnetwork_generator.generate_samples(test_config, testing_snapshots)
                   test_features = test_powers
               
               if f_metric == 'hH':
                   test_features = test_powers
               elif f_metric == 'hD':
                   test_features = lr_H_mat_code.create_features(te_dist, test_powers)
               elif f_metric == 'dD':
                   test_features = te_dist
                  
               
               norm_train_features, norm_test_features = lr_H_mat_code.normalize_data(training_features, test_features)
               test_Graph_list = lr_H_mat_code.create_graph_list(norm_test_features, test_powers)
               test_loader = lr_H_mat_code.DataLoader(test_Graph_list, batch_size=100, shuffle=False)
               
               
               model_name_ = 'models/'+ model_name
               GNN_sum_rate, GNN_capacities, GNN_weights, GNN_powers = lr_H_mat_code.GNN_test(model_name_, test_loader, num_of_subnetworks, Noise_power)
               
               test_loader = lr_H_mat_code.DataLoader(test_Graph_list, batch_size=testing_snapshots, shuffle=False)
               for data in test_loader:
                   break
               
               
               OLPC_powers2 = 10*np.log10(te_OLPC_powers)
               OLPC_capacities, power = lr_H_mat_code.mycapacity(torch.tensor(te_OLPC_powers), data, data.num_graphs,test_config.num_of_subnetworks, Noise_power)
        
               weights_ones = torch.ones_like(torch.tensor(te_OLPC_powers))
               capacities_ones, Uniform_pow = lr_H_mat_code.mycapacity(weights_ones, data, data.num_graphs,num_of_subnetworks, Noise_power)
               
               av_ones_cap = (torch.sum(torch.sum(capacities_ones,1))/(testing_snapshots*num_of_subnetworks)).item()
               av_OLPC_cap = (torch.sum(torch.sum(OLPC_capacities,1))/(testing_snapshots*num_of_subnetworks)).item()
               av_GNN_cap = (torch.sum(GNN_sum_rate)/(testing_snapshots*num_of_subnetworks)).item()
               df_av.loc[testsh_sd] = [av_GNN_cap, av_OLPC_cap, av_ones_cap]
               
               ##transmit powers
               plt.figure(str(trainsh_sd)+str(testsh_sd)+str(num_subn)+str(0))
               x,y = lr_H_mat_code.generate_cdf(10*np.log10(GNN_weights + 1e-18), 1000)
               plt.plot(x,y, label = "GNN"+f_metric)
                
               x,y = lr_H_mat_code.generate_cdf(OLPC_powers2, 1000)
               plt.plot(x,y, label = "OLPC")
               
               plt.title('Transmit power: test subnetworks = '+str(num_subn))#+str(trainsh_sd)+str(testsh_sd))
               plt.legend()
               plt.grid(which='both')
               plt.xlabel('Transmit Power dBm')
               plt.ylabel('cdf')
               pow_plot_name = "tikz files/"+f_metric + l_metric+"_pow"+" test subnetworks = "+str(num_subn)+ str(time.time()) + ".tex"
               tikzplotlib.save(pow_plot_name)
               
               ## individual capacites
               plt.figure(str(trainsh_sd)+str(testsh_sd)+str(num_subn)+str(1))
               x,y = lr_H_mat_code.generate_cdf(GNN_capacities,1000)
               plt.plot(x,y, label = "GNN"+f_metric)
               md_GNN_cap = lr_H_mat_code.findcdfvalue(x, y, 0.49, 0.51)
                
               x,y = lr_H_mat_code.generate_cdf(capacities_ones,1000)
               plt.plot(x,y, label = "Uniform Power")
               md_ones_cap = lr_H_mat_code.findcdfvalue(x, y, 0.49, 0.51)
               
               x,y = lr_H_mat_code.generate_cdf(OLPC_capacities,1000)
               plt.plot(x,y, label = "OLPC")
               md_OLPC_cap = lr_H_mat_code.findcdfvalue(x, y, 0.49, 0.51)
               df_md.loc[testsh_sd] = [md_GNN_cap, md_OLPC_cap, md_ones_cap]
               
               
               plt.title('Individual Subnetworks SE : test subnetworks ='+str(num_subn))#+str(trainsh_sd)+str(testsh_sd))
               plt.legend()
               plt.grid(which='both')
               plt.ylabel('cdf')
               plt.xlabel('SE (b/s/Hz)')
               ind_sub_plot_name = "tikz files/"+f_metric + l_metric+"_indSE"+" test subnetworks = "+str(num_subn)+str(time.time()) + ".tex"
               tikzplotlib.save(ind_sub_plot_name)
               
               ## individual capacities outage
               plt.figure(str(trainsh_sd)+str(testsh_sd)+str(num_subn)+str(2))
               x,y = lr_H_mat_code.generate_cdf(GNN_capacities,50000)
               ot_GNN_cap = lr_H_mat_code.findcdfvalue(x, y, 0.049, 0.051)
               plt.plot(x[0:3000],y[0:3000])
               
               x,y = lr_H_mat_code.generate_cdf(capacities_ones,50000)
               ot_ones_cap = lr_H_mat_code.findcdfvalue(x, y, 0.049, 0.051)
               plt.plot(x[0:3000],y[0:3000])
               
               x,y = lr_H_mat_code.generate_cdf(OLPC_capacities,50000)
               plt.plot(x[0:3000],y[0:3000])
               ot_OLPC_cap = lr_H_mat_code.findcdfvalue(x, y, 0.049, 0.051)
               df_ot.loc[testsh_sd] = [ot_GNN_cap, ot_OLPC_cap, ot_ones_cap]
               
               plt.grid(which='both')
               plt.xlim([0,0.5])
               plt.ylim([0,0.3])
               ind_sub_out_plot_name =  "tikz files/"+f_metric + l_metric+"_indSE_Outage"+" test subnetworks = "+str(num_subn)+str(time.time()) +".tex"
               tikzplotlib.save(ind_sub_out_plot_name)
               
               ## sum capacites
               plt.figure(str(trainsh_sd)+str(testsh_sd)+str(num_subn)+str(3))
               x,y = lr_H_mat_code.generate_cdf(GNN_sum_rate/num_of_subnetworks,1000)
               plt.plot(x,y, label = "GNN"+f_metric)
               x,y = lr_H_mat_code.generate_cdf(torch.sum(capacities_ones,1)/num_of_subnetworks,1000)
               plt.plot(x,y, label = "Uniform Power")
                
               x,y = lr_H_mat_code.generate_cdf(torch.sum(OLPC_capacities,1)/num_of_subnetworks,1000)
               plt.plot(x,y, label = "OLPC Po = -50dBm, Pmax= 0dBm")
               
               #Average             
               plt.title('Sum Subnetworks SE: test subnetworks ='+str(num_subn)+str(trainsh_sd)+str(testsh_sd))
               plt.legend()
               plt.grid(which='both')
               plt.ylabel('cdf')
               plt.xlabel('SE (b/s/Hz)')
               sumlog_hH_plot_name = "tikz files/"+f_metric + l_metric+"_sumSE"+" test subnetworks = "+str(num_subn)+ str(time.time()) + ".tex"
               tikzplotlib.save(sumlog_hH_plot_name)
               d = d + 1
               
           
       df = pd.concat([df_av,df_md,df_ot],keys=["av","md","ot"])
       df_data_name = "df_data/"+f_metric + l_metric+ str(num_subn) + str(time.time()) + ".csv"
       df.to_csv(df_data_name)
       print("done_train_sd"+str(trainsh_sd))
           
       
       
       
       
       
       
       
