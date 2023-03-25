# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:56:04 2022

@author: Daniel Abode

This code implements the functions for the power control GNN algorithm

References:
D. Abode, R. Adeogun, and G. Berardinelli, “Power control for 6g industrial wireless subnetworks: A graph neural network approach,”
2022. [Online]. Available: https://arxiv.org/abs/2212.14051  
"""

import numpy as np                         
import torch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid


def create_features(dist_matrix, power_matrix):
    K = power_matrix.shape[1]
    mask = np.eye(K)
    mask = np.expand_dims(mask,axis=0)
    mask_1 = 1 - mask
    rcv_power = np.multiply(mask, power_matrix)
    int_dist_matrix = np.multiply(mask_1, dist_matrix)
    feature = rcv_power + int_dist_matrix
    return feature

def normalize_data(train_data, test_data):
    Nt = 1
    train_K = train_data.shape[1]
    test_K = test_data.shape[1]
    train_layouts = train_data.shape[0]
    tmp_mask = np.eye(train_K)
    mask = tmp_mask
    mask = np.expand_dims(mask,axis=0)
    
    train_copy = np.copy(train_data)
    diag_H = np.multiply(mask,train_copy)
    diag_mean = np.sum(diag_H/Nt)/train_layouts/train_K
    diag_var = np.sqrt(np.sum(np.square(diag_H))/train_layouts/train_K/Nt)
    tmp_diag = (diag_H - diag_mean)/diag_var

    off_diag = train_copy - diag_H 
    off_diag_mean = np.sum(off_diag/Nt)/train_layouts/train_K/(train_K-1)
    off_diag_var = np.sqrt(np.sum(np.square(off_diag))/Nt/train_layouts/train_K/(train_K-1))
    tmp_off = (off_diag - off_diag_mean)/off_diag_var 
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask) 
    
    norm_train = np.multiply(tmp_diag,mask) + tmp_off_diag 
    
    tmp_mask = np.eye(test_K)
    mask = tmp_mask
    mask = np.expand_dims(mask,axis=0)
    
    test_copy = np.copy(test_data)
    diag_H = np.multiply(mask,test_copy)
    tmp_diag = (diag_H - diag_mean)/diag_var
    
    off_diag = test_copy - diag_H
    tmp_off = (off_diag - off_diag_mean)/off_diag_var
    tmp_off_diag = tmp_off - np.multiply(tmp_off,mask)
    
    norm_test = np.multiply(tmp_diag,mask) + tmp_off_diag
    return norm_train, norm_test

def create_graph_list(features, powers):
    Graph_list = []
    for i in range(features.shape[0]):
        feature1 = features[i,:,:]
        mask = np.eye(feature1.shape[0])

        nodes_feature1 = np.sum(mask * feature1, axis=1)
        
        edges_features1 = (1-mask) * feature1

        nodes_features_ = np.concatenate((np.ones_like(np.expand_dims(nodes_feature1,-1)),np.expand_dims(nodes_feature1, -1)), axis=1)
        
        nodes_features = torch.tensor(nodes_features_,dtype=torch.float)
        
        edges_features1 = (1-mask) * feature1
        
        edges = torch.tensor(np.transpose(np.argwhere(edges_features1)), dtype=torch.long)
        
        edges_features1_ = np.expand_dims(edges_features1[np.nonzero(edges_features1)],-1) 
        
        edges_features = torch.tensor(edges_features1_,dtype=torch.float)
        
        graph = Data(nodes_features, edges, edges_features, y=torch.tensor(powers[i],dtype=torch.float))
        
        Graph_list.append(graph)
        
    return Graph_list

class NNConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(NNConv, self).__init__(aggr='mean', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        
    def update(self, aggr_out, x): 
        tmp = torch.cat([x, aggr_out], dim=1) 
        comb = self.mlp2(tmp)
        
        return torch.cat([comb, x[:,1:3]],dim=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) 

    def message(self, x_i, x_j, edge_attr): 
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)
    
class PCGNN(torch.nn.Module):
    def __init__(self):
        super(PCGNN, self).__init__()
        self.mlp1 = Seq(Lin(3,32), Lin(32,32),  Lin(32,32),  ReLU())
        self.mlp2 = Seq(Lin(34,32),  Lin(32,16), Lin(16,1), Sigmoid())
        self.conv = NNConv(self.mlp1,self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x = x0, edge_index = edge_index, edge_attr = edge_attr)
        x2 = self.conv(x = x1, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        return out
    
def myloss2(out, data, batch_size, num_subnetworks,Noise_power, device):
    out = out.reshape([-1,num_subnetworks])

    out = out.reshape([-1,num_subnetworks,1,1])
    power_mat = data.y.reshape([-1,num_subnetworks,num_subnetworks,1])
    weighted_powers = torch.mul(out,power_mat)
    eye = torch.eye(num_subnetworks).to(device)
    desired_rcv_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),eye), dim=1)
    
    Interference_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),1-eye), dim=1)
    signal_interference_ratio = torch.divide(desired_rcv_power,Interference_power+Noise_power)
    capacity = torch.log2(1+signal_interference_ratio)
    Capacity_ = torch.mean(torch.sum(capacity, axis=1))
    
    return torch.neg(Capacity_/num_subnetworks)

def train(model2, train_loader, optimizer, num_of_subnetworks, Noise_power, device):
    model2.train()
    total_loss = 0
    count = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model2(data)
        loss = myloss2(out[:,0].to(device), data, data.num_graphs,num_of_subnetworks, Noise_power, device)
        total_loss += loss.item()
        count = count+1
        loss.backward()
        optimizer.step()
        
    total = total_loss / count   
    return total

def test(model2,validation_loader, num_of_subnetworks, Noise_power, device):
    model2.eval()
    total_loss = 0
    count = 0
    for data in validation_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model2(data)
            loss = myloss2(out[:,0].to('cuda'), data, data.num_graphs,num_of_subnetworks,Noise_power, device)
            total_loss += loss.item()
            count = count+1
    total = total_loss / count
    print('power weight for 1 snapshot \n', out[0:20,0])
    
    return total

def trainmodel(name, model2, scheduler, train_loader, validation_loader, optimizer, num_of_subnetworks, Noise_power, device):
    loss_ = []
    losst_ = []
    for epoch in range(1,1500):
        losst = train(model2, train_loader, optimizer, num_of_subnetworks, Noise_power, device)
        loss1 = test(model2,validation_loader, num_of_subnetworks, Noise_power, device)
        loss_.append(loss1)
        losst_.append(losst)
        if (loss1 == min(loss_)):
            torch.save(model2, str(name))
        print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
            epoch, losst, loss1))
        scheduler.step()
    return loss_, losst_

def mycapacity(weights, data, batch_size, num_subnetworks, Noise_power):

    weights = weights.reshape([-1,num_subnetworks,1,1])
    
    power_mat = data.y.reshape([-1,num_subnetworks,num_subnetworks,1])

    weighted_powers = torch.mul(weights,power_mat)
    
    eye = torch.eye(num_subnetworks)
    
    desired_rcv_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),eye), dim=1)
   
    Interference_power = torch.sum(torch.mul(weighted_powers.squeeze(-1),1-eye), dim=1)

    signal_interference_ratio = torch.divide(desired_rcv_power,Interference_power+Noise_power)
    
    capacity = torch.log2(1+signal_interference_ratio)
    
    return capacity, weighted_powers 

def GNN_test(GNNmodel, test_loader, num_of_subnetworks, Noise_power,device):    
    model2 = torch.load(GNNmodel)
    model2.eval()
    capacities = torch.Tensor()
    GNN_powers = torch.Tensor() 
    GNN_weights = torch.Tensor() 
    GNN_sum_rate = torch.Tensor()
    Pmax = 1
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model2(data)
            cap, GNN_pow = mycapacity(Pmax*out[:,0].cpu(), data.cpu(), data.num_graphs,num_of_subnetworks, Noise_power)        
        GNN_powers = torch.cat((GNN_powers, GNN_pow.cpu()),0)
        GNN_weights = torch.cat((GNN_weights, out[:,0].cpu()),0)
        capacities = torch.cat((capacities,cap.cpu()),0)
        GNN_sum_rate = torch.cat((GNN_sum_rate,torch.sum(cap,1)),0)
        
    return GNN_sum_rate, capacities, GNN_weights, GNN_powers

def generate_cdf(values, bins_):
    data = np.array(values)
    count, bins_count = np.histogram(data, bins=bins_)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return bins_count[1:], cdf

def findcdfvalue(x,y,yval1,yval2):
    a = x[np.logical_and(y>yval1, y<yval2)]
    if a.size < 1:
        return 0
    else:
        m = np.mean(a)
        return m.item()