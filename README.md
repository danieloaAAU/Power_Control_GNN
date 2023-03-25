# Power_Control_GNN
These codes are for results presented in the paper - Power Control for 6G Industrial Wireless Subnetworks: A Graph Neural Network Approach
https://arxiv.org/abs/2212.14051

# About the code
## How to use the code
Download all the files in the same folder

The file main.py contains the run_experiment(f_metric, train_num_subn, test_num_subn, trainsh_sd, testsh_sd, device) function. Run this function and call the function in a console with appropriate arguments, or import main in a python file and run main.run_experiment(f_metric, train_num_subn, test_num_subn, trainsh_sd, testsh_sd, device).

f_metric options - hH, dD, hD :The type of graph attribute; hH - use full channel gain, dD - use only distance information, hD - use desired link channel gain and  interfering link distances  
train_num_subn - number of subnetworks in training deployment - choose between values 20, 25, 10, for larger values, there is a need to increase self.deploy_length
test_num_subn  - number of subnetworks in testing deployments - choose between values 20, 25, 10, for larger values, there is a need to increase self.deploy_length
trainsh_sd - the shadowing standard deviation of the training deployments - reasonable values between 4 and 10
testsh_sd - the shadowing standard deviation of the testing environment - reasonable values between 4 and 10
device - Choose appropriate device - 'cuda', 'cpu'

The results are returned as plots with appropriate titles.

## Dependencies
Pytorch, pytorch_geometric, numpy, matplotlib, Sum_rate_power_allocator, subnetwork_generator

You can view the evaluation for the WMMSE in the WMMSE_eval.ipynb
