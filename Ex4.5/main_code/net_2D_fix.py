import time
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
torch.set_default_dtype(torch.float64)

def weights_init(m, R_m,b_below,b_upper):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            if m.in_features == 1:
                m.weight[0, 0] = 1.0
                if m.weight.shape[0] > 1:
                    m.weight[1:, :] = 1.0
                
                if m.bias.shape[0] > 1:
                    m.bias[:] = torch.empty(m.bias.shape[0]).uniform_(b_below,b_upper )
                
            else:
                nn.init.uniform_(m.weight, a = -R_m, b = R_m)
                nn.init.uniform_(m.bias, a = -R_m, b = R_m)


class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, y_max, y_min,r_low,r_upper,device):
        super(local_rep, self).__init__()
        self.device=device
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.M = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(y_max - y_min)]).to(device)
        self.x_0 = torch.tensor([(x_max + x_min)/2,(y_max + y_min)/2]).to(device)
        self.hidden_layer= nn.Sequential(nn.Linear(in_features, self.hidden_features, bias=True))
        self.hidden_layer_1 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))
        self.hidden_layer_2 = nn.Sequential(nn.Linear(1, self.hidden_features, bias=True))
        self.r_values = np.random.uniform(low=r_low, high=r_upper, size=self.M)

    def forward(self,x,af):
        y = self.a * (x - self.x_0)
        y = self.hidden_layer(y)
        if af==0:
            y=torch.tanh(y)
        elif af=="Tanh":
            y=torch.tanh(y)
        elif af=="sigmoid":
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[...,1:2])
            r_hat= torch.sqrt(y1**2+y2**2)

            r= torch.tensor(np.tile(self.r_values, (y1.shape[0], 1))).to(self.device) 
            y=(torch.sigmoid(2000*(r-r_hat)))
            del r
        elif af==1:
            y=torch.sin(y)
        elif af=="Relu":
            y1 = self.hidden_layer_1(y[..., 0:1])
            y2 = self.hidden_layer_2(y[...,1:2])
            r_hat= torch.sqrt(y1**2+y2**2)
            y=(torch.relu(0.4-r_hat))**3
        else:
            y = torch.sin(y)*torch.cos(y) 
        return y
    


def pre_define(Nx, Ny,M,R_m,tau_x_min,tau_x_max,tau_y_min,tau_y_max,r_low,r_upper,b_low,b_upper,device):
    models = []
    for k in range(Nx):
        model_for_x = []
        x_min = (tau_x_max-tau_x_min)/Nx * k+tau_x_min
        x_max = (tau_x_max-tau_x_min)/Nx * (k+1)+tau_x_min
        for n in range(Ny):
            y_min = (tau_y_max-tau_y_min)/Ny * n+tau_y_min
            y_max = (tau_y_max-tau_y_min)/Ny * (n+1)+tau_y_min
            model = local_rep(in_features = 2, out_features = 2, hidden_layers = 1, M = M, x_min = x_min, 
                              x_max = x_max, y_min = y_min, y_max = y_max,r_low=r_low,r_upper=r_upper,device=device).to(device)
            model.apply(lambda m: weights_init(m, R_m=R_m,b_below=b_low,b_upper=b_upper))
            model = model.double().to(device)
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
        models.append(model_for_x)
    return models