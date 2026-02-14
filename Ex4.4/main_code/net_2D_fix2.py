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

def weights_init(m, R_m,b_min,b_max):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            if m.in_features == 1:
                m.weight[0, 0] = 1.0
                if m.weight.shape[0] > 1:
                    m.weight[1:, :] = 1.0
                
                if m.bias.shape[0] > 1:
                    m.bias[:] = torch.empty(m.bias.shape[0]).uniform_(b_min,b_max)
                
            else:
                nn.init.uniform_(m.weight, a = -R_m, b = R_m)
                nn.init.uniform_(m.bias, a = -R_m, b = R_m)


class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_min, x_max, y_min, y_max,r_min,r_max,b1_min,b1_max,b2_min,b2_max,peak_min,peak_max,v_min,v_max,K_min,K_max,R_m_for_init,af,device):
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
        self.af=af
        if self.af=="mix":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            weights_init(self.hidden_layer[0], R_m=R_m_for_init, b_min=-R_m_for_init, b_max=R_m_for_init) 
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif self.af=="Tanh":
            weights_init(self.hidden_layer[0], R_m=R_m_for_init, b_min=-R_m_for_init, b_max=R_m_for_init) 
        elif self.af=="sigmoid":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
        elif self.af=="Gauss":
            weights_init(self.hidden_layer_1[0], R_m=R_m_for_init, b_min=b1_min, b_max=b1_max)
            weights_init(self.hidden_layer_2[0], R_m=R_m_for_init, b_min=b2_min, b_max=b2_max)
            self.r_values = torch.empty(self.M, dtype=torch.float64).uniform_(r_min, r_max).to(self.device)
            self.K = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)
            self.gauss_amp = torch.empty(self.M, dtype=torch.float64).uniform_(peak_min, peak_max).to(self.device)
            self.gauss_scale_exp = torch.empty(self.M, dtype=torch.float64).uniform_(v_min, v_max).to(self.device)
            self.gauss_scale_sig = torch.empty(self.M, dtype=torch.float64).uniform_(K_min, K_max).to(self.device)

            
    
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
            r= self.r_values.repeat(y1.shape[0],1)
            K=self.K.repeat(y1.shape[0],1)
            y=(torch.sigmoid(K*(r-r_hat)))
            del r
        elif af=="Gauss":
            y1 = self.hidden_layer_1(x[..., 0:1])
            y2 = self.hidden_layer_2(x[...,1:2])
            r_hat_sq = y1**2 + y2**2
            r_hat = torch.sqrt(r_hat_sq)
            r= self.r_values.repeat(y1.shape[0],1)
            K=self.K.repeat(y1.shape[0],1)
            gauss_amp_b = self.gauss_amp.repeat(y1.shape[0],1)
            gauss_scale_exp_b = self.gauss_scale_exp.repeat(y1.shape[0],1)
            gauss_scale_sig_b = self.gauss_scale_sig.repeat(y1.shape[0],1)
            y = gauss_amp_b * torch.exp(gauss_scale_exp_b * r_hat_sq) * \
                (torch.sigmoid(gauss_scale_sig_b * (r**2- r_hat_sq)))
            return y 
            
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
    


def pre_define(Nx, Ny,M,R_m,af,tau_x_min,tau_x_max,tau_y_min,tau_y_max,r_min,r_max,b1_min,b1_max,b2_min,b2_max,peak_min,peak_max,v_min,v_max,K_min,K_max,device):
    models = []
    for k in range(Nx):
        model_for_x = []
        x_min = (tau_x_max-tau_x_min)/Nx * k+tau_x_min
        x_max = (tau_x_max-tau_x_min)/Nx * (k+1)+tau_x_min
        for n in range(Ny):
            y_min = (tau_y_max-tau_y_min)/Ny * n+tau_y_min
            y_max = (tau_y_max-tau_y_min)/Ny * (n+1)+tau_y_min
            model = local_rep(in_features = 2, out_features = 2, hidden_layers = 1, M = M, x_min = x_min, 
                              x_max = x_max, y_min = y_min, y_max = y_max,r_min=r_min,r_max=r_max,b1_min=b1_min,b1_max=b1_max,b2_min=b2_min,b2_max=b2_max,peak_min=peak_min,peak_max=peak_max,v_min=v_min,v_max=v_max,K_min=K_min,K_max=K_max,R_m_for_init=R_m,af=af,device=device).to(device)
            model = model.double().to(device)
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
        models.append(model_for_x)
    return models