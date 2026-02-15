import torch
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

from scipy.special import hankel1,h1vp
class GaussLegendre2D_rec:
    def __init__(self,number,x_min,x_max,y_min,y_max,z_min,z_max,x):
        self.number=number
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.z_min=z_min
        self.z_max=z_max
        self.Gauss_points,self.weights=np.polynomial.legendre.leggauss(self.number)
        self.y1=0.5 * (self.x_max - self.x_min) * self.Gauss_points + 0.5 * (self.x_max + self.x_min)
        self.y2= 0.5 * (self.y_max - self.y_min) *self.Gauss_points + 0.5 * (self.y_max + self.y_min)
        self.y3= 0.5 * (self.z_max - self.z_min) *self.Gauss_points + 0.5 * (self.z_max + self.z_min)
        self.points_int=self.int_points()
        self.y1_weights=0.5 * (self.x_max - self.x_min) * self.weights
        self.y2_weights = 0.5 * (self.y_max - self.y_min) * self.weights
        self.y3_weights = 0.5 * (self.z_max - self.z_min) * self.weights
        self.w_y1_y2=np.outer(self.y1_weights, self.y2_weights).flatten()
        self.w=np.outer(self.w_y1_y2,self.y3_weights).flatten()
        self.X=x[:,np.newaxis,:]
        self.Y=self.points_int[np.newaxis,:,:]
        self.diff=self.X-self.Y
        self.R=np.linalg.norm(self.diff,axis=2)
        self.sub_y1=((self.diff)[:,:,0])[:int(2/6*self.R.shape[0]),:]
        self.sub_y2=((self.diff)[:,:,1])[int(2/6*self.R.shape[0]):int(4/6*self.R.shape[0]),:]
        self.sub_y3=((self.diff)[:,:,2])[int(4/6*self.R.shape[0]):int(self.R.shape[0]),:]

    def int_points(self):
        x_flat = self.y1.flatten()
        y_flat = self.y2.flatten()
        z_flat = self.y3.flatten()
        X, Y , Z= np.meshgrid(x_flat, y_flat,z_flat,indexing='ij')
        samples = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        return samples

class forward_operator_rec():
    def __init__(self,k,R,w,sub_y1,sub_y2,sub_y3,mea,device,cupy_device):
        self.mea=mea
        self.device=device
        self.k=k
        self.R=R
        self.w=w
        self.sub_y1=sub_y1
        self.sub_y2=sub_y2
        self.sub_y3=sub_y3
        self.Phi=np.exp(1j*self.k *self.R)/(4*np.pi*self.R)
        self.R_x1=self.R[:int(2/6*self.R.shape[0]),:]
        self.R_x2=self.R[int(2/6*self.R.shape[0]):int(4/6*self.R.shape[0]),:]
        self.R_x3=self.R[int(4/6*self.R.shape[0]):int(self.R.shape[0]),:]
        self.dPhi_x1=np.exp(1j*k*self.R_x1)*((1j*self.k*self.R_x1-1)/(4*np.pi*self.R_x1**2))/self.R_x1
        self.dPhi_x2=np.exp(1j*k*self.R_x2)*((1j*self.k*self.R_x2-1)/(4*np.pi*self.R_x2**2))/self.R_x2
        self.dPhi_x3=np.exp(1j*k*self.R_x3)*((1j*self.k*self.R_x3-1)/(4*np.pi*self.R_x3**2))/self.R_x3
        
    def int_f(self,SS,batch_number):
        w_sol=self.Phi*self.w
        if self.mea==1:
            SS=SS.cpu().detach().numpy()
            result=w_sol@SS
        if self.mea==0:
            batch_size=int(SS.shape[0]/batch_number)
            result=np.zeros([w_sol.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128)
                w_sol_block=torch.tensor(w_sol[:,start_idx:end_idx]).to(self.device)
                result+=torch.mm(w_sol_block,SS_block).cpu().detach().numpy()
                del SS_block, w_sol_block
                torch.cuda.empty_cache()
        if self.mea==5:
            SS=SS.to(self.device)
            batch_size=int(w_sol.shape[0]/batch_number)
            result=np.zeros([w_sol.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result

    def int_df_x1(self,SS,batch_number):
        result_y1=self.dPhi_x1*self.sub_y1
        w_sol_y1=result_y1*self.w
        if self.mea==1:
            SS=SS.cpu().detach().numpy()
            result_y1=w_sol_y1@SS
        if self.mea==0:
            batch_size=int(SS.shape[0]/batch_number)
            result_y1=np.zeros([w_sol_y1.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128)
                w_sol_block_y1=torch.tensor(w_sol_y1[:,start_idx:end_idx]).to(self.device)
                result_y1+=torch.mm(w_sol_block_y1,SS_block).cpu().detach().numpy()
                del SS_block, w_sol_block_y1
                torch.cuda.empty_cache()
        if self.mea==5:
            SS=SS.to(self.device)
            batch_size=int(w_sol_y1.shape[0]/batch_number)
            result_y1=np.zeros([w_sol_y1.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol_y1[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result_y1[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result_y1

    def int_df_x2(self,SS,batch_number):
        result_y2=self.dPhi_x2*self.sub_y2
        w_sol_y2=result_y2*self.w
        if self.mea==1:
            SS=SS.cpu().detach().numpy()
            result_y2=w_sol_y2@SS
        if self.mea==0:
            batch_size=int(SS.shape[0]/batch_number)
            result_y2=np.zeros([w_sol_y2.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128)
                w_sol_block_y2=torch.tensor(w_sol_y2[:,start_idx:end_idx]).to(self.device)
                result_y2+=torch.mm(w_sol_block_y2,SS_block).cpu().detach().numpy()
                del SS_block, w_sol_block_y2
                torch.cuda.empty_cache()
                
        if self.mea==5:
            SS=SS.to(self.device)
            batch_size=int(w_sol_y2.shape[0]/batch_number)
            result_y2=np.zeros([w_sol_y2.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol_y2[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result_y2[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result_y2
    
    def int_df_x3(self,SS,batch_number):
        result_y3=self.dPhi_x3*self.sub_y3
        w_sol_y3=result_y3*self.w
        if self.mea==1:
            SS=SS.cpu().detach().numpy()
            result_y3=w_sol_y3@SS
        if self.mea==0:
            batch_size=int(SS.shape[0]/batch_number)
            result_y3=np.zeros([w_sol_y3.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128)
                w_sol_block_y3=torch.tensor(w_sol_y3[:,start_idx:end_idx]).to(self.device)
                result_y3+=torch.mm(w_sol_block_y3,SS_block).cpu().detach().numpy()
                del SS_block, w_sol_block_y3
                torch.cuda.empty_cache()
                
        if self.mea==5:
            SS=SS.to(self.device)
            batch_size=int(w_sol_y3.shape[0]/batch_number)
            result_y3=np.zeros([w_sol_y3.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol_y3[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result_y3[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result_y3

def boundary_data_rec(kk,number,p_b,ana_S,x_left,x_right,y_below,y_upper,z_min,z_max,center1,center2,r,eps,mea,batch_number_rec_mea,device,cupy_device):
    """
    Args:
        number: 高斯点的数量
        tau_x_min (float64),tau_x_max (float64): 支集
        tau_y_min (float64),tau_y_max (float64): 支集
        p_b (narray): 边界点 左右下上
        ana_S(function): 真实源函数
        center(array):源的中心
        r(array):源的半径
        eps(float64): 扰动
    """
    g_rec=GaussLegendre2D_rec(number,x_left,x_right,y_below,y_upper,z_min,z_max,p_b)
    RR_rec=g_rec.R
    w=g_rec.w
    p_rec=g_rec.points_int
    S_int_true=ana_S(p_rec,x_left,x_right,y_below,y_upper,z_min,z_max,center1,center2,r).reshape(-1,1)
    sub_y1=g_rec.sub_y1 
    sub_y2=g_rec.sub_y2
    sub_y3=g_rec.sub_y3
    
    R_mea_b=[]
    R_mea_db_x1=[]
    R_mea_db_x2=[]
    R_mea_db_x3=[]
    for i in range(len(kk)):
        f_rec=forward_operator_rec(kk[i],RR_rec,w,sub_y1,sub_y2,sub_y3,mea)
        u_mea_db_x1=f_rec.int_df_x1(S_int_true,batch_number_rec_mea,device)
        u_mea_db_x2=f_rec.int_df_x2(S_int_true,batch_number_rec_mea,device)
        u_mea_db_x3=f_rec.int_df_x3(S_int_true,batch_number_rec_mea,device)
        r1 = np.random.uniform(-1, 1, size=u_mea_db_x1.shape)
        r2 = np.random.uniform(-1, 1, size=u_mea_db_x2.shape)
        r3 = np.random.uniform(-1, 1, size=u_mea_db_x3.shape)
        abs_u_x1=np.abs(u_mea_db_x1)
        abs_u_x2=np.abs(u_mea_db_x2)
        abs_u_x3=np.abs(u_mea_db_x3)
        u_per_db_x1=u_mea_db_x1+eps*r1*abs_u_x1*np.exp(1j * np.pi * r1)
        u_per_db_x2=u_mea_db_x2+eps*r2*abs_u_x2*np.exp(1j * np.pi * r2)
        u_per_db_x3=u_mea_db_x3+eps*r3*abs_u_x3*np.exp(1j * np.pi * r3)
        R_mea_db_x1.append(u_per_db_x1) 
        R_mea_db_x2.append(u_per_db_x2) 
        R_mea_db_x3.append(u_per_db_x3)
        u_mea_b=f_rec.int_f(S_int_true)
        r = np.random.uniform(-1, 1, size=u_mea_b.shape)
        abs_u=np.abs(u_mea_b)
        u_per_b=u_mea_b+ eps*r*abs_u*np.exp(1j*np.pi*r)
        R_mea_b.append(u_per_b)
    F_mea_db_x1=np.concatenate((np.real(R_mea_db_x1),np.imag(R_mea_db_x1)),axis=2)
    F_mea_db_x2=np.concatenate((np.real(R_mea_db_x2),np.imag(R_mea_db_x2)),axis=2)
    F_mea_db_x3=np.concatenate((np.real(R_mea_db_x3),np.imag(R_mea_db_x3)),axis=2)
    F_mea_b=np.concatenate((np.real(R_mea_b),np.imag(R_mea_b)),axis=2)
    return S_int_true,F_mea_b,F_mea_db_x1,F_mea_db_x2,F_mea_db_x3

from scipy.special import hankel1,h1vp
class GaussLegendre2D_circle:
    def __init__(self,number,center,R,x): 
        self.number=number
        self.center=center
        self.R=R
        self.Gauss_points,self.weights=np.polynomial.legendre.leggauss(self.number)
        self.r=0.5 * self.R * self.Gauss_points + 0.5 * self.R
        self.theta= 0.5 * np.pi *self.Gauss_points + 0.5 * np.pi
        self.phi= 0.5 * 2 *np.pi *self.Gauss_points + 0.5 * 2 *np.pi 
        self.points_int=self.int_points()
        self.y1=(self.points_int[:,0]*np.sin(self.points_int[:,1])*np.cos(self.points_int[:,2])).reshape(-1,1)+center[0]
        self.y2=(self.points_int[:,0]*np.sin(self.points_int[:,1])*np.sin(self.points_int[:,2])).reshape(-1,1)+center[1]
        self.y3=(self.points_int[:,0]*np.cos(self.points_int[:,1])).reshape(-1,1)+center[2]
        self.y=np.concatenate([self.y1,self.y2,self.y3],axis=1)
        self.r_weights=0.5 * self.R * self.weights
        self.theta_weights = 0.5 * np.pi * self.weights
        self.phi_weights = 0.5 * 2*np.pi * self.weights
        self.w_y1_y2=np.outer(self.r_weights, self.theta_weights).flatten()
        self.w=np.outer(self.w_y1_y2,self.phi_weights).flatten()
        self.X=x[:,np.newaxis,:]
        self.Y=self.y[np.newaxis,:,:]
        self.diff=self.X-self.Y
        self.RR=np.linalg.norm(self.diff,axis=2)
        self.sub_y1=((self.diff)[:,:,0])[:int(2/6*self.RR.shape[0]),:]
        self.sub_y2=((self.diff)[:,:,1])[int(2/6*self.RR.shape[0]):int(4/6*self.RR.shape[0]),:]
        self.sub_y3=((self.diff)[:,:,2])[int(4/6*self.RR.shape[0]):int(self.RR.shape[0]),:]

    def int_points(self):
        r_flat = self.r.flatten()
        theta_flat = self.theta.flatten()
        phi_flat = self.phi.flatten()
        R, Theta, Phi = np.meshgrid(r_flat, theta_flat, phi_flat, indexing='ij')
    
        samples = np.vstack((R.ravel(), Theta.ravel(), Phi.ravel())).T
        return samples

class forward_operator_circle():
    def __init__(self,k,RR,w,r_points,theta_points,sub_y1,sub_y2,sub_y3,mea):
        self.mea=mea
        self.k=k
        self.RR=RR
        self.w=w
        self.r=r_points
        self.theta=theta_points
        self.sub_y1=sub_y1
        self.sub_y2=sub_y2
        self.sub_y3=sub_y3
        self.Phi=np.exp(1j*self.k *self.RR)/(4*np.pi*self.RR)
        self.R_x1=self.RR[:int(2/6*self.RR.shape[0]),:]
        self.R_x2=self.RR[int(2/6*self.RR.shape[0]):int(4/6*self.RR.shape[0]),:]
        self.R_x3=self.RR[int(4/6*self.RR.shape[0]):int(self.RR.shape[0]),:]
        self.dPhi_x1=np.exp(1j*k*self.R_x1)*((1j*self.k*self.R_x1-1)/(4*np.pi*self.R_x1**2))/self.R_x1
        self.dPhi_x2=np.exp(1j*k*self.R_x2)*((1j*self.k*self.R_x2-1)/(4*np.pi*self.R_x2**2))/self.R_x2
        self.dPhi_x3=np.exp(1j*k*self.R_x3)*((1j*self.k*self.R_x3-1)/(4*np.pi*self.R_x3**2))/self.R_x3
    
    def int_f(self,SS,batch_number):
        w_sol=self.r**2*np.sin(self.theta)*self.Phi*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            result=w_sol@SS
        else:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(w_sol.shape[0]/batch_number)
            result=np.zeros([w_sol.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result
    
    def int_df_x1(self,SS,batch_number):
        result_y1=self.dPhi_x1*self.sub_y1
        w_sol_y1=self.r**2*np.sin(self.theta)*result_y1*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            result_y1=w_sol_y1@SS
        else:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(w_sol_y1.shape[0]/batch_number)
            result_y1=np.zeros([w_sol_y1.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol_y1[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result_y1[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result_y1

    def int_df_x2(self,SS,batch_number):
        result_y2=self.dPhi_x2*self.sub_y2
        w_sol_y2=self.r**2*np.sin(self.theta)*result_y2*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            result_y2=w_sol_y2@SS
        else:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(w_sol_y2.shape[0]/batch_number)
            result_y2=np.zeros([w_sol_y2.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size

                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol_y2[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result_y2[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result_y2
    
    def int_df_x3(self,SS,batch_number):
        result_y3=self.dPhi_x3*self.sub_y3
        w_sol_y3=self.r**2*np.sin(self.theta)*result_y3*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            result_y3=w_sol_y3@SS
        else:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device)  
            batch_size=int(w_sol_y3.shape[0]/batch_number)
            result_y3=np.zeros([w_sol_y3.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number):
                start_idx=i*batch_size
                end_idx=(i+1)*batch_size
                batch_data=torch.tensor(w_sol_y3[start_idx:end_idx]).to(self.device)
                batch_output=torch.mm(batch_data,SS.to(torch.complex128))
                result_y3[start_idx:end_idx] = batch_output.cpu().detach().numpy()
                del batch_data,batch_output
        return result_y3
    
def boundary_data_circle(kk,number,p_b,ana_S,x_left,x_right,y_below,y_upper,z_min,z_max,center,r,eps,mea,batch_number_circle_mea,device,cupy_device):
    """
    Args:
        number: 高斯点的数量
        p_b (narray): 边界观测点 左右下上
        ana_S(function): 真实源函数
        center(array):源的中心
        r(array):源的半径
        eps(float64): 扰动
    """
    g_circle=GaussLegendre2D_circle(number,center,r,p_b)
    RR_circle=g_circle.RR
    w=g_circle.w
    p_polar=g_circle.points_int
    p_y=g_circle.y
    S_int_true=ana_S(p_y,x_left,x_right,y_below,y_upper,z_min,z_max,center,r).reshape(-1,1)
    sub_y1=g_circle.sub_y1 
    sub_y2=g_circle.sub_y2
    sub_y3=g_circle.sub_y3
    r_points=p_polar[:,0]
    theta_points=p_polar[:,1]
    R_mea_b=[]
    R_mea_db_x1=[]
    R_mea_db_x2=[]
    R_mea_db_x3=[]
    for i in range(len(kk)):
        f_circle=forward_operator_circle(kk[i],RR_circle,w,r_points,theta_points,sub_y1,sub_y2,sub_y3,mea)
        u_mea_db_x1=f_circle.int_df_x1(S_int_true,batch_number_circle_mea)
        u_mea_db_x2=f_circle.int_df_x2(S_int_true,batch_number_circle_mea)
        u_mea_db_x3=f_circle.int_df_x3(S_int_true,batch_number_circle_mea)
        r1 = np.random.uniform(-1, 1, size=u_mea_db_x1.shape)
        r2 = np.random.uniform(-1, 1, size=u_mea_db_x2.shape)
        r3 = np.random.uniform(-1, 1, size=u_mea_db_x3.shape)
        abs_u_x1=np.abs(u_mea_db_x1)
        abs_u_x2=np.abs(u_mea_db_x2)
        abs_u_x3=np.abs(u_mea_db_x3)
        u_per_db_x1=u_mea_db_x1+eps*r1*abs_u_x1*np.exp(1j * np.pi * r1)
        u_per_db_x2=u_mea_db_x2+eps*r2*abs_u_x2*np.exp(1j * np.pi * r2)
        u_per_db_x3=u_mea_db_x3+eps*r3*abs_u_x3*np.exp(1j * np.pi * r3)
        R_mea_db_x1.append(u_per_db_x1) 
        R_mea_db_x2.append(u_per_db_x2) 
        R_mea_db_x3.append(u_per_db_x3) 
        u_mea_b=f_circle.int_f(S_int_true,batch_number_circle_mea)
        r = np.random.uniform(-1, 1, size=u_mea_b.shape)
        abs_u=np.abs(u_mea_b)
        u_per_b=u_mea_b+ eps*r*abs_u*np.exp(1j*np.pi*r)
        R_mea_b.append(u_per_b)
    F_mea_db_x1=np.concatenate((np.real(R_mea_db_x1),np.imag(R_mea_db_x1)),axis=2)
    F_mea_db_x2=np.concatenate((np.real(R_mea_db_x2),np.imag(R_mea_db_x2)),axis=2)
    F_mea_db_x3=np.concatenate((np.real(R_mea_db_x3),np.imag(R_mea_db_x3)),axis=2)
    F_mea_b=np.concatenate((np.real(R_mea_b),np.imag(R_mea_b)),axis=2)
    return S_int_true,F_mea_b,F_mea_db_x1,F_mea_db_x2,F_mea_db_x3


