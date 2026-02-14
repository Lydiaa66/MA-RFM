import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
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
import cupy as cp
from cupyx.scipy.special import j0, y0,j1,y1
torch.set_default_dtype(torch.float64)

from scipy.special import hankel1,h1vp

def cupy_hankel1(z): #第一类hankel函数
    return j0(z) + 1j * y0(z)
 
def cupy_dhankel1(z): #第一类hankel函数对应的导数
    return -(j1(z) + 1j * y1(z))

class GaussLegendre2D_rec:
    def __init__(self,number,x_min,x_max,y_min,y_max,x): #应该是支集上的x_min x_max y_min y_max
        self.number=number #高斯点的数量
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.Gauss_points,self.weights=np.polynomial.legendre.leggauss(self.number) #返回标准区间的高斯节点和权重
        self.y1=0.5 * (self.x_max - self.x_min) * self.Gauss_points + 0.5 * (self.x_max + self.x_min) #对标准区间的高斯点 第一分量进行变换
        self.y2= 0.5 * (self.y_max - self.y_min) *self.Gauss_points + 0.5 * (self.y_max + self.y_min) #对标准区间的高斯点 第二分量进行变换
        self.points_int=self.int_points() #被积点(经过变换后)
        self.y1_weights=0.5 * (self.x_max - self.x_min) * self.weights
        self.y2_weights = 0.5 * (self.y_max - self.y_min) * self.weights
        self.w=np.outer(self.y1_weights, self.y2_weights).flatten()
        self.X=x[:,np.newaxis,:]
        self.Y=self.points_int[np.newaxis,:,:]
        self.diff=self.X-self.Y  #有一点慢
        self.R=np.linalg.norm(self.diff,axis=2)
        self.sub_y1=((self.diff)[:,:,0])[:int(0.5*self.R.shape[0]),:]
        self.sub_y2=((self.diff)[:,:,1])[int(0.5*self.R.shape[0]):int(self.R.shape[0]),:]

    def int_points(self): #高斯积分点 也是输入点
        x_flat = self.y1.flatten()
        y_flat = self.y2.flatten()
        # 使用 meshgrid 生成网格
        X, Y = np.meshgrid(x_flat, y_flat,indexing='ij')
        # 将网格展平为点云
        samples = np.vstack((X.ravel(), Y.ravel())).T
        return samples

class forward_operator_rec(): #不用继承 源于后面对kk进行遍历的时候 对k都需要遍历
    def __init__(self,k,R,w,sub_y1,sub_y2,mea,device,cupy_device): #应该是支集上的x_min x_max y_min y_max
        self.mea=mea
        self.k=k
        self.R=R
        self.cupy_device=cupy_device
        cp.cuda.Device(self.cupy_device).use()
        self.w=cp.asarray(w)
        self.sub_y1=cp.asarray(sub_y1)
        self.sub_y2=cp.asarray(sub_y2)
        self.device=device
        R_cuda=cp.asarray(self.R)
        self.Phi=(1j/4 *cupy_hankel1(self.k*R_cuda))
        #self.Phi=1j/4 *hankel1(0,self.k*self.R)
        self.R_x1=self.R[:int(0.5*self.R.shape[0]),:] #左右边界，关于第一分量
        self.R_x2=self.R[int(0.5*self.R.shape[0]):int(self.R.shape[0]),:] #上下边界，关于第二分量
        R_x1_cuda=cp.asarray(self.R_x1)
        R_x2_cuda=cp.asarray(self.R_x2)
        self.dPhi_x1=(1j/4 *self.k*cupy_dhankel1(self.k*R_x1_cuda)/R_x1_cuda)
        self.dPhi_x2=(1j/4 *self.k*cupy_dhankel1(self.k*R_x2_cuda)/R_x2_cuda)
        # self.dPhi_x1=(1j/4 *self.k*h1vp(0,self.k*self.R_x1)/self.R_x1)
        # self.dPhi_x2=(1j/4 *self.k*h1vp(0,self.k*self.R_x2)/self.R_x2)
        
    
    def int_f(self,SS,batch_number): # 求源与基本解的积分
        w_sol=self.Phi*self.w #带权的基本解矩阵 轻微slow complex128  CUPY加速
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
                
            result=w_sol@SS #slow！
            del SS, w_sol
            
        if self.mea==0: #分块计算
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(SS.shape[0]/batch_number) #每块的行数
            result=np.zeros([w_sol.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number): #
                start_idx=i*batch_size #0
                end_idx=(i+1)*batch_size #10
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128) #SS的 
                w_sol_block=torch.tensor(w_sol[:,start_idx:end_idx]).to(self.device)
                result+=torch.mm(w_sol_block,SS_block).cpu().detach().numpy()
                # 及时释放显存
                del SS_block, w_sol_block
                torch.cuda.empty_cache()
        return result

    def int_df_x1(self,SS,batch_number): #边界点的第1分量，导数积分
        result_y1=self.dPhi_x1*self.sub_y1 #基本解关于第一分量的导数
        w_sol_y1=result_y1*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y1=w_sol_y1@SS
            del SS,w_sol_y1
        if self.mea==0: #分块计算
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(SS.shape[0]/batch_number) #每块的行数
            result_y1=np.zeros([w_sol_y1.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number): #
                start_idx=i*batch_size #0
                end_idx=(i+1)*batch_size #10
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128) #SS的 
                w_sol_block_y1=torch.tensor(w_sol_y1[:,start_idx:end_idx]).to(self.device)
                result_y1+=torch.mm(w_sol_block_y1,SS_block).cpu().detach().numpy()
                # 及时释放显存
                del SS_block, w_sol_block_y1
                torch.cuda.empty_cache()
        return result_y1

    def int_df_x2(self,SS,batch_number): #边界点的第2分量，导数积分
        result_y2=self.dPhi_x2*self.sub_y2 #基本解关于第2分量的导数
        w_sol_y2=result_y2*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y2=w_sol_y2@SS
            del SS,w_sol_y2
        if self.mea==0: #分块计算
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(SS.shape[0]/batch_number) #每块的行数
            result_y2=np.zeros([w_sol_y2.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number): #
                start_idx=i*batch_size #0
                end_idx=(i+1)*batch_size #10
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128) #SS的 
                w_sol_block_y2=torch.tensor(w_sol_y2[:,start_idx:end_idx]).to(self.device)
                result_y2+=torch.mm(w_sol_block_y2,SS_block).cpu().detach().numpy()
                # 及时释放显存
                del SS_block, w_sol_block_y2
                torch.cuda.empty_cache()
        return result_y2

def boundary_data_rec(kk,number,p_b,ana_S,x_left,x_right,y_below,y_upper,center,r,eps,mea,batch_number_rec_mea,device,cupy_device):
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
    g_rec=GaussLegendre2D_rec(number,x_left,x_right,y_below,y_upper,p_b)
    RR_rec=g_rec.R
    w=g_rec.w
    p_rec=g_rec.points_int #[y1 y2] 高斯点
    S_int_true=ana_S(p_rec,x_left,x_right,y_below,y_upper,center,r).reshape(-1,1) #内部点源在高斯点的真值
    sub_y1=g_rec.sub_y1 
    sub_y2=g_rec.sub_y2
    
    R_mea_b=[] #储存 区域边界处测量值
    R_mea_db_x1=[] #储存 区域边界导数测量值 第1分量
    R_mea_db_x2=[] #储存 区域边界导数测量值 第2分量

    for i in range(len(kk)):
        f_rec=forward_operator_rec(kk[i],RR_rec,w,sub_y1,sub_y2,mea,device,cupy_device)
        u_mea_db_x1=(f_rec.int_df_x1(S_int_true,batch_number_rec_mea)).get()#左右边界点 关于x第1分量求导的 真实观测值
        u_mea_db_x2=(f_rec.int_df_x2(S_int_true,batch_number_rec_mea)).get() #下上边界点 关于x第2分量求导的 真实观测值 
        r1 = np.random.uniform(-1, 1, size=u_mea_db_x1.shape)  #模+角度的扰动 与 u_x1 相同长度的随机数数组
        r2 = np.random.uniform(-1, 1, size=u_mea_db_x2.shape)  #模+角度的扰动 与 u_x2 相同长度的随机数数组
        abs_u_x1=np.abs(u_mea_db_x1)
        abs_u_x2=np.abs(u_mea_db_x2)
        u_per_db_x1=u_mea_db_x1+eps*r1*abs_u_x1*np.exp(1j * np.pi * r1)
        u_per_db_x2=u_mea_db_x2+eps*r2*abs_u_x2*np.exp(1j * np.pi * r2)
        R_mea_db_x1.append(u_per_db_x1) 
        R_mea_db_x2.append(u_per_db_x2) 
        u_mea_b=(f_rec.int_f(S_int_true,batch_number_rec_mea)).get() #边界点真实函数值 
        r = np.random.uniform(-1, 1, size=u_mea_b.shape)  #模+角度的扰动 与 u_x1 相同长度的随机数数组
        abs_u=np.abs(u_mea_b)
        u_per_b=u_mea_b+ eps*r*abs_u*np.exp(1j*np.pi*r)
        R_mea_b.append(u_per_b)
    F_mea_db_x1=np.concatenate((np.real(R_mea_db_x1),np.imag(R_mea_db_x1)),axis=2)
    F_mea_db_x2=np.concatenate((np.real(R_mea_db_x2),np.imag(R_mea_db_x2)),axis=2)
    F_mea_b=np.concatenate((np.real(R_mea_b),np.imag(R_mea_b)),axis=2)
    return S_int_true,F_mea_b,F_mea_db_x1,F_mea_db_x2

from scipy.special import hankel1,h1vp
class GaussLegendre2D_circle: #极坐标下的G-L积分公式
    def __init__(self,number,center,R,x): 
        self.number=number #高斯点的数量
        self.center=center
        self.R=R
        self.Gauss_points,self.weights=np.polynomial.legendre.leggauss(self.number) #返回标准区间的高斯节点和权重
        self.r=0.5 * self.R * self.Gauss_points + 0.5 * self.R #对标准区间的高斯点 r进行变换
        self.theta= 0.5 * 2*np.pi *self.Gauss_points + 0.5 * 2*np.pi #对标准区间的高斯点 \theta进行变换
        self.points_int=self.int_points() #被积点(经过变换后)
        self.y1=(self.points_int[:,0]*np.cos(self.points_int[:,1])).reshape(-1,1)+center[0]
        self.y2=(self.points_int[:,0]*np.sin(self.points_int[:,1])).reshape(-1,1)+center[1]
        self.y=np.concatenate([self.y1,self.y2],axis=1)
        self.r_weights=0.5 * self.R * self.weights
        self.theta_weights = 0.5 * 2*np.pi * self.weights
        self.w=np.outer(self.r_weights, self.theta_weights).flatten()
        self.X=x[:,np.newaxis,:]
        self.Y=self.y[np.newaxis,:,:]
        self.diff=self.X-self.Y
        self.RR=np.linalg.norm(self.diff,axis=2)
        self.sub_y1=((self.diff)[:,:,0])[:int(0.5*self.RR.shape[0]),:] #左右边界，关于第一分量
        self.sub_y2=((self.diff)[:,:,1])[int(0.5*self.RR.shape[0]):int(self.RR.shape[0]),:] #上下边界，关于第二分量

    def int_points(self): #高斯积分点 也是输入点
        r_flat = self.r.flatten()
        theta_flat = self.theta.flatten()
        # 使用 meshgrid 生成网格
        X, Y = np.meshgrid(r_flat, theta_flat,indexing='ij')
        # 将网格展平为点云
        samples = np.vstack((X.ravel(), Y.ravel())).T
        return samples

class forward_operator_circle(): #不用继承 源于后面对kk进行遍历的时候 对k都需要遍历
    def __init__(self,k,RR,w,r_points,sub_y1,sub_y2,mea,device,cupy_device): #r_points 为半径积分点的点云形式
        self.mea=mea
        self.k=k
        self.RR=RR
        self.r=cp.asarray(r_points)
        self.cupy_device=cupy_device
        cp.cuda.Device(self.cupy_device).use()
        self.w=cp.asarray(w)
        self.sub_y1=cp.asarray(sub_y1)
        self.sub_y2=cp.asarray(sub_y2)
        self.device=device
        RR_cuda=cp.asarray(self.RR)
        self.Phi=(1j/4 *cupy_hankel1(self.k*RR_cuda))
        self.R_x1=self.RR[:int(0.5*self.RR.shape[0]),:]
        self.R_x2=self.RR[int(0.5*self.RR.shape[0]):int(self.RR.shape[0]),:]
        R_x1_cuda=cp.asarray(self.R_x1)
        R_x2_cuda=cp.asarray(self.R_x2)
        self.dPhi_x1=(1j/4 *self.k*cupy_dhankel1(self.k*R_x1_cuda)/R_x1_cuda)
        self.dPhi_x2=(1j/4 *self.k*cupy_dhankel1(self.k*R_x2_cuda)/R_x2_cuda)
        #self.r=r_points
        # self.w=w
        # self.sub_y1=sub_y1
        # self.sub_y2=sub_y2
        #self.Phi=1j/4 *hankel1(0,self.k*self.RR)
        # self.dPhi_x1=(1j/4 *self.k*h1vp(0,self.k*self.R_x1)/self.R_x1)
        # self.dPhi_x2=(1j/4 *self.k*h1vp(0,self.k*self.R_x2)/self.R_x2)
    
    def int_f(self,SS,batch_number): # 求源与基本解的积分
        w_sol=self.r*self.Phi*self.w #带权的基本解矩阵 轻微slow
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
                
            result=w_sol@SS #slow！
            del SS, w_sol
            
        if self.mea==0: #分块计算
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(SS.shape[0]/batch_number) #每块的行数
            result=np.zeros([w_sol.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number): #
                start_idx=i*batch_size #0
                end_idx=(i+1)*batch_size #10
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128) #SS的 
                w_sol_block=torch.tensor(w_sol[:,start_idx:end_idx]).to(self.device)
                result+=torch.mm(w_sol_block,SS_block).cpu().detach().numpy()
                # 及时释放显存
                del SS_block, w_sol_block
                torch.cuda.empty_cache()
        return result
    
    def int_df_x1(self,SS,batch_number): #边界点的第1分量，导数积分
        result_y1=self.dPhi_x1*self.sub_y1 #基本解关于第一分量的导数
        w_sol_y1=self.r*result_y1*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y1=w_sol_y1@SS
            del SS,w_sol_y1
        if self.mea==0: #分块计算
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(SS.shape[0]/batch_number) #每块的行数
            result_y1=np.zeros([w_sol_y1.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number): #
                start_idx=i*batch_size #0
                end_idx=(i+1)*batch_size #10
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128) #SS的 
                w_sol_block_y1=torch.tensor(w_sol_y1[:,start_idx:end_idx]).to(self.device)
                result_y1+=torch.mm(w_sol_block_y1,SS_block).cpu().detach().numpy()
                # 及时释放显存
                del SS_block, w_sol_block_y1
                torch.cuda.empty_cache()
        return result_y1
    
    def int_df_x2(self,SS,batch_number): #边界点的第2分量，导数积分
        result_y2=self.dPhi_x2*self.sub_y2 #基本解关于第2分量的导数
        w_sol_y2=self.r*result_y2*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y2=w_sol_y2@SS
            del SS,w_sol_y2
        if self.mea==0: #分块计算
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
            batch_size=int(SS.shape[0]/batch_number) #每块的行数
            result_y2=np.zeros([w_sol_y2.shape[0],SS.shape[1]],dtype=np.complex128)
            for i in range(batch_number): #
                start_idx=i*batch_size #0
                end_idx=(i+1)*batch_size #10
                SS_block=(SS[start_idx:end_idx,:].to(self.device)).to(torch.complex128) #SS的 
                w_sol_block_y2=torch.tensor(w_sol_y2[:,start_idx:end_idx]).to(self.device)
                result_y2+=torch.mm(w_sol_block_y2,SS_block).cpu().detach().numpy()
                # 及时释放显存
                del SS_block, w_sol_block_y2
                torch.cuda.empty_cache()
        return result_y2

def boundary_data_circle(kk,number,p_b,ana_S,x_left,x_right,y_below,y_upper,center,r,eps,mea,batch_number_circle_mea,device,cupy_device):
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
    w=g_circle.w #高斯点权
    p_polar=g_circle.points_int #[r,theta] 极坐标
    p_y=g_circle.y #高斯点 [y1,y2]
    S_int_true=ana_S(p_y,x_left,x_right,y_below,y_upper,center,r).reshape(-1,1) #内部点源在高斯点的真值
    sub_y1=g_circle.sub_y1 
    sub_y2=g_circle.sub_y2
    r_points=p_polar[:,0] #r
    R_mea_b=[] #储存 区域边界处测量值
    R_mea_db_x1=[] #储存 区域边界导数测量值 第1分量
    R_mea_db_x2=[] #储存 区域边界导数测量值 第2分量

    for i in range(len(kk)):
        f_circle=forward_operator_circle(kk[i],RR_circle,w,r_points,sub_y1,sub_y2,mea,device,cupy_device)
        u_mea_db_x1=(f_circle.int_df_x1(S_int_true,batch_number_circle_mea)).get() #左右边界点 关于x第1分量求导的 真实观测值
        u_mea_db_x2=(f_circle.int_df_x2(S_int_true,batch_number_circle_mea)).get() #下上边界点 关于x第2分量求导的 真实观测值
        r1 = np.random.uniform(-1, 1, size=u_mea_db_x1.shape)  #模+角度的扰动 与 u_x1 相同长度的随机数数组
        r2 = np.random.uniform(-1, 1, size=u_mea_db_x2.shape)  #模+角度的扰动 与 u_x2 相同长度的随机数数组
        abs_u_x1=np.abs(u_mea_db_x1)
        abs_u_x2=np.abs(u_mea_db_x2)
        u_per_db_x1=u_mea_db_x1+eps*r1*abs_u_x1*np.exp(1j * np.pi * r1)
        u_per_db_x2=u_mea_db_x2+eps*r2*abs_u_x2*np.exp(1j * np.pi * r2)
        R_mea_db_x1.append(u_per_db_x1) 
        R_mea_db_x2.append(u_per_db_x2) 
        u_mea_b=(f_circle.int_f(S_int_true,batch_number_circle_mea)).get() #边界点真实函数值 
        r = np.random.uniform(-1, 1, size=u_mea_b.shape)  #模+角度的扰动 与 u_x1 相同长度的随机数数组
        abs_u=np.abs(u_mea_b)
        u_per_b=u_mea_b+ eps*r*abs_u*np.exp(1j*np.pi*r)
        R_mea_b.append(u_per_b)
    F_mea_db_x1=np.concatenate((np.real(R_mea_db_x1),np.imag(R_mea_db_x1)),axis=2)
    F_mea_db_x2=np.concatenate((np.real(R_mea_db_x2),np.imag(R_mea_db_x2)),axis=2)
    F_mea_b=np.concatenate((np.real(R_mea_b),np.imag(R_mea_b)),axis=2)
    return S_int_true,F_mea_b,F_mea_db_x1,F_mea_db_x2


