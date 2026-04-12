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
import cupy as cp
from cupyx.scipy.special import j0, y0,j1,y1
torch.set_default_dtype(torch.float64)

from scipy.special import hankel1,h1vp

def cupy_hankel1(z):
    return j0(z) + 1j * y0(z)
 
def cupy_dhankel1(z):
    return -(j1(z) + 1j * y1(z))


def _is_radial_boundary(points):
    if len(points) == 0:
        return False
    radii = np.linalg.norm(points, axis=1)
    return np.max(radii) > 0 and np.std(radii) <= 1e-10 * np.max(radii) + 1e-12


def dir_neu(N_m, N_m_inv, kk, R, rho, n_trun, F_mea_b):
    F_mea_b_complex = F_mea_b[..., 0] + 1j * F_mea_b[..., 1]

    theta = np.linspace(0, 2 * np.pi, N_m, endpoint=False)
    n_vals = np.arange(-n_trun, n_trun + 1)
    exp_term = np.exp(-1j * n_vals[:, np.newaxis] * theta[np.newaxis, :])
    U_fourier = (1 / N_m) * np.sum(F_mea_b_complex[:, np.newaxis, :] * exp_term[np.newaxis, :, :], axis=2)

    hankel_ratio = np.zeros((len(kk), 2 * n_trun + 1), dtype=complex)
    dhankel_ratio = np.zeros((len(kk), 2 * n_trun + 1), dtype=complex)
    for i, n in enumerate(n_vals):
        h_R = hankel1(n, kk * R)
        h_rho = hankel1(n, kk * rho)
        dh_rho = h1vp(n, kk * rho)
        hankel_ratio[:, i] = h_rho / h_R
        dhankel_ratio[:, i] = dh_rho / h_R

    cof = U_fourier * hankel_ratio
    kk_grid = np.repeat(kk[:, np.newaxis], repeats=dhankel_ratio.shape[1], axis=1)
    d_cof = kk_grid * U_fourier * dhankel_ratio

    theta_inv = np.linspace(0, 2 * np.pi, N_m_inv, endpoint=False)
    inv_exp_term = np.exp(1j * n_vals[:, np.newaxis] * theta_inv[np.newaxis, :])

    v = np.sum(cof[:, :, np.newaxis] * inv_exp_term[np.newaxis, :, :], axis=1)
    v_rho = np.zeros([kk.shape[0], N_m_inv, 2])
    v_rho[:, :, 0] = np.real(v)
    v_rho[:, :, 1] = np.imag(v)

    dv = np.sum(d_cof[:, :, np.newaxis] * inv_exp_term[np.newaxis, :, :], axis=1)
    dv_rho = np.zeros([kk.shape[0], N_m_inv, 2])
    dv_rho[:, :, 0] = np.real(dv)
    dv_rho[:, :, 1] = np.imag(dv)

    return v_rho, dv_rho

class GaussLegendre2D_rec:
    def __init__(self,number,x_min,x_max,y_min,y_max,x):
        self.number=number
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max
        self.Gauss_points,self.weights=np.polynomial.legendre.leggauss(self.number)
        self.y1=0.5 * (self.x_max - self.x_min) * self.Gauss_points + 0.5 * (self.x_max + self.x_min)
        self.y2= 0.5 * (self.y_max - self.y_min) *self.Gauss_points + 0.5 * (self.y_max + self.y_min)
        self.points_int=self.int_points()
        self.y1_weights=0.5 * (self.x_max - self.x_min) * self.weights
        self.y2_weights = 0.5 * (self.y_max - self.y_min) * self.weights
        self.w=np.outer(self.y1_weights, self.y2_weights).flatten()
        self.X=x[:,np.newaxis,:]
        self.Y=self.points_int[np.newaxis,:,:]
        self.diff=self.X-self.Y
        self.R=np.linalg.norm(self.diff,axis=2)
        self.sub_y1=((self.diff)[:,:,0])[:int(0.5*self.R.shape[0]),:]
        self.sub_y2=((self.diff)[:,:,1])[int(0.5*self.R.shape[0]):int(self.R.shape[0]),:]

    def int_points(self):
        x_flat = self.y1.flatten()
        y_flat = self.y2.flatten()
        X, Y = np.meshgrid(x_flat, y_flat,indexing='ij')
        samples = np.vstack((X.ravel(), Y.ravel())).T
        return samples

class forward_operator_rec():
    def __init__(self,k,R,w,sub_y1,sub_y2,mea,device,cupy_device):
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
        self.R_x1=self.R[:int(0.5*self.R.shape[0]),:]
        self.R_x2=self.R[int(0.5*self.R.shape[0]):int(self.R.shape[0]),:]
        R_x1_cuda=cp.asarray(self.R_x1)
        R_x2_cuda=cp.asarray(self.R_x2)
        self.dPhi_x1=(1j/4 *self.k*cupy_dhankel1(self.k*R_x1_cuda)/R_x1_cuda)
        self.dPhi_x2=(1j/4 *self.k*cupy_dhankel1(self.k*R_x2_cuda)/R_x2_cuda)
        
    
    def int_f(self,SS,batch_number):
        w_sol=self.Phi*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
                
            result=w_sol@SS
            del SS, w_sol
            
        if self.mea==0:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
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
        return result

    def int_df_x1(self,SS,batch_number):
        result_y1=self.dPhi_x1*self.sub_y1
        w_sol_y1=result_y1*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y1=w_sol_y1@SS
            del SS,w_sol_y1
        if self.mea==0:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
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
        return result_y1

    def int_df_x2(self,SS,batch_number):
        result_y2=self.dPhi_x2*self.sub_y2
        w_sol_y2=result_y2*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y2=w_sol_y2@SS
            del SS,w_sol_y2
        if self.mea==0:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
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
        return result_y2


class GaussLegendre2D_rec_normal:
    def __init__(self, number, x_min, x_max, y_min, y_max, x):
        self.number = number
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.Gauss_points, self.weights = np.polynomial.legendre.leggauss(self.number)
        self.y1 = 0.5 * (self.x_max - self.x_min) * self.Gauss_points + 0.5 * (self.x_max + self.x_min)
        self.y2 = 0.5 * (self.y_max - self.y_min) * self.Gauss_points + 0.5 * (self.y_max + self.y_min)
        self.points_int = self.int_points()
        self.y1_weights = 0.5 * (self.x_max - self.x_min) * self.weights
        self.y2_weights = 0.5 * (self.y_max - self.y_min) * self.weights
        self.w = np.outer(self.y1_weights, self.y2_weights).flatten()
        self.X = x[:, np.newaxis, :]
        self.r = np.sqrt(x[0, 0] ** 2 + x[0, 1] ** 2)
        self.normal = np.tile(self.X, (1, self.points_int.shape[0], 1)) / self.r
        self.Y = self.points_int[np.newaxis, :, :]
        self.diff = self.X - self.Y
        self.R = np.linalg.norm(self.diff, axis=2)

    def int_points(self):
        x_flat = self.y1.flatten()
        y_flat = self.y2.flatten()
        X, Y = np.meshgrid(x_flat, y_flat, indexing="ij")
        samples = np.vstack((X.ravel(), Y.ravel())).T
        return samples


class forward_operator_rec_normal:
    def __init__(self, k, R, w, diff, normal, mea, device, cupy_device):
        self.mea = mea
        self.k = k
        self.R = R
        self.cupy_device = cupy_device
        cp.cuda.Device(self.cupy_device).use()
        self.w = cp.asarray(w)
        self.diff = cp.asarray(diff)
        self.normal = cp.asarray(normal)
        self.device = device
        R_cuda = cp.asarray(self.R)
        self.Phi = 1j / 4 * cupy_hankel1(self.k * R_cuda)
        self.dPhi = (1j / 4 * self.k * cupy_dhankel1(self.k * R_cuda) / R_cuda) * np.sum(self.diff * self.normal, axis=2)

    def int_f(self, SS, batch_number):
        w_sol = self.Phi * self.w
        if self.mea == 1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result = w_sol @ SS
            del SS, w_sol
            return result

        if isinstance(SS, np.ndarray):
            SS = torch.tensor(SS).to(self.device)
        batch_size = int(SS.shape[0] / batch_number)
        result = np.zeros([w_sol.shape[0], SS.shape[1]], dtype=np.complex128)
        for i in range(batch_number):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            SS_block = (SS[start_idx:end_idx, :].to(self.device)).to(torch.complex128)
            w_sol_block = torch.tensor(w_sol[:, start_idx:end_idx]).to(self.device)
            result += torch.mm(w_sol_block, SS_block).cpu().detach().numpy()
            del SS_block, w_sol_block
            torch.cuda.empty_cache()
        return result

    def int_df(self, SS, batch_number):
        result_y1 = self.dPhi
        w_sol_y1 = result_y1 * self.w
        if self.mea == 1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y1 = w_sol_y1 @ SS
            del SS, w_sol_y1
            return result_y1

        if isinstance(SS, np.ndarray):
            SS = torch.tensor(SS).to(self.device)
        batch_size = int(SS.shape[0] / batch_number)
        result_y1 = np.zeros([w_sol_y1.shape[0], SS.shape[1]], dtype=np.complex128)
        for i in range(batch_number):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            SS_block = (SS[start_idx:end_idx, :].to(self.device)).to(torch.complex128)
            w_sol_block_y1 = torch.tensor(w_sol_y1[:, start_idx:end_idx]).to(self.device)
            result_y1 += torch.mm(w_sol_block_y1, SS_block).cpu().detach().numpy()
            del SS_block, w_sol_block_y1
            torch.cuda.empty_cache()
        return result_y1

def boundary_data_rec(kk,number,p_b,ana_S,x_left,x_right,y_below,y_upper,center,r,eps,mea,batch_number_rec_mea,device,cupy_device):
    """
    Args:
        number: Number of Gauss points.
        tau_x_min (float64), tau_x_max (float64): Support interval in x.
        tau_y_min (float64), tau_y_max (float64): Support interval in y.
        p_b (ndarray): Boundary points on the left, right, bottom, and top sides.
        ana_S (function): Exact source function.
        center (array): Source center.
        r (array): Source radius.
        eps (float64): Perturbation level.
    """
    if _is_radial_boundary(p_b):
        g_rec = GaussLegendre2D_rec_normal(number, x_left, x_right, y_below, y_upper, p_b)
        RR_rec = g_rec.R
        w = g_rec.w
        p_rec = g_rec.points_int
        S_int_true = ana_S(p_rec, x_left, x_right, y_below, y_upper, center, r).reshape(-1, 1)
        diff = g_rec.diff
        normal = g_rec.normal

        R_mea_b = []
        R_mea_db_x = []

        for i in range(len(kk)):
            f_rec = forward_operator_rec_normal(kk[i], RR_rec, w, diff, normal, mea, device, cupy_device)
            u_mea_db = (f_rec.int_df(S_int_true, batch_number_rec_mea)).get()
            rand = np.random.uniform(-1, 1, size=u_mea_db.shape)
            abs_du = np.abs(u_mea_db)
            u_per_db = u_mea_db + eps * rand * abs_du * np.exp(1j * np.pi * rand)
            R_mea_db_x.append(u_per_db)

            u_mea_b = (f_rec.int_f(S_int_true, batch_number_rec_mea)).get()
            rand = np.random.uniform(-1, 1, size=u_mea_b.shape)
            abs_u = np.abs(u_mea_b)
            u_per_b = u_mea_b + eps * rand * abs_u * np.exp(1j * np.pi * rand)
            R_mea_b.append(u_per_b)

        F_mea_db = np.concatenate((np.real(R_mea_db_x), np.imag(R_mea_db_x)), axis=2)
        F_mea_b = np.concatenate((np.real(R_mea_b), np.imag(R_mea_b)), axis=2)
        return S_int_true, F_mea_b, F_mea_db

    g_rec=GaussLegendre2D_rec(number,x_left,x_right,y_below,y_upper,p_b)
    RR_rec=g_rec.R
    w=g_rec.w
    p_rec=g_rec.points_int
    S_int_true=ana_S(p_rec,x_left,x_right,y_below,y_upper,center,r).reshape(-1,1)
    sub_y1=g_rec.sub_y1 
    sub_y2=g_rec.sub_y2
    
    R_mea_b=[]
    R_mea_db_x1=[]
    R_mea_db_x2=[]

    for i in range(len(kk)):
        f_rec=forward_operator_rec(kk[i],RR_rec,w,sub_y1,sub_y2,mea,device,cupy_device)
        u_mea_db_x1=(f_rec.int_df_x1(S_int_true,batch_number_rec_mea)).get()
        u_mea_db_x2=(f_rec.int_df_x2(S_int_true,batch_number_rec_mea)).get()
        r1 = np.random.uniform(-1, 1, size=u_mea_db_x1.shape)
        r2 = np.random.uniform(-1, 1, size=u_mea_db_x2.shape)
        abs_u_x1=np.abs(u_mea_db_x1)
        abs_u_x2=np.abs(u_mea_db_x2)
        u_per_db_x1=u_mea_db_x1+eps*r1*abs_u_x1*np.exp(1j * np.pi * r1)
        u_per_db_x2=u_mea_db_x2+eps*r2*abs_u_x2*np.exp(1j * np.pi * r2)
        R_mea_db_x1.append(u_per_db_x1) 
        R_mea_db_x2.append(u_per_db_x2) 
        u_mea_b=(f_rec.int_f(S_int_true,batch_number_rec_mea)).get()
        r = np.random.uniform(-1, 1, size=u_mea_b.shape)
        abs_u=np.abs(u_mea_b)
        u_per_b=u_mea_b+ eps*r*abs_u*np.exp(1j*np.pi*r)
        R_mea_b.append(u_per_b)
    F_mea_db_x1=np.concatenate((np.real(R_mea_db_x1),np.imag(R_mea_db_x1)),axis=2)
    F_mea_db_x2=np.concatenate((np.real(R_mea_db_x2),np.imag(R_mea_db_x2)),axis=2)
    F_mea_b=np.concatenate((np.real(R_mea_b),np.imag(R_mea_b)),axis=2)
    return S_int_true,F_mea_b,F_mea_db_x1,F_mea_db_x2

from scipy.special import hankel1,h1vp
class GaussLegendre2D_circle:
    def __init__(self, number, center, R, *args):
        self.number = number
        self.center = center
        self.R = R
        if len(args) == 1:
            self.a = None
            x = args[0]
        elif len(args) == 2:
            self.a = args[0]
            x = args[1]
        else:
            raise TypeError("Unsupported GaussLegendre2D_circle signature")

        self.Gauss_points, self.weights = np.polynomial.legendre.leggauss(self.number)
        self.r = 0.5 * self.R * self.Gauss_points + 0.5 * self.R
        self.theta = 0.5 * 2 * np.pi * self.Gauss_points + 0.5 * 2 * np.pi
        self.points_int = self.int_points()

        if self.a is None:
            self.y1 = (self.points_int[:, 0] * np.cos(self.points_int[:, 1])).reshape(-1, 1) + center[0]
            self.y2 = (self.points_int[:, 0] * np.sin(self.points_int[:, 1])).reshape(-1, 1) + center[1]
        else:
            self.y1 = (
                self.points_int[:, 0] * self.a * (3 * np.cos(self.points_int[:, 1]) - np.cos(3 * self.points_int[:, 1]))
            ).reshape(-1, 1) + center[0]
            self.y2 = (
                self.points_int[:, 0] * self.a * (3 * np.sin(self.points_int[:, 1]) - np.sin(3 * self.points_int[:, 1]))
            ).reshape(-1, 1) + center[1]

        self.y = np.concatenate([self.y1, self.y2], axis=1)
        self.r_weights = 0.5 * self.R * self.weights
        self.theta_weights = 0.5 * 2 * np.pi * self.weights
        self.w = np.outer(self.r_weights, self.theta_weights).flatten()
        self.X = x[:, np.newaxis, :]
        self.Y = self.y[np.newaxis, :, :]
        self.diff = self.X - self.Y
        self.RR = np.linalg.norm(self.diff, axis=2)
        self.sub_y1 = ((self.diff)[:, :, 0])[: int(0.5 * self.RR.shape[0]), :]
        self.sub_y2 = ((self.diff)[:, :, 1])[int(0.5 * self.RR.shape[0]) : int(self.RR.shape[0]), :]

    def int_points(self):
        r_flat = self.r.flatten()
        theta_flat = self.theta.flatten()
        X, Y = np.meshgrid(r_flat, theta_flat,indexing='ij')
        samples = np.vstack((X.ravel(), Y.ravel())).T
        return samples

class forward_operator_circle():
    def __init__(self, k, *args):
        self.mea = None
        self.a = None
        self.phi = None
        if len(args) == 8:
            RR, w, r_points, sub_y1, sub_y2, mea, device, cupy_device = args
        elif len(args) == 10:
            self.a, RR, w, r_points, phi_points, sub_y1, sub_y2, mea, device, cupy_device = args
            self.phi = cp.asarray(phi_points)
        else:
            raise TypeError("Unsupported forward_operator_circle signature")

        self.mea = mea
        self.k = k
        self.RR = RR
        self.r = cp.asarray(r_points)
        self.cupy_device = cupy_device
        cp.cuda.Device(self.cupy_device).use()
        self.w = cp.asarray(w)
        self.sub_y1 = cp.asarray(sub_y1)
        self.sub_y2 = cp.asarray(sub_y2)
        self.device = device
        RR_cuda = cp.asarray(self.RR)
        self.Phi = (1j / 4 * cupy_hankel1(self.k * RR_cuda))
        self.R_x1 = self.RR[: int(0.5 * self.RR.shape[0]), :]
        self.R_x2 = self.RR[int(0.5 * self.RR.shape[0]) : int(self.RR.shape[0]), :]
        R_x1_cuda = cp.asarray(self.R_x1)
        R_x2_cuda = cp.asarray(self.R_x2)
        self.dPhi_x1 = (1j / 4 * self.k * cupy_dhankel1(self.k * R_x1_cuda) / R_x1_cuda)
        self.dPhi_x2 = (1j / 4 * self.k * cupy_dhankel1(self.k * R_x2_cuda) / R_x2_cuda)

    def _geom_factor(self):
        factor = self.r
        if self.a is not None and self.phi is not None:
            factor = factor * 12 * self.a**2 * (1 - cp.cos(2 * self.phi))
        return factor
    
    def int_f(self,SS,batch_number):
        w_sol=self._geom_factor()*self.Phi*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
                
            result=w_sol@SS
            del SS, w_sol
            
        if self.mea==0:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
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
        return result
    
    def int_df_x1(self,SS,batch_number):
        result_y1=self.dPhi_x1*self.sub_y1
        w_sol_y1=self._geom_factor()*result_y1*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y1=w_sol_y1@SS
            del SS,w_sol_y1
        if self.mea==0:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
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
        return result_y1
    
    def int_df_x2(self,SS,batch_number):
        result_y2=self.dPhi_x2*self.sub_y2
        w_sol_y2=self._geom_factor()*result_y2*self.w
        if self.mea==1:
            if isinstance(SS, torch.Tensor):
                SS = SS.detach().cpu().numpy()
            SS = cp.asarray(SS)
            result_y2=w_sol_y2@SS
            del SS,w_sol_y2
        if self.mea==0:
            if isinstance(SS, np.ndarray):
                SS = torch.tensor(SS).to(self.device) 
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
        return result_y2

def boundary_data_circle(kk,number,p_b,ana_S,x_left,x_right,y_below,y_upper,center,r,*args):
    """
    Args:
        number: Number of Gauss points.
        p_b (ndarray): Boundary observation points on the left, right, bottom, and top sides.
        ana_S (function): Exact source function.
        center (array): Source center.
        r (array): Source radius.
        eps (float64): Perturbation level.
    """
    if len(args) == 5:
        a = None
        eps, mea, batch_number_circle_mea, device, cupy_device = args
    elif len(args) == 6:
        a, eps, mea, batch_number_circle_mea, device, cupy_device = args
    else:
        raise TypeError("Unsupported generate_data.boundary_data_circle signature")

    if a is None:
        g_circle=GaussLegendre2D_circle(number,center,r,p_b)
    else:
        g_circle=GaussLegendre2D_circle(number,center,r,a,p_b)
    RR_circle=g_circle.RR
    w=g_circle.w
    p_polar=g_circle.points_int
    p_y=g_circle.y
    S_int_true=ana_S(p_y,x_left,x_right,y_below,y_upper,center,r).reshape(-1,1)
    sub_y1=g_circle.sub_y1 
    sub_y2=g_circle.sub_y2
    r_points=p_polar[:,0]
    R_mea_b=[]
    R_mea_db_x1=[]
    R_mea_db_x2=[]

    for i in range(len(kk)):
        if a is None:
            f_circle=forward_operator_circle(kk[i],RR_circle,w,r_points,sub_y1,sub_y2,mea,device,cupy_device)
        else:
            phi_points=p_polar[:,1]
            f_circle=forward_operator_circle(kk[i],a,RR_circle,w,r_points,phi_points,sub_y1,sub_y2,mea,device,cupy_device)
        u_mea_db_x1=(f_circle.int_df_x1(S_int_true,batch_number_circle_mea)).get()
        u_mea_db_x2=(f_circle.int_df_x2(S_int_true,batch_number_circle_mea)).get()
        r1 = np.random.uniform(-1, 1, size=u_mea_db_x1.shape)
        r2 = np.random.uniform(-1, 1, size=u_mea_db_x2.shape)
        abs_u_x1=np.abs(u_mea_db_x1)
        abs_u_x2=np.abs(u_mea_db_x2)
        u_per_db_x1=u_mea_db_x1+eps*r1*abs_u_x1*np.exp(1j * np.pi * r1)
        u_per_db_x2=u_mea_db_x2+eps*r2*abs_u_x2*np.exp(1j * np.pi * r2)
        R_mea_db_x1.append(u_per_db_x1) 
        R_mea_db_x2.append(u_per_db_x2) 
        u_mea_b=(f_circle.int_f(S_int_true,batch_number_circle_mea)).get()
        r = np.random.uniform(-1, 1, size=u_mea_b.shape)
        abs_u=np.abs(u_mea_b)
        u_per_b=u_mea_b+ eps*r*abs_u*np.exp(1j*np.pi*r)
        R_mea_b.append(u_per_b)
    F_mea_db_x1=np.concatenate((np.real(R_mea_db_x1),np.imag(R_mea_db_x1)),axis=2)
    F_mea_db_x2=np.concatenate((np.real(R_mea_db_x2),np.imag(R_mea_db_x2)),axis=2)
    F_mea_b=np.concatenate((np.real(R_mea_b),np.imag(R_mea_b)),axis=2)
    return S_int_true,F_mea_b,F_mea_db_x1,F_mea_db_x2
