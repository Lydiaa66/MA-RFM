from scipy.linalg import lstsq,pinv
import numpy as np
from scipy.linalg import block_diag
import scipy
import matplotlib.pyplot as plt
import cupy as cp

import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.preamble'] = ''

def A(b_matrix,db_x1_matrix,db_x2_matrix):
    r_b=np.real(b_matrix)
    i_b=np.imag(b_matrix)
    r_db_x1=np.real(db_x1_matrix)
    i_db_x1=np.imag(db_x1_matrix)
    r_db_x2=np.real(db_x2_matrix)
    i_db_x2=np.imag(db_x2_matrix)
    A_b=r_b.reshape(-1,r_b.shape[-1])
    B_b=i_b.reshape(-1,i_b.shape[-1])
    A_db_x1=r_db_x1.reshape(-1,r_db_x1.shape[-1])
    B_db_x1=i_db_x1.reshape(-1,i_db_x1.shape[-1])
    A_db_x2=r_db_x2.reshape(-1,r_db_x2.shape[-1])
    B_db_x2=i_db_x2.reshape(-1,i_db_x2.shape[-1])
    mat=np.concatenate((A_b,B_b,A_db_x1,B_db_x1,A_db_x2,B_db_x2),axis=0)

    return mat

def F(F_mea_b,F_mea_db_x1,F_mea_db_x2):
    F_mea_b=F_mea_b.reshape(-1,F_mea_b.shape[-1])
    F_mea_db_x1=F_mea_db_x1.reshape(-1,F_mea_db_x1.shape[-1])
    F_mea_db_x2=F_mea_db_x2.reshape(-1,F_mea_db_x2.shape[-1])
    F=np.concatenate((F_mea_b[:,0:1],F_mea_b[:,1:],F_mea_db_x1[:,0:1],F_mea_db_x1[:,1:],F_mea_db_x2[:,0:1],F_mea_db_x2[:,1:]),axis=0)
    return F

def L_curve(M0,M1,M2,af,mat,F,lamb_regu,cupy_device):
    iter=len(lamb_regu)
    res=np.zeros([1,iter])
    reg_norm=np.zeros([1,iter])
    if af=="mix":
        n=M0+M1+M2
    else:
        n=M0
    w_store=np.zeros([n,iter])
    cp.cuda.Device(cupy_device).use()
    for i in range(iter):
        regu=lamb_regu[i]*np.eye(n)
        mat_regu=np.concatenate((mat,regu),axis=0)
        F_regu=np.concatenate((F,lamb_regu[i]*np.zeros([n,1])),axis=0)
        mat_regu_cuda=cp.asarray(mat_regu)
        F_regu_cuda=cp.asarray(F_regu)
        w_regu=cp.linalg.lstsq(mat_regu_cuda,F_regu_cuda)[0]
        w=w_regu.reshape(-1,1).get()
        w_store[:,i:i+1]=w
        res[:,i]= np.linalg.norm(mat@w-F)
        reg_norm[:,i]= np.linalg.norm(w)
    plt.figure(figsize=(6, 4))
    log_res=np.log10(res.reshape(-1))
    log_reg_norm=np.log10(reg_norm.reshape(-1))
    plt.plot(log_res,log_reg_norm , marker='o', linestyle='-', color='b', label="Regularization Path")
    for i in range(iter):
        plt.text(log_res[i]+0.0000001, log_reg_norm[i] - 0.001,   f"{i}: {lamb_regu[i]:.2e}", fontsize=12, ha='center', color='red')
    plt.xlabel("Residual Norm (||Ax - y||)")
    plt.ylabel("Regularization Norm (||w||)")
    plt.title("Regularization Path (Residual Norm vs Regularization Norm)")
    plt.grid(True)
    plt.show()
    return res,reg_norm,w_store