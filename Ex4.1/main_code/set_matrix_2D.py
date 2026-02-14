from scipy.linalg import lstsq,pinv
import numpy as np
from scipy.linalg import block_diag
import scipy
import matplotlib.pyplot as plt
import cupy as cp

def set(b_matrix,db_x1_matrix,db_x2_matrix,F_mea_b,F_mea_db_x1,F_mea_db_x2,lamb_regu,device,cupy_device):
    #内部点+边界点
    # cp.cuda.Device(cupy_device).use()
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
    F_mea_b=F_mea_b.reshape(-1,F_mea_b.shape[-1])
    F_mea_db_x1=F_mea_db_x1.reshape(-1,F_mea_db_x1.shape[-1])
    F_mea_db_x2=F_mea_db_x2.reshape(-1,F_mea_db_x2.shape[-1])
    F=np.concatenate((F_mea_b[:,0:1],F_mea_b[:,1:],F_mea_db_x1[:,0:1],F_mea_db_x1[:,1:],F_mea_db_x2[:,0:1],F_mea_db_x2[:,1:]),axis=0)
    mat=np.concatenate((A_b,B_b,A_db_x1,B_db_x1,A_db_x2,B_db_x2),axis=0)
    regu=lamb_regu*np.eye(A_b.shape[1])
    mat_regu=np.concatenate((mat,regu),axis=0)
    F_regu=np.concatenate((F,lamb_regu*np.zeros([A_b.shape[1],1])),axis=0)
    # mat_regu_cuda=cp.asarray(mat_regu)
    # F_regu_cuda=cp.asarray(F_regu)
    # w_regu=cp.linalg.lstsq(mat_regu_cuda,F_regu_cuda)
    # del mat_regu_cuda,F_regu_cuda
    return mat,F,mat_regu,F_regu

