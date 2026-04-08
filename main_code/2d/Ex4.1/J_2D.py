import numpy as np
import cupy as cp
def cal_j(w,mat_regu,F_regu,grads_y1,grads_y2,lambda_x_y,device,cupy_device):
    cp.cuda.Device(cupy_device).use()
    grads_y1_cuda=cp.asarray(grads_y1)
    grads_y2_cuda=cp.asarray(grads_y2)
    w=cp.asarray(w)
    J_pde_cuda=cp.asarray(mat_regu)
    dS_y1_cuda=cp.dot(grads_y1_cuda,w).reshape(-1,1)  #源关于第1分量的梯度
    dS_y2_cuda=cp.dot(grads_y2_cuda,w).reshape(-1,1)
    eps=1e-8
    lambda_x_y_cuda=cp.asarray(lambda_x_y)
    J_g_sum_cuda=cp.sqrt(lambda_x_y_cuda)*(dS_y1_cuda*grads_y1_cuda+dS_y2_cuda*grads_y2_cuda)/cp.sqrt(dS_y1_cuda**2+dS_y2_cuda**2+eps)
    J_all_cuda=cp.concatenate((J_pde_cuda,J_g_sum_cuda),axis=0)
    del grads_y1_cuda, grads_y2_cuda, w, dS_y1_cuda, dS_y2_cuda, J_pde_cuda, J_g_sum_cuda
    # dS_y1=np.dot(grads_y1,w).reshape(-1,1)
    # dS_y2=np.dot(grads_y2,w).reshape(-1,1)
    #J_g_sum=np.sqrt(lambda_x_y)*(dS_y1*grads_y1+dS_y2*grads_y2)/np.sqrt(dS_y1**2+dS_y2**2+eps)
    # J_all=np.concatenate((J_pde,J_g_sum),axis=0)
    return J_all_cuda.get()
