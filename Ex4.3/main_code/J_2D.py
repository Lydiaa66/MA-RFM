import numpy as np
import cupy as cp
def cal_j(w,mat_regu,F_regu,grads_y1,grads_y2,lambda_x_y,device,cupy_device):
    cp.cuda.Device(cupy_device).use()
    grads_y1_cuda=cp.asarray(grads_y1)
    grads_y2_cuda=cp.asarray(grads_y2)
    w=cp.asarray(w)
    J_pde_cuda=cp.asarray(mat_regu)
    dS_y1_cuda=cp.dot(grads_y1_cuda,w).reshape(-1,1)
    dS_y2_cuda=cp.dot(grads_y2_cuda,w).reshape(-1,1)
    eps=1e-8
    lambda_x_y_cuda=cp.asarray(lambda_x_y)
    J_g_x_sum_cuda=cp.sqrt(lambda_x_y_cuda)*grads_y1_cuda
    J_g_y_sum_cuda=cp.sqrt(lambda_x_y_cuda)*grads_y2_cuda
    J_all_cuda=cp.concatenate((J_pde_cuda,J_g_x_sum_cuda,J_g_y_sum_cuda),axis=0)
    del grads_y1_cuda, grads_y2_cuda, w, dS_y1_cuda, dS_y2_cuda, J_pde_cuda, J_g_x_sum_cuda,J_g_y_sum_cuda
    return J_all_cuda.get()





