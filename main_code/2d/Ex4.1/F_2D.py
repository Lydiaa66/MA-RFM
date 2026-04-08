import numpy as np
import cupy as cp
def cal_f(w,mat_regu,F_regu,grads_y1,grads_y2,lambda_x_y,device,cupy_device):
    cp.cuda.Device(cupy_device).use()
    grads_y1_cuda=cp.asarray(grads_y1)
    grads_y2_cuda=cp.asarray(grads_y2)
    w=cp.asarray(w)
    mat_regu_cuda=cp.asarray(mat_regu)
    dS_y1_cuda=cp.dot(grads_y1_cuda,w).reshape(-1,1)  #源关于第1分量的梯度
    dS_y2_cuda=cp.dot(grads_y2_cuda,w).reshape(-1,1)
    F_regu_cuda=cp.asarray(F_regu)
    loss_pde_regu_cuda=(cp.dot(mat_regu_cuda,w)-F_regu_cuda.reshape(-1)).reshape(-1,1)  
    eps=1e-8
    lambda_x_y_cuda=cp.asarray(lambda_x_y)
    loss_g_sum_cuda=cp.sqrt(lambda_x_y_cuda)*cp.sqrt(dS_y1_cuda**2+dS_y2_cuda**2)  #源梯度的模的平方
    #loss_g_sum_cuda=cp.sqrt(lambda_x_y_cuda)*(dS_y1_cuda**2+dS_y2_cuda**2)**(0.25)  #源梯度的模的平方
    loss_cuda= cp.concatenate((loss_pde_regu_cuda,loss_g_sum_cuda),axis=0).reshape(-1)
    del grads_y1_cuda, grads_y2_cuda, w, dS_y1_cuda, dS_y2_cuda, loss_pde_regu_cuda, loss_g_sum_cuda
    return loss_cuda.get()

# import numpy as np
# import cupy as cp
# def cal_f(w,mat_regu,F_regu,grads_y1,grads_y2,lambda_x_y,device,cupy_device):
#     cp.cuda.Device(cupy_device).use()
#     grads_y1_cuda=cp.asarray(grads_y1)
#     grads_y2_cuda=cp.asarray(grads_y2)
#     w=cp.asarray(w)
#     mat_regu_cuda=cp.asarray(mat_regu)
#     dS_y1_cuda=cp.dot(grads_y1_cuda,w).reshape(-1,1)  #源关于第1分量的梯度
#     dS_y2_cuda=cp.dot(grads_y2_cuda,w).reshape(-1,1)
#     F_regu_cuda=cp.asarray(F_regu)
#     loss_pde_regu_cuda=(cp.dot(mat_regu_cuda,w)-F_regu_cuda.reshape(-1)).reshape(-1,1)  
#     eps=1e-8
#     lambda_x_y_cuda=cp.asarray(lambda_x_y)
#     loss_g_sum_cuda=cp.sqrt(lambda_x_y_cuda)*cp.sqrt(dS_y1_cuda**2+dS_y2_cuda**2+eps)  #源梯度的模的平方
#     loss_cuda= cp.concatenate((loss_pde_regu_cuda,loss_g_sum_cuda),axis=0).reshape(-1)
#     del grads_y1_cuda, grads_y2_cuda, w, dS_y1_cuda, dS_y2_cuda, loss_pde_regu_cuda, loss_g_sum_cuda
#     return loss_cuda.get()