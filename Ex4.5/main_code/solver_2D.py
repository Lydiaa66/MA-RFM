import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import F_2D,J_2D,set_matrix_2D

def L_curve(M,b_matrix,db_x1_matrix,db_x2_matrix,f_mea_b,f_mea_db_x1,f_mea_db_x2,lamb_regu,grads_y1,grads_y2,lambda_x_y,device,cupy_device):
    iter=len(lamb_regu)
    res=np.zeros([1,iter])
    reg_norm=np.zeros([1,iter])
    w_store=np.zeros([M,iter])
    for i in range(iter):
        mat,F,mat_regu,F_regu=set_matrix_2D.set(b_matrix,db_x1_matrix,db_x2_matrix,f_mea_b,f_mea_db_x1,f_mea_db_x2,lamb_regu[i],device,cupy_device)
        x0=np.random.normal(0,10,M)
        w = least_squares(F_2D.cal_f,x0,J_2D.cal_j,args=(mat_regu,F_regu,grads_y1,grads_y2,lambda_x_y,device,cupy_device),ftol=1e-6,gtol=1e-6,xtol=1e-6,verbose=2)
        w_x=(w.x).reshape(-1,1)
        res[:,i]= np.linalg.norm(mat@w_x-F)
        reg_norm[:,i]= np.linalg.norm(w_x)
        w_store[:,i:i+1]=w_x
    plt.figure(figsize=(8, 6))
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
            