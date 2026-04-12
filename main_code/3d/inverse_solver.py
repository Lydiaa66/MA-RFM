from scipy.linalg import lstsq,pinv
import numpy as np
from scipy.linalg import block_diag
import scipy
import matplotlib.pyplot as plt
import cupy as cp


def A(b_matrix,db_x1_matrix,db_x2_matrix,db_x3_matrix,condition="Cauchy"):
    if condition=="Cauchy":
        r_b=np.real(b_matrix)
        i_b=np.imag(b_matrix)
        r_db_x1=np.real(db_x1_matrix)
        i_db_x1=np.imag(db_x1_matrix)
        r_db_x2=np.real(db_x2_matrix)
        i_db_x2=np.imag(db_x2_matrix)
        r_db_x3=np.real(db_x3_matrix)
        i_db_x3=np.imag(db_x3_matrix)
        A_b=r_b.reshape(-1,r_b.shape[-1])
        B_b=i_b.reshape(-1,i_b.shape[-1])
        A_db_x1=r_db_x1.reshape(-1,r_db_x1.shape[-1])
        B_db_x1=i_db_x1.reshape(-1,i_db_x1.shape[-1])
        A_db_x2=r_db_x2.reshape(-1,r_db_x2.shape[-1])
        B_db_x2=i_db_x2.reshape(-1,i_db_x2.shape[-1])
        A_db_x3=r_db_x3.reshape(-1,r_db_x3.shape[-1])
        B_db_x3=i_db_x3.reshape(-1,i_db_x3.shape[-1])
        mat=np.concatenate((A_b,B_b,A_db_x1,B_db_x1,A_db_x2,B_db_x2,A_db_x3,B_db_x3),axis=0)
    if condition=="Dirichlet":
        r_b=np.real(b_matrix)
        i_b=np.imag(b_matrix)
        A_b=r_b.reshape(-1,r_b.shape[-1])
        B_b=i_b.reshape(-1,i_b.shape[-1])
        mat=np.concatenate((A_b,B_b),axis=0)

    return mat

def F(F_mea_b,F_mea_db_x1,F_mea_db_x2,F_mea_db_x3,condition="Cauchy"):
    if condition=="Cauchy":
        F_mea_b=F_mea_b.reshape(-1,F_mea_b.shape[-1])
        F_mea_db_x1=F_mea_db_x1.reshape(-1,F_mea_db_x1.shape[-1])
        F_mea_db_x2=F_mea_db_x2.reshape(-1,F_mea_db_x2.shape[-1])
        F_mea_db_x3=F_mea_db_x3.reshape(-1,F_mea_db_x3.shape[-1])
        F=np.concatenate((F_mea_b[:,0:1],F_mea_b[:,1:],F_mea_db_x1[:,0:1],F_mea_db_x1[:,1:],F_mea_db_x2[:,0:1],F_mea_db_x2[:,1:],F_mea_db_x3[:,0:1],F_mea_db_x3[:,1:]),axis=0)
    if condition=="Dirichlet":
        F_mea_b=F_mea_b.reshape(-1,F_mea_b.shape[-1])
        F=np.concatenate((F_mea_b[:,0:1],F_mea_b[:,1:]),axis=0)
    return F

def L_curve(*args):
    if len(args) == 4:
        mat,F,lamb_regu,cupy_device = args
        n = mat.shape[1]
    elif len(args) == 10:
        M0,M_relu1,M_relu2,M_gauss1,M_gauss2,af,mat,F,lamb_regu,cupy_device = args
        n_from_mat = mat.shape[1]
        if af=="mixer":
            n=M0+M_relu1+M_relu2+M_gauss1+M_gauss2
        elif af=="mix":
            n=M0+M_relu1+M_relu2
        elif af=="relu":
            n=M_relu1+M_relu2
        else:
            n=M0
        if n != n_from_mat:
            print(
                f"L_curve: parameter width {n} does not match matrix width {n_from_mat}; "
                "using matrix width."
            )
            n = n_from_mat
    else:
        raise TypeError("L_curve expects either 4 args or 10 args")

    iter=len(lamb_regu)
    res=np.zeros([1,iter])
    reg_norm=np.zeros([1,iter])
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
