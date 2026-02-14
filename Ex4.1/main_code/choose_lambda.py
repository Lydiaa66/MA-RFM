import numpy as np
from scipy.linalg import fractional_matrix_power
import cupy as cp

def Inconsistency(all_mat,eta_value,nu,w):
    store=all_mat.T@all_mat
    s_star=(fractional_matrix_power(store, nu))@w
    cp_mat=cp.asarray(all_mat)
    r= (cp.linalg.matrix_rank(cp_mat)).get()
    U,s,Vt= cp.linalg.svd(cp_mat, full_matrices=True)
    U_null_space_basis = U[:, r:] #左零空间
    # 在左零空间中生成一个随机向量
    np.random.seed(2)
    random_coeffs = np.random.randn(U.shape[0]- r, 1).reshape(-1,1)
    eta_vec = U_null_space_basis.get() @ random_coeffs  #任意生成正交空间对应的向量

    # 将其范数缩放到我们想要的值
    eta_vec = eta_vec / np.linalg.norm(eta_vec) * eta_value
    U_explainable =all_mat@s_star
    U_true= U_explainable+eta_vec
    return s_star,U_true

    
def prior(M,w,all_mat,S_basis,perturbation,eta_value,nu,s_star,U_true):
    # s_star=(fractional_matrix_power(store, nu))@w
    S_star=S_basis@s_star
    # U_true=all_mat@s_star
    #perturbation = np.random.uniform(-eps, eps, size=U_true.shape)
    U_perturbed = U_true *(1+perturbation)
    delta=np.linalg.norm(U_true-U_perturbed)
    if nu ==1:
        C_nu=1
    else:
        optim_p=nu/(1-nu)
        C_nu=optim_p**nu/(optim_p+1)
    #eta_value=scale*delta
    lambda_reg2=((delta+eta_value)/(4*C_nu*np.linalg.norm(w)))**(2/(2*nu+1))
    A_matrix = all_mat.T@all_mat+ lambda_reg2 * np.eye(M)
    b_vector = all_mat.T @ U_perturbed
    cp_A=cp.asarray(A_matrix)
    cp_b=cp.asarray(b_vector)
    s_delta = (cp.linalg.solve(cp_A, cp_b)).get()
    # s_delta=np.linalg.inv(store+lambda_reg2*np.eye(M))@all_mat.T@U_perturbed
    error_s=np.linalg.norm(s_star-s_delta)
    S_delta=S_basis@s_delta
    error_S=np.linalg.norm(S_delta-S_star)
    bound=((2*nu+1)*(4*nu)**(-2*nu/(2*nu+1))*(C_nu*np.linalg.norm(w))**(1/(2*nu+1))*(delta+eta_value)**(2*nu/(2*nu+1)))
    return delta,lambda_reg2,s_delta,s_star,S_delta,S_star,error_s,error_S,bound