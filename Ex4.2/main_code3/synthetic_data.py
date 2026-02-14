import numpy as np
from scipy.special import hankel1, h1vp


def dir_neu(N_m,N_m_inv,kk,R,rho,n_trun,F_mea_b):
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
        dh_rho  = h1vp(n, kk * rho)
        hankel_ratio[:, i] = h_rho / h_R
        dhankel_ratio[:,i] = dh_rho/h_R
    COF = U_fourier * hankel_ratio
    KK=np.repeat(kk[:,np.newaxis],repeats=dhankel_ratio.shape[1],axis=1)
    d_COF = KK*U_fourier *dhankel_ratio
    theta_inv= np.linspace(0, 2 * np.pi, N_m_inv, endpoint=False)
    inv_exp_term = np.exp(1j * n_vals[:, np.newaxis] * theta_inv[np.newaxis, :])

    v = np.sum(COF[:, :, np.newaxis] * inv_exp_term[np.newaxis, :, :], axis=1)
    v_rho = np.zeros([kk.shape[0], N_m_inv, 2])
    v_rho[:,:,0]=np.real(v)
    v_rho[:,:,1]=np.imag(v)
    
    dv = np.sum(d_COF[:, :, np.newaxis] * inv_exp_term[np.newaxis, :, :], axis=1)
    dv_rho = np.zeros([kk.shape[0], N_m_inv, 2])
    dv_rho[:,:,0]=np.real(dv)
    dv_rho[:,:,1]=np.imag(dv)
    
    return v_rho,dv_rho