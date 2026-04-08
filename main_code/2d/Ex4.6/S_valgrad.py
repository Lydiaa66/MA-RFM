import math
import numpy as np
import torch
def valgrad(models,af,Shape,signed_distances,M,all_points,w,ana_S,center,r,device):
    length=all_points.shape[0]
    all_g_p=torch.tensor(all_points,requires_grad=True).to(device)
    w_last=torch.tensor(w.reshape(-1,1)).to(device)
    S_basis_=models(all_g_p,af,Shape,signed_distances).reshape(-1,M).to(device)
    S_num=torch.mm(S_basis_,w_last)
    grad_S_num = torch.autograd.grad(
        outputs=S_num,
        inputs=all_g_p,
        grad_outputs=torch.ones_like(S_num),
        create_graph=True)[0]
    S_true=ana_S(all_points,x_left=0.29,x_right=0.49,y_below=0.3,y_upper=0.7,center=center,r=r).reshape(-1,1)
    S_num_np=S_num.cpu().detach().numpy()
    S_epsilon=np.abs(S_true-S_num_np)
    S_o=math.sqrt(np.sum(S_true**2)/length)
    print('S_l_inf=',S_epsilon.max()/S_o,'S_L_2=',math.sqrt(np.sum(S_epsilon**2)/length),'S_l_2=',math.sqrt(np.sum(S_epsilon**2)/length)/S_o)
    return S_num,grad_S_num