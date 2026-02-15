import torch
import numpy as np
import matplotlib.pyplot as plt
import generate_data2
def gradient_adap_gauss(models,M,w,af,x_min,x_max,y_min,y_max,number,points_b,device):
    g=generate_data2.GaussLegendre2D_rec(number,x_min,x_max,y_min,y_max,points_b) 
    gauss_points=g.points_int
    g_T=torch.tensor(gauss_points,requires_grad=True).to(device)
    out=models[0][0](g_T,af)
    w_tensor=torch.tensor(w,dtype=torch.float64).to(device)
    S_num_pred=(out.reshape(-1,M)@w_tensor).reshape(-1,1)
    
    grad_S_num = torch.autograd.grad(
        outputs=S_num_pred,
        inputs=g_T,
        grad_outputs=torch.ones_like(S_num_pred),
        create_graph=True)[0]
     
    g_S_num=grad_S_num.cpu().detach().numpy()
    g_S=np.linalg.norm(g_S_num,axis=-1)
    g_S[g_S > 500] = 0

    plt.tricontourf(gauss_points[:,0], gauss_points[:,1],g_S, levels=20, cmap='viridis')
    plt.colorbar(label='Gradient Magnitude')
    plt.title('Gradient Contour Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    return g_S