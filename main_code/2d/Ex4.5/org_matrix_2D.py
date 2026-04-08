import time
import torch
import numpy as np
import generate_data2
def batch(sample,num_batches,models,af,M):
    n=sample.shape[1]
    out_tensor=torch.zeros([1,n,M])
    batch_size=int(n/num_batches)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_x = sample[:, start_idx:end_idx, :]
        batch_output = models(batch_x,af,[])
        out_tensor[:, start_idx:end_idx, :] = batch_output
    return out_tensor

def residual_vector(models,p_b,x_min,x_max,y_min,y_max,center_temp,r_temp,M,af,choose,mea,kk,number,num_batches_gauss,batch_number_mea,grad_temp,device,cupy_device):
    points_b=torch.tensor(p_b).to(device)
    points_b.requires_grad_(True)
    if choose=="rec":
        g=generate_data2.GaussLegendre2D_rec(number,x_min,x_max,y_min,y_max,p_b)
        gauss_points=g.points_int
        RR=g.R
        w=g.w
        sub_y1=g.sub_y1
        sub_y2=g.sub_y2
    if choose=="circle":
        g=generate_data2.GaussLegendre2D_circle(number,center_temp,r_temp,p_b)
        gauss_points=g.y
        RR=g.RR
        w=g.w
        r_points=g.points_int[:,0]
        sub_y1=g.sub_y1
        sub_y2=g.sub_y2
    g_p=torch.tensor(gauss_points,requires_grad=True).to(device)
    g_p_T=g_p[None,]
    out_tensor = batch(g_p_T,num_batches_gauss,models[0][0],af,M).to(device)  
    S_basis=(out_tensor.reshape(-1,M)).to(device)
    grads_y1=np.zeros([S_basis.shape[0],M])
    grads_y2=np.zeros([S_basis.shape[0],M])
    if grad_temp==True:
        for i in range(M):
            grad=torch.autograd.grad(outputs=S_basis[:,i],inputs=g_p,grad_outputs=torch.ones_like(S_basis[:,i]),create_graph=True,retain_graph=True)[0]
            grads_y1[:,i]=grad[:,0].cpu().detach().numpy()
            grads_y2[:,i]=grad[:,1].cpu().detach().numpy()

    S_basis=S_basis.cpu().detach().numpy()
    b_matrix=[]
    db_x1_matrix=[]
    db_x2_matrix=[]

    for i in range(len(kk)):   
        if choose=="rec":     
            f=generate_data2.forward_operator_rec(kk[i],RR,w,sub_y1,sub_y2,mea,device,cupy_device)
        if choose=="circle":
            f=generate_data2.forward_operator_circle(kk[i],RR,w,r_points,sub_y1,sub_y2,mea,device,cupy_device)
        b_matrix.append((f.int_f(S_basis,batch_number_mea)).get())
        db_x1_matrix.append((f.int_df_x1(S_basis,batch_number_mea)).get())
        db_x2_matrix.append((f.int_df_x2(S_basis,batch_number_mea)).get())
    return S_basis,b_matrix,db_x1_matrix,db_x2_matrix,grads_y1,grads_y2
