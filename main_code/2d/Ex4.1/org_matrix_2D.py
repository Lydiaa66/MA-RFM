import time
import torch
import numpy as np
import generate_data2
def batch(sample,num_batches,models,af,M): #输入3D张量 (1,,)
    n=sample.shape[1]
    #out = np.empty((1, n, Nx*Ny*M))  
    out_tensor=torch.zeros([1,n,M])
    batch_size=int(n/num_batches)
    # 分批处理数据
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_x = sample[:, start_idx:end_idx, :]  # 取当前批次数据
        # 传入模型计算当前批次的输出
        batch_output = models(batch_x,af,[])  # 假设模型返回的输出是 (1, batch_size, 2)
        # 将当前批次的输出合并到总输出中
        out_tensor[:, start_idx:end_idx, :] = batch_output
    return out_tensor

def residual_vector(models,p_b,x_min,x_max,y_min,y_max,center_temp,r_temp,M,af,choose,mea,kk,number,num_batches_gauss,batch_number_mea,grad_temp,device,cupy_device): #number为高斯点的数量
     #points 为采样点 p_b为边界采样点 左右下上
    points_b=torch.tensor(p_b).to(device)
    points_b.requires_grad_(True)
    if choose=="rec":
        g=generate_data2.GaussLegendre2D_rec(number,x_min,x_max,y_min,y_max,p_b) #由于未知支集，先用矩形上的进行逼近
        gauss_points=g.points_int #积分点 也是输入点 (,2)
        RR=g.R
        w=g.w
        sub_y1=g.sub_y1
        sub_y2=g.sub_y2
    if choose=="circle":
        g=generate_data2.GaussLegendre2D_circle(number,center_temp,r_temp,p_b) #由于未知支集，先用矩形上的进行逼近
        gauss_points=g.y #[y1,y2]
        RR=g.RR
        w=g.w
        r_points=g.points_int[:,0]
        sub_y1=g.sub_y1
        sub_y2=g.sub_y2
    # contour_points=boundary_detect.detect("boundary_points.npy",0)
    # path=mpath.Path(contour_points)
    # cond=path.contains_points(gauss_points) #属于边界区域内的点
    g_p=torch.tensor(gauss_points,requires_grad=True).to(device) #将积分点转化为张量形式
    g_p_T=g_p[None,]
    out_tensor = batch(g_p_T,num_batches_gauss,models[0][0],af,M).to(device)  
    S_basis=(out_tensor.reshape(-1,M)).to(device)#(number^2,M) 
    grads_y1=np.zeros([S_basis.shape[0],M])
    grads_y2=np.zeros([S_basis.shape[0],M])
    if grad_temp==True:
        for i in range(M): #遍历每个基函数 分别求梯度 节省显存
            grad=torch.autograd.grad(outputs=S_basis[:,i],inputs=g_p,grad_outputs=torch.ones_like(S_basis[:,i]),create_graph=True,retain_graph=True)[0]
            grads_y1[:,i]=grad[:,0].cpu().detach().numpy()
            grads_y2[:,i]=grad[:,1].cpu().detach().numpy()

    #S_basis[~cond]=0
    S_basis=S_basis.cpu().detach().numpy()
    b_matrix=[] #储存多波数边界点的积分矩阵 (len(k)*p_b.shape[0],M)
    db_x1_matrix=[] #储存多波数 左右边界点 关于x第1分量求导的 积分矩阵 (len(k)*p_lr.shape[0],M)
    db_x2_matrix=[] #储存多波数 下上边界点 关于x第2分量求导的 积分矩阵 (len(k)*p_du.shape[0],M)

    for i in range(len(kk)):   
        if choose=="rec":     
            f=generate_data2.forward_operator_rec(kk[i],RR,w,sub_y1,sub_y2,mea,device,cupy_device)
        if choose=="circle":
            f=generate_data2.forward_operator_circle(kk[i],RR,w,r_points,sub_y1,sub_y2,mea,device,cupy_device)
        b_matrix.append((f.int_f(S_basis,batch_number_mea)).get()) #边界点的积分矩阵 (p_b.shape[0],M)
        db_x1_matrix.append((f.int_df_x1(S_basis,batch_number_mea)).get()) #左右边界点 关于x第1分量求导的 积分矩阵
        db_x2_matrix.append((f.int_df_x2(S_basis,batch_number_mea)).get()) #下上边界点 关于x第2分量求导的 积分矩阵    
    return S_basis,b_matrix,db_x1_matrix,db_x2_matrix,grads_y1,grads_y2
