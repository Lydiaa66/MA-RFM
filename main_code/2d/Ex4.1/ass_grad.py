import generate_data2
import numpy as np
import torch
import pickle
import org_m
torch.set_default_dtype(torch.float64)


def mat_assemble(cells,models0,models1,models2,af,Shape1,Shape2,kk,temp,points_b,grad_temp,device,cupy_device):
    
    cells_copy = pickle.loads(pickle.dumps(cells))
    all_mat= [] #储存所有网格的mat
    all_S_basis=[]
    all_g_t=[] #储存高斯点
    weight_store=[]
    #p_b=torch.tensor(points_b).to(device)
    if temp==0:
        for cell in cells_copy: #遍历每个网格 先把所有的gauss点统计好 一起计算
            loc_g=cell.gauss_points
            all_g_t.append(torch.tensor(loc_g,requires_grad=True).to(device))
            weight_store.append(cell.w)
            
        all_g_p = torch.cat(all_g_t, dim=0)
        all_g_p.requires_grad_(True)
        #g_p_tensor=torch.tensor(all_g_p).to(device)
        if af=="mix":
            out_cell00 = models0[0][0](all_g_p, "Tanh",[])
            out_cell01 = models1[0][0](all_g_p, "sigmoid",Shape1)
            out_cell02 = models2[0][0](all_g_p, "sigmoid",Shape2)  # 每个单元格对应的基函数的值
            out_cell = torch.cat((out_cell00,out_cell01, out_cell02), dim=1)  # 按列拼接
        elif af=="circle+rec":
            out_cell01 = models1[0][0](all_g_p, "sigmoid",Shape1)
            out_cell02 = models2[0][0](all_g_p, "sigmoid",Shape2)  # 每个单元格对应的基函数的值
            out_cell = torch.cat((out_cell01,out_cell02),dim=1)
        else:
            out_cell = models0[0][0](all_g_p, af,[])  # 按列拼接
        S_basis=out_cell #√
        M=S_basis.shape[1]
        grads_y1=np.zeros([S_basis.shape[0],M])
        grads_y2=np.zeros([S_basis.shape[0],M])
        if grad_temp==1:
            for i in range(M):
                grad=torch.autograd.grad(outputs=S_basis[:,i],inputs=all_g_p,grad_outputs=torch.ones_like(S_basis[:,i]),create_graph=True,retain_graph=True)[0]
                grads_y1[:,i]=grad[:,0].cpu().detach().numpy()
                grads_y2[:,i]=grad[:,1].cpu().detach().numpy()
        
        
        X=points_b[:,np.newaxis,:]
        Y=all_g_p.cpu().detach().numpy()[np.newaxis,:,:]
        diff=X-Y
        RR=np.linalg.norm(diff,axis=2)#√
        sub_y1=((diff)[:,:,0])[:int(0.5*RR.shape[0]),:] #√
        sub_y2=((diff)[:,:,1])[int(0.5*RR.shape[0]):int(RR.shape[0]),:] #√
        W=np.concatenate(weight_store,axis=0).reshape(-1) #确实因为权值的问题

        b_matrix=[] #储存多波数边界点的积分矩阵 (len(k)*p_b.shape[0],M)
        db_x1_matrix=[] #储存多波数 左右边界点 关于x第1分量求导的 积分矩阵 (len(k)*p_lr.shape[0],M)
        db_x2_matrix=[] #储存多波数 下上边界点 关于x第2分量求导的 积分矩阵 (len(k)*p_du.shape[0],M)
        mea=1
        batch_number=1
        for i in range(len(kk)):
            f=generate_data2.forward_operator_rec(kk[i],RR,W,sub_y1,sub_y2,mea,device,cupy_device)
            b_matrix.append(f.int_f(out_cell,batch_number).get())
            db_x1_matrix.append(f.int_df_x1(out_cell,batch_number).get())
            db_x2_matrix.append(f.int_df_x2(out_cell,batch_number).get())

        all_mat=org_m.A(b_matrix,db_x1_matrix,db_x2_matrix)

    return all_mat,all_g_t,all_g_p,cells_copy,grads_y1,grads_y2
