import generate_data3
import numpy as np
import torch
import pickle
import org_m
torch.set_default_dtype(torch.float64)


def mat_assemble2(cells,models0,models1,models2,S_basis,af,Shape,kk,temp,points_b,condition,device,cupy_device):
    cells_copy = pickle.loads(pickle.dumps(cells))
    all_mat= []
    all_S_basis=[]
    all_g_t=[]
    weight_store=[]
    if temp==0:
        for cell in cells_copy:
            loc_g=cell.gauss_points
            all_g_t.append(torch.tensor(loc_g,requires_grad=True).to(device))
            weight_store.append(cell.w)
            
        all_g_p=torch.cat(all_g_t,dim=0).to(device)
        if af=="mix":
            out_cell00 = torch.tensor(S_basis).to(device)
            out_cell01 = models1(all_g_p, "sigmoid",Shape, device)
            out_cell = torch.cat((out_cell00,out_cell01), dim=1)
        elif af=="mixer":
            out_cell00 = torch.tensor(S_basis).to(device)
            out_cell01 = models1(all_g_p, "relu", device)
            out_cell02 = models2(all_g_p, "relu",device)
            out_cell03 = model_gauss1(all_g_p, "Gauss",device)
            out_cell04 = model_gauss2(all_g_p, "Gauss",device)
            out_cell = torch.cat((out_cell00,out_cell01, out_cell02,out_cell03,out_cell04), dim=1)
        elif af=="Gauss":
            out_cell01 = models1(all_g_p, "Gauss",device)
            out_cell02 = models2(all_g_p, "Gauss",device)
            out_cell = torch.cat((out_cell01, out_cell02), dim=1)
        elif af=="sigmoid" and Shape=="sweet":
            out_cell = models1(all_g_p, "sigmoid","sweet",device)
        else:
            out_cell = models0(all_g_p, af,Shape,device)
        S_basis=out_cell.cpu().detach().numpy()
        X=points_b[:,np.newaxis,:]
        Y=all_g_p.cpu().detach().numpy()[np.newaxis,:,:]
        diff=X-Y
        RR=np.linalg.norm(diff,axis=2)
        sub_y1=((diff)[:,:,0])[:int(2/6*RR.shape[0]),:]
        sub_y2=((diff)[:,:,1])[int(2/6*RR.shape[0]):int(4/6*RR.shape[0]),:]
        sub_y3=((diff)[:,:,2])[int(4/6*RR.shape[0]):int(RR.shape[0]),:]

        W=np.concatenate(weight_store,axis=0).reshape(-1)


        b_matrix=[]
        db_x1_matrix=[]
        db_x2_matrix=[]
        db_x3_matrix=[]
        mea=1
        batch_number=1
        for i in range(len(kk)):
            f=generate_data3.forward_operator_rec(kk[i],RR,W,sub_y1,sub_y2,sub_y3,mea,device,cupy_device)
            if condition=="Dirichlet":
                b_matrix.append(f.int_f(out_cell,batch_number,device,cupy_device).get())
                
            if condition=="Cauchy":
                b_matrix.append(f.int_f(out_cell,batch_number,device,cupy_device).get())
                db_x1_matrix.append(f.int_df_x1(out_cell,batch_number,device,cupy_device).get())
                db_x2_matrix.append(f.int_df_x2(out_cell,batch_number,device,cupy_device).get())
                db_x3_matrix.append(f.int_df_x3(out_cell,batch_number,device,cupy_device).get())


        all_mat=org_m.A(b_matrix,db_x1_matrix,db_x2_matrix,db_x3_matrix,condition)

    return all_mat,all_g_t,all_g_p,cells_copy,S_basis
