import generate_data2
import numpy as np
import torch
import pickle
import org_m,smooth
torch.set_default_dtype(torch.float64)

def mat_assemble2(cells,models0,models1,models2,models3,af,Shape1,Shape2,sign_distances,kk,temp,points_b,device,cupy_device):

        
    cells_copy = pickle.loads(pickle.dumps(cells))
    all_mat= []
    all_g_t=[]
    weight_store=[]
    if temp==0:
        for cell in cells_copy:
            loc_g=cell.gauss_points
            all_g_t.append(torch.tensor(loc_g,requires_grad=True).to(device))
            weight_store.append(cell.w)
            
        all_g_p=torch.cat(all_g_t,dim=0)
        if af=="mix+gauss":
            if isinstance(sign_distances,np.ndarray):
                sign_distances=torch.tensor(sign_distances).to(device) 
            out_cell00 = models0(all_g_p, "Tanh",[],[])
            out_cell01 = models1(all_g_p, "sigmoid",Shape1,sign_distances)
            out_cell02 = models2(all_g_p, "sigmoid",Shape2,[])
            out_cell03 = models3(all_g_p, "continue_gauss",[],[])
            out_cell = torch.cat((out_cell00,out_cell01, out_cell02,out_cell03), dim=1)
        if af=="mix":
            if isinstance(sign_distances,np.ndarray):
                sign_distances=torch.tensor(sign_distances).to(device) 
            out_cell00 = models0(all_g_p, "Tanh",[],[])
            out_cell01 = models1(all_g_p, "sigmoid",Shape1,sign_distances)
            out_cell02 = models2(all_g_p, "sigmoid",Shape2,[])
            out_cell = torch.cat((out_cell00,out_cell01, out_cell02), dim=1)
        elif af=="circle+rec":
            out_cell01 = models1(all_g_p, "sigmoid",Shape1,[])
            out_cell02 = models2(all_g_p, "sigmoid",Shape2,[])
            out_cell = torch.cat((out_cell01,out_cell02),dim=1)
        elif af=="sigmoid" and Shape1=="general" and Shape2 !="noise":
            if isinstance(sign_distances,np.ndarray):
                sign_distances=torch.tensor(sign_distances).to(device) 
            out_cell = models0(all_g_p, "sigmoid",Shape1,sign_distances)
        elif af=="sigmoid" and Shape1=="general" and Shape2=="noise":
            if isinstance(sign_distances,np.ndarray):
                sign_distances=torch.tensor(sign_distances).to(device) 
            out_cell00 = models0(all_g_p, "sigmoid",Shape1,sign_distances)
            out_cell01 = models1(all_g_p, "sigmoid",Shape2,[])
            out_cell = torch.cat((out_cell00,out_cell01), dim=1)
        else:
            out_cell = models0(all_g_p, af,[],[])
        X=points_b[:,np.newaxis,:]
        Y=all_g_p.cpu().detach().numpy()[np.newaxis,:,:]
        diff=X-Y
        RR=np.linalg.norm(diff,axis=2)
        sub_y1=((diff)[:,:,0])[:int(0.5*RR.shape[0]),:]
        sub_y2=((diff)[:,:,1])[int(0.5*RR.shape[0]):int(RR.shape[0]),:]
        W=np.concatenate(weight_store,axis=0).reshape(-1)


        b_matrix=[]
        db_x1_matrix=[]
        db_x2_matrix=[]
        mea=1
        batch_number=1
        for i in range(len(kk)):
            f=generate_data2.forward_operator_rec(kk[i],RR,W,sub_y1,sub_y2,mea,device,cupy_device)
            b_matrix.append(f.int_f(out_cell,batch_number).get())
            db_x1_matrix.append(f.int_df_x1(out_cell,batch_number).get())
            db_x2_matrix.append(f.int_df_x2(out_cell,batch_number).get())

        all_mat=org_m.A(b_matrix,db_x1_matrix,db_x2_matrix)
    return all_mat,all_g_t,all_g_p,cells_copy