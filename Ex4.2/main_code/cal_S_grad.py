import generate_data2
import numpy as np
import torch
import pickle
import org_m
torch.set_default_dtype(torch.float64)
def array_difference(B, A):
    A_tuples = [tuple(row) for row in A]
    B_tuples = [tuple(row) for row in B]
    
    diff_tuples = [b for b in B_tuples if b not in A_tuples]
    
    return np.array(diff_tuples)

def mat_assemble1(cells,models,af,kk,temp,points_b,device,cupy_device):
    cells_copy1 = pickle.loads(pickle.dumps(cells))
    all_mat1= []
    all_S_basis1=[]
    all_g_p1=[]
    if temp==0:
        for cell in cells_copy1:
            loc_g1=torch.tensor(cell.gauss_points,requires_grad=True).to(device)
            if af=="mix":
                out_cell01 = models[0][0](loc_g1, "sigmoid")
                out_cell02 = models[0][0](loc_g1, "Tanh")

                out_cell1 = torch.cat((out_cell01, out_cell02), dim=1)
            else:
                out_cell1 = models[0][0](loc_g1, af)
            loc_S_basis1=out_cell1.cpu().detach().numpy()
            RR1=cell.R
            w1=cell.w
            sub_y11=cell.sub_y1
            sub_y21=cell.sub_y2
            loc_b_matrix1=[]
            loc_db_x1_matrix1=[]
            loc_db_x2_matrix1=[]
            mea=1
            batch_number=1
            for i in range(len(kk)):
                f=generate_data2.forward_operator_rec(kk[i],RR1,w1,sub_y11,sub_y21,mea,device,cupy_device)
                loc_b_matrix1.append(f.int_f(out_cell1,batch_number).get())
                loc_db_x1_matrix1.append(f.int_df_x1(out_cell1,batch_number).get())
                loc_db_x2_matrix1.append(f.int_df_x2(out_cell1,batch_number).get())

            loc_mat1=org_m.A(loc_b_matrix1,loc_db_x1_matrix1,loc_db_x2_matrix1)
            all_mat1.append(loc_mat1)
            all_S_basis1.append(loc_S_basis1)
            all_g_p1.append(loc_g1)
    sum_mat=np.sum(all_mat1,axis=0)
    all_g_n=torch.cat(all_g_p1,dim=0)
    return sum_mat,all_g_p1,all_g_n,cells_copy1

def mat_assemble2(cells,models0,models1,models2,af,kk,temp,points_b,device,cupy_device):
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
            
        all_g_p=torch.cat(all_g_t,dim=0)
        if af=="mix":
            out_cell00 = models0[0][0](all_g_p,"Tanh")
            out_cell01 = models1[0][0](all_g_p, "Gauss")
            out_cell02 = models2[0][0](all_g_p, "Gauss")

            out_cell = torch.cat((out_cell00,out_cell01, out_cell02), dim=1)
        elif af=="Gauss":
            out_cell01 = models1[0][0](all_g_p, "Gauss")
            out_cell02 = models2[0][0](all_g_p, "Gauss")
            out_cell = torch.cat((out_cell01, out_cell02), dim=1)
        else:
            out_cell = models0[0][0](all_g_p, af)
        S_basis=out_cell.cpu().detach().numpy()
        X=points_b[:,np.newaxis,:]
        Y=all_g_p.cpu().detach().numpy()[np.newaxis,:,:]
        diff=X-Y
        r=np.sqrt(points_b[0,0]**2+points_b[0,1]**2)
        normal=np.tile(X,(1,all_g_p.shape[0],1))/r
        RR=np.linalg.norm(diff,axis=2)
        W=np.concatenate(weight_store,axis=0).reshape(-1)


        b_matrix=[]
        db_matrix=[]
        mea=1
        batch_number=1
        for i in range(len(kk)):
            f=generate_data2.forward_operator_rec(kk[i],RR,W,diff,normal,mea,device,cupy_device)
            b_matrix.append(f.int_f(out_cell,batch_number).get())
            db_matrix.append(f.int_df(out_cell,batch_number).get())

        all_mat=org_m.A(b_matrix,db_matrix)

    return all_mat,all_g_t,all_g_p,cells_copy
