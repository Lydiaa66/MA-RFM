import grid_cell,cal_S_grad,error_general_ada,adaptive_int,org_m,S_valgrad,visual,ass_grad
import numpy as np

def ada_int(iter_int,delta,cells,nx,models,M,af,kk,F,tau_x_min,tau_x_max,tau_y_min,tau_y_max,r_low,r_upper,center_true,r_true,ana_S,points_b,lamb_regu,Qx,Qy,device,cupy_device):
    cells_store=[]
    Cells=cells
    refinement_stats_store=[]
    leaf_cells=grid_cell.collect_leaf_cells(Cells)
    all_points0, all_weights = grid_cell.collect_all_gauss_points(leaf_cells)
    w_=[]
    point_number=[]
    S_num_store=[]
    S_l2=[]
    S_l_20=0
    g_store=[]
    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat,all_g_list,all_g_p,Cells,grads_y1,grads_y2=ass_grad.mat_assemble(Cells,models,[],[],af,[],[],kk,0,points_b,0,device,cupy_device) 
        res,reg_norm,w_store=org_m.L_curve(all_mat,F,[],[],[],lamb_regu,"linear",device,cupy_device)
        print("第",i,"次迭代的train误差：")
        S_num,grad_S_num=S_valgrad.valgrad(models,af,[],M,all_points0,w_store[:,0],ana_S,center_true,r_true,device)
        print("第",i,"次迭代的泛化误差：")
        S_test_num,S_test_true,S_l_inf1,S_l_21,g_S=error_general_ada.test(models,[],[],M,Qx,Qy,w_store[:,0],af,[],[],True,True,tau_x_min,tau_x_max,tau_y_min,tau_y_max,[],[],[],[],center_true,r_true,ana_S,1,device)
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        w_.append(w_store[:,0])
        
        if i>=1:
            if np.linalg.norm(S_test_num-S_num_store[i-1])/np.linalg.norm(S_test_num)<delta:
                print("在第",i,"次迭代结束")
                break
        
        print("第",i,"次自适应网格细分...")
        _, Cells, refinement_stats, amr = adaptive_int.integrate_adaptive_refinement(Cells, all_g_list, S_num, grad_S_num,indicator_type="mixed", threhold_strategy="smart_fixed", iteration_count=i,n=nx)
        refinement_stats_store.append(refinement_stats)
        leaf_cells = grid_cell.collect_leaf_cells(Cells)
        all_points1, all_weights = grid_cell.collect_all_gauss_points(leaf_cells)

        all_points0=all_points1
        
    return cells_store,refinement_stats_store,w_,point_number,g_store,g_S,S_l2,S_num_store
        
