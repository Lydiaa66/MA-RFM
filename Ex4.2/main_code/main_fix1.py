import grid_cell,cal_S_grad,error_general_2D,adaptive_int_fix1,org_m,S_valgrad,visual
import numpy as np
import torch


def ada_int(iter_int,delta,cells,nx,models,M,af,kk,F,tau_x_min,tau_x_max,tau_y_min,tau_y_max,r_low,r_upper,center_true,r_true,ana_S,points_b,lamb_regu,Qx,Qy,device,cupy_device,refine_threshold_S,refine_threshold_grad,refine_threshold_noise,current_maxiter,max_level):
    cells_store=[]
    Cells=cells
    refinement_stats_store=[]
    leaf_cells=grid_cell.collect_leaf_cells(Cells)
    all_points0, all_weights = grid_cell.collect_all_gauss_points(leaf_cells)
    w_=[]
    point_number=[]
    S_num_store=[]
    S_l2=[]
    g_store=[]
    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat,all_g_list,all_g_p,Cells=cal_S_grad.mat_assemble2(Cells,models,[],[],af,kk,0,points_b,device,cupy_device) 
        res,reg_norm,w_store=org_m.L_curve(M,[],[],af,all_mat,F,lamb_regu,cupy_device)
        print("第",i,"次迭代的train误差：")
        S_num,grad_S_num=S_valgrad.valgrad(models,af,M,all_points0,w_store[:,0],ana_S,center_true,r_true,device)
        print("第",i,"次迭代的泛化误差：")
        S_test_num,S_test_true,S_l_inf1,S_l_21,g_S=error_general_2D.test(models,[],[],Qx,Qy,w_store[:,0],af,True,True,tau_x_min,tau_x_max,tau_y_min,tau_y_max,[],[],[],[],center_true,r_true,ana_S,0,device)
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        current_w = w_store[:,0]
        w_.append(current_w)
        
        if i>=1:
            if np.linalg.norm(S_test_num-S_num_store[i-1])/np.linalg.norm(S_test_num)<delta:
                print("在第",i,"次迭代结束")
                break
        
        print("第",i,"次自适应网格细分...")
        def evaluator_func(pts_np):

            s, g_s = S_valgrad.valgrad(models, af, M, pts_np, current_w, ana_S, center_true, r_true, device)
            
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(g_s, torch.Tensor):
                g_s = g_s.detach().cpu().numpy()
            
            g_norm = np.linalg.norm(g_s, axis=1).reshape(-1, 1)
            return np.hstack([s.reshape(-1, 1), g_s])
            
        
        refined_cells_list, Cells, refinement_stats = adaptive_int_fix1.integrate_adaptive_refinement_fix(
            Cells, evaluator_func, refine_threshold_S[i],refine_threshold_grad[i],refine_threshold_noise[i], nx, device,current_maxiter=current_maxiter,max_level=max_level
        )
        
        refinement_stats_store.append(refinement_stats)
        leaf_cells = Cells

        try:
            visual.visualize_adaptive_refinement(
                cells=cells_store[i],
                refined_cells=Cells,
                all_g_p=all_points0,
                S_num=S_num,
                grad_S_num=grad_S_num,
                refinement_stats=refinement_stats,
                cell_indicators=None
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
        Cells = Cells
        all_points0, all_weights = grid_cell.collect_all_gauss_points(Cells)

    return cells_store,refinement_stats_store,w_,point_number,g_store,g_S,S_l2,S_num_store
