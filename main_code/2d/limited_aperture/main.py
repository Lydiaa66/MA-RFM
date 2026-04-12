import mesh,matrix_assemble,error_general_2d,adaptive_int,inverse_solver,source_eval,visual
import numpy as np
import torch


def _threshold_at(threshold, i):
    if isinstance(threshold, np.ndarray):
        if threshold.ndim == 0:
            return threshold.item()
        return threshold[i]
    if isinstance(threshold, (list, tuple)):
        return threshold[i]
    return threshold


def _ada_int_impl(iter_int,delta,cells,nx,models,M,af,kk,F,tau_x_min,tau_x_max,tau_y_min,tau_y_max,r_low,r_upper,center_true,r_true,ana_S,points_b,lamb_regu,Qx,Qy,device,cupy_device,refine_threshold_S,refine_threshold_grad,refine_threshold_noise,current_maxiter,max_level):
    cells_store=[]
    Cells=cells
    refinement_stats_store=[]
    leaf_cells=mesh.collect_leaf_cells(Cells)
    all_points0, all_weights = mesh.collect_all_gauss_points(leaf_cells)
    w_=[]
    point_number=[]
    S_num_store=[]
    S_l2=[]

    g_store=[]
    for i in range(iter_int):
        cells_store.append(Cells)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat,all_g_list,all_g_p,Cells=matrix_assemble.mat_assemble2(Cells,models,[],[],af,kk,0,points_b,device,cupy_device) 
        res,reg_norm,w_store=inverse_solver.L_curve(M,[],[],af,all_mat,F,lamb_regu,cupy_device)
        print("Training error at iteration", i, ":")
        S_num,grad_S_num=source_eval.valgrad(models,af,M,all_points0,w_store[:,0],ana_S,center_true,r_true,device)
        print("Generalization error at iteration", i, ":")
        S_test_num,S_test_true,S_l_inf1,S_l_21,g_S=error_general_2d.test(models,[],[],Qx,Qy,w_store[:,0],af,True,True,tau_x_min,tau_x_max,tau_y_min,tau_y_max,[],[],[],[],center_true,r_true,ana_S,1,device)
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        current_w = w_store[:,0]
        w_.append(w_store[:,0])
        
        if i>=1:
            if np.linalg.norm(S_test_num-S_num_store[i-1])/np.linalg.norm(S_test_num)<delta:
                print("Stop at iteration", i)
                break
        
        def evaluator_func(pts_np):

            s, g_s = source_eval.valgrad(models, af,M, pts_np, current_w, ana_S, center_true, r_true, device)
            
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(g_s, torch.Tensor):
                g_s = g_s.detach().cpu().numpy()
            
            return np.hstack([s.reshape(-1, 1), g_s])
            
            
        print("Adaptive mesh refinement at iteration", i, "...")
        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
                Cells, evaluator_func, _threshold_at(refine_threshold_S, i),_threshold_at(refine_threshold_grad, i),_threshold_at(refine_threshold_noise, i), nx, device,current_maxiter=current_maxiter,max_level=max_level
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
            import traceback
            traceback.print_exc()
            print(f"Visualization failed: {e}")
        Cells = Cells
        all_points0, all_weights = mesh.collect_all_gauss_points(Cells)
        
        
        
    return cells_store,refinement_stats_store,w_,point_number,g_store,g_S,S_num_store,S_l2


def ada_int(*args):
    if len(args) == 28:
        return _ada_int_impl(*(args[:-2] + (args[-3],) + args[-2:]))
    if len(args) == 29:
        return _ada_int_impl(*args)
    raise TypeError("Unsupported limited_aperture.main.ada_int signature")
        
