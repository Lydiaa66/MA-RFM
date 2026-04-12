import adaptive_int
import error_general_3d
import inverse_solver
import matrix_assemble
import mesh
import numpy as np
import source_eval
import torch
import visual


def _threshold_at(threshold, i):
    if isinstance(threshold, np.ndarray):
        if threshold.ndim == 0:
            return threshold.item()
        return threshold[i]
    if isinstance(threshold, (list, tuple)):
        return threshold[i]
    return threshold


def _reuse_grad_threshold_as_noise(args):
    # New 3D AMR API uses refine_threshold_grad as the only gradient threshold.
    # Internally we keep the legacy tail
    # (refine_threshold_S, refine_threshold_grad, refine_threshold_noise, current_maxiter, max_level)
    # so old notebooks continue to work.
    return args[:-2] + (args[-3],) + args[-2:]
def _ada_int_ex47(iter_int,delta,cells,nx,models,M,af,kk,F,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max,center_true,r_true,ana_S,points_b,lamb_regu,Qx,Qy,Qz,ratio1,ratio2,device,cupy_device,refine_threshold_S,refine_threshold_grad,refine_threshold_noise,current_maxiter,max_level):
    model_width = getattr(models, "M", M)
    if model_width != M:
        print(f"ada_int: passed M={M} but loaded model width is {model_width}; using loaded model width.")
        M = model_width
    cells_store=[]
    Cells=cells
    refinement_stats_store=[]
    leaf_cells=mesh.collect_leaf_cells(Cells) # Initial leaf cells.
    all_points0, all_weights = mesh.collect_all_gauss_points(leaf_cells) # Initial Gauss points.
    w_=[]
    point_number=[]
    S_num_store=[]
    S_l2=[]

    g_store=[]
    sample_s=error_general_3d.test_p(Qx,Qy,Qz,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max)
    for i in range(iter_int):
        cells_store.append(Cells)
        visual.visualize_3d_grid(cells_store[-1], output_filename=None)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat,all_g_list,all_g_p,Cells,S_basis=matrix_assemble.mat_assemble2(Cells,models,[],[],[],[],[],af,kk,0,points_b,device,cupy_device) 
        #lamb_regu=np.logspace(-13,-12, 1, endpoint = True) 
        res,reg_norm,w_store=inverse_solver.L_curve(M,[],[],[],[],af,all_mat,F,lamb_regu,cupy_device) # Solve first.
        print("Training error at iteration", i, ":")
        S_num,grad_S_num=source_eval.valgrad(models,af,M,all_points0,w_store[:,0],ana_S,center_true,r_true,device)
        print("Generalization error at iteration", i, ":")
        S_test_num,S_test_true,S_l_inf1,S_l_21,g_S=error_general_3d.test(models,[],[],[],[],sample_s,w_store[:,0],af,True,True,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max,[],[],[],[],center_true,r_true,ana_S,1,ratio1,ratio2,device)
        S_num_store.append(S_test_num)
        S_l2.append(S_l_21)
        current_w = w_store[:,0] # Use best lambda
        w_.append(current_w)
        
        if i>=1:
            if np.linalg.norm(S_test_num-S_num_store[i-1])/np.linalg.norm(S_test_num)<delta:
                print("Stop at iteration", i)
                break
        
        print("Adaptive mesh refinement at iteration", i, "...")
        def evaluator_func(pts_np):

            s, g_s = source_eval.valgrad(
                models, af, M, pts_np, current_w, ana_S, center_true, r_true, device, verbose=False
            )
            
            # Ensure s and g_s are numpy arrays on CPU
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(g_s, torch.Tensor):
                g_s = g_s.detach().cpu().numpy()
            
            #g_norm = np.linalg.norm(g_s, axis=1).reshape(-1, 1) # Gradient norm
            return np.hstack([s.reshape(-1, 1), g_s]) # Return (N, 2): |u| and |grad u|.
            
        
            # Call the new adaptive refinement routine
        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
                Cells,
                evaluator_func,
                _threshold_at(refine_threshold_S, i),
                _threshold_at(refine_threshold_grad, i),
                _threshold_at(refine_threshold_noise, i),
                nx,
                device,
                current_maxiter=current_maxiter,
                max_level=max_level,
            ) # Returned Cells is the refined leaf-cell list; the input cells are the current leaf cells.
        
        refinement_stats_store.append(refinement_stats)
        leaf_cells = Cells # Cells is now the leaf list? No.
        print("Current number of cells:", len(leaf_cells))
        Cells = Cells # Update Cells to be the new leaves
        all_points0, all_weights = mesh.collect_all_gauss_points(Cells)
    
    return cells_store,refinement_stats_store,w_,point_number,g_store,g_S,S_num_store,S_basis


def _ada_int_donut(iter_int,delta,cells,nx,models,M,af,kk,F,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max,center_true,R_true,r_true,S0,ana_S,points_b,lamb_regu,Qx,Qy,Qz,ratio1,condition,device,cupy_device,refine_threshold_S,refine_threshold_grad,refine_threshold_noise,current_maxiter,max_level):
    model_width = getattr(models, "M", M)
    if model_width != M:
        print(f"ada_int: passed M={M} but loaded model width is {model_width}; using loaded model width.")
        M = model_width
    cells_store=[]
    Cells=cells
    refinement_stats_store=[]
    leaf_cells=mesh.collect_leaf_cells(Cells)
    all_points0, all_weights = mesh.collect_all_gauss_points(leaf_cells)
    w_=[]
    point_number=[]
    S_num_store=[]
    S_basis_store=[]
    S_l2=[]
    g_store=[]
    sample_s=error_general_3d.test_p(Qx,Qy,Qz,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max)
    for i in range(iter_int):
        cells_store.append(Cells)
        visual.visualize_3d_grid(cells_store[-1], output_filename=None)
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat,all_g_list,all_g_p,Cells,S_basis=matrix_assemble.mat_assemble2(Cells,models,[],[],[],af,[],kk,0,points_b,condition,device,cupy_device)
        res,reg_norm,w_store=inverse_solver.L_curve(all_mat,F,lamb_regu,cupy_device)
        print("Training error at iteration", i, ":")
        S_num,grad_S_num=source_eval.valgrad(models,af,M,all_points0,w_store[:,0],ana_S,center_true,R_true,r_true,S0,device)
        print("Generalization error at iteration", i, ":")
        S_test_num,S_test_true,S_l_inf1,S_l_21,g_S=error_general_3d.test(models,[],sample_s,w_store[:,0],af,[],True,True,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max,center_true,R_true,r_true,S0,ana_S,1,ratio1,device)
        S_num_store.append(S_test_num)
        S_basis_store.append(S_basis)
        S_l2.append(S_l_21)
        current_w = w_store[:,0]
        w_.append(current_w)
        if i>=1:
            if np.linalg.norm(S_test_num-S_num_store[i-1])/np.linalg.norm(S_test_num)<delta:
                print("Stop at iteration", i)
                break

        def evaluator_func(pts_np):
            s, g_s = source_eval.valgrad(
                models, af, M, pts_np, current_w, ana_S, center_true, R_true, r_true, S0, device, verbose=False
            )
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(g_s, torch.Tensor):
                g_s = g_s.detach().cpu().numpy()
            return np.hstack([s.reshape(-1, 1), g_s])

        refined_cells_list, Cells, refinement_stats = adaptive_int.integrate_adaptive_refinement_fix(
                Cells,
                evaluator_func,
                _threshold_at(refine_threshold_S, i),
                _threshold_at(refine_threshold_grad, i),
                _threshold_at(refine_threshold_noise, i),
                nx,
                device,
                current_maxiter=current_maxiter,
                max_level=max_level,
            )

        refinement_stats_store.append(refinement_stats)
        leaf_cells = Cells
        print("Current number of cells:", len(leaf_cells))
        Cells = Cells
        all_points0, all_weights = mesh.collect_all_gauss_points(Cells)

    return cells_store,refinement_stats_store,w_,point_number,g_store,g_S,S_num_store,S_l2,S_basis_store


def ada_int(*args):
    if len(args) == 31:
        return _ada_int_ex47(*_reuse_grad_threshold_as_noise(args))
    if len(args) == 32:
        return _ada_int_ex47(*args)
    if len(args) == 33:
        return _ada_int_donut(*_reuse_grad_threshold_as_noise(args))
    if len(args) == 34:
        return _ada_int_donut(*args)
    raise TypeError("ada_int expects either 31/32 args (Ex4.7) or 33/34 args (3D_donut)")
        
