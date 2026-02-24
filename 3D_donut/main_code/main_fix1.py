import grid_cell,cal_S_grad,error_general_3D,adaptive_int_fix,org_m,S_valgrad,visual
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
def visualize_3d_grid(cells, title="3D Adaptive Grid", alpha=0.9):
    """
    Visualizes the 3D grid cells as wireframes.
    
    Args:
        cells: List of Cell objects (leaf cells).
    """
    if not cells:
        print("No cells to visualize.")
        return

    leaf_cells = cells
    
    segments = []
    colors = []
    
    levels = [c.level for c in leaf_cells]
    min_level = min(levels) if levels else 0
    max_level = max(levels) if levels else 1
    
    cmap = plt.get_cmap('jet', max_level - min_level + 1)
    
    print(f"Plotting {len(leaf_cells)} cells. Levels: {min_level} to {max_level}")

    for cell in leaf_cells:
        x, y, z = cell.x0, cell.y0, cell.z0
        s = cell.size
        lvl = cell.level
        
        c = cmap((lvl - min_level) / (max_level - min_level + 1e-6))
        
        current_alpha = alpha
        
        if lvl == min_level:
            current_alpha = 0.05 
        elif lvl == min_level + 1:
            current_alpha = 0.2
        else:
            current_alpha = 0.9 

        cell_segments = [
            [(x, y, z), (x+s, y, z)],    [(x+s, y, z), (x+s, y+s, z)],
            [(x+s, y+s, z), (x, y+s, z)],[(x, y+s, z), (x, y, z)],
            [(x, y, z+s), (x+s, y, z+s)],       [(x+s, y, z+s), (x+s, y+s, z+s)],
            [(x+s, y+s, z+s), (x, y+s, z+s)],   [(x, y+s, z+s), (x, y, z+s)],
            [(x, y, z), (x, y, z+s)],           [(x+s, y, z), (x+s, y, z+s)],
            [(x+s, y+s, z), (x+s, y+s, z+s)],   [(x, y+s, z), (x, y, z+s)]
        ]
        
        segments.extend(cell_segments)
        c_rgba = list(c) 
        c_rgba[3] = current_alpha 
        colors.extend([c_rgba] * 12)
    
    if leaf_cells:
        all_x = [c.x0 for c in leaf_cells] + [c.x0+c.size for c in leaf_cells]
        all_y = [c.y0 for c in leaf_cells] + [c.y0+c.size for c in leaf_cells]
        all_z = [c.z0 for c in leaf_cells] + [c.z0+c.size for c in leaf_cells]
        xlims = (min(all_x), max(all_x))
        ylims = (min(all_y), max(all_y))
        zlims = (min(all_z), max(all_z))
    else:
        xlims = ylims = zlims = (0, 1)

    fig = plt.figure(figsize=(18, 6)) 
    
    views = [(0, 90), (90, 180), (0, 180)]
    
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        lc = Line3DCollection(segments, colors=colors, linewidths=0.5) 
        ax.add_collection3d(lc)
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Elev={elev}, Azim={azim}")
        
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(f"{title}\nCounts: {len(leaf_cells)}\n(Low levels are transparent)")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_level, vmax=max_level))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=fig.axes, shrink=0.8, pad=0.02, location='right')
    cbar.set_label('Refinement Level')
    try:
         cbar.set_ticks(range(min_level, max_level + 1))
    except:
         pass
    
    plt.tight_layout()
    plt.show()
    
def ada_int(iter_int,delta,cells,nx,models,M,af,kk,F,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max,center_true,R_true,r_true,S0,ana_S,points_b,lamb_regu,Qx,Qy,Qz,ratio1,condition,device,cupy_device,refine_threshold_S,refine_threshold_grad,refine_threshold_noise,current_maxiter,max_level):
    cells_store=[]
    Cells=cells
    refinement_stats_store=[]
    leaf_cells=grid_cell.collect_leaf_cells(Cells)
    all_points0, all_weights = grid_cell.collect_all_gauss_points(leaf_cells)
    w_=[]
    point_number=[]
    S_num_store=[]
    S_basis_store=[]
    S_l2=[]
    g_store=[]
    sample_s=error_general_3D.test_p(Qx,Qy,Qz,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max)
    for i in range(iter_int):
        cells_store.append(Cells)
        visualize_3d_grid(cells_store[-1])
        point_number.append(all_points0.shape[0])
        g_store.append(all_points0)
        all_mat,all_g_list,all_g_p,Cells,S_basis=cal_S_grad.mat_assemble2(Cells,models,[],[],[],af,[],kk,0,points_b,condition,device,cupy_device) 
        res,reg_norm,w_store=org_m.L_curve(all_mat,F,lamb_regu,cupy_device)
        print("第",i,"次迭代的train误差：")
        S_num,grad_S_num=S_valgrad.valgrad(models,af,M,all_points0,w_store[:,0],ana_S,center_true,R_true,r_true,S0,device)
        print("第",i,"次迭代的泛化误差：")
        S_test_num,S_test_true,S_l_inf1,S_l_21,g_S=error_general_3D.test(models,[],sample_s,w_store[:,0],af,[],True,True,tau_x_min,tau_x_max,tau_y_min,tau_y_max,tau_z_min,tau_z_max,center_true,R_true,r_true,S0,ana_S,1,ratio1,device)
        S_num_store.append(S_test_num)
        S_basis_store.append(S_basis)
        S_l2.append(S_l_21)
        current_w = w_store[:,0]
        w_.append(current_w)
        if i>=1:
            if np.linalg.norm(S_test_num-S_num_store[i-1])/np.linalg.norm(S_test_num)<delta:
                print("在第",i,"次迭代结束")
                break

        def evaluator_func(pts_np):

            s, g_s = S_valgrad.valgrad(models, af, M, pts_np, current_w, ana_S, center_true, R_true,r_true, S0,device)
            
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            if isinstance(g_s, torch.Tensor):
                g_s = g_s.detach().cpu().numpy()
            
            return np.hstack([s.reshape(-1, 1), g_s])
            
        
        refined_cells_list, Cells, refinement_stats = adaptive_int_fix.integrate_adaptive_refinement_fix(
                Cells, evaluator_func, refine_threshold_S[i],refine_threshold_grad[i], refine_threshold_noise[i],nx, device,current_maxiter=current_maxiter,max_level=max_level
            )
        
        refinement_stats_store.append(refinement_stats)
        leaf_cells = Cells
        print('当前总网格数目:', len(leaf_cells))
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
    
    return cells_store,refinement_stats_store,w_,point_number,g_store,g_S,S_num_store,S_l2,S_basis_store
        
