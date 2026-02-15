import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sns
import heat_map_2D
import generate_data3,heat_map_2D
from importlib import reload
reload(heat_map_2D)
np.random.seed(2)
import warnings
warnings.filterwarnings("ignore", message="findfont:")

def test_p(Qx,Qy,Qz,x_min,x_max,y_min,y_max,z_min,z_max):
    test_Qx = 2*Qx
    test_Qy = 2*Qy
    test_Qz = 2*Qz
    x_devide = np.linspace(x_min, x_max, test_Qx + 1)
    y_devide = np.linspace(y_min, y_max, test_Qy + 1)
    z_devide = np.linspace(z_min, z_max, test_Qz + 1)
    x_flat = x_devide.flatten()
    y_flat = y_devide.flatten() 
    z_flat = z_devide.flatten()
    X, Y , Z= np.meshgrid(x_flat, y_flat,z_flat,indexing='ij')
    sample_s = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T  
    return sample_s


def test(models0,model_sigmoid,sample_s,w,af,Shape,temp,jump,x_min,x_max,y_min,y_max,z_min,z_max,center_true,R_true,r_true,S0,ana_S,grad_temp,ratio1,device):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    length=sample_s.shape[0]
    test_Qx = round(length ** (1/3)) - 1
    test_Qy = round(length ** (1/3)) - 1
    test_Qz = round(length ** (1/3)) - 1
    test_point = torch.tensor(sample_s,requires_grad=True).to(device)

    S_num_pred=torch.zeros([length,1])
    batch_size=int(test_point.shape[0]/(test_Qx+1))
    num_batches=int(test_point.shape[0]/batch_size)
    w = torch.tensor(w.reshape(-1,1)).to(device)
    g_S_list=[]
    for i in range(num_batches):
        start_idx = i * batch_size
        if i<num_batches-1:
            end_idx =  (i + 1) * batch_size
        else:
            end_idx =test_point.shape[0]+1
        batch_x = test_point[start_idx:end_idx, :]
        if af=="mixer":
            with torch.no_grad():
                basis_0=models0(batch_x,"Tanh",device).to(device)
                basis_1=model_relu1(batch_x,"relu",device).to(device)
                basis_2=model_relu2(batch_x,"relu",device).to(device)
                basis_3=model_gauss1(batch_x,"Gauss",device).to(device)
                basis_4=model_gauss2(batch_x,"Gauss",device).to(device)
            af_basis=torch.cat((basis_0,basis_1,basis_2,basis_3,basis_4),dim=1)
            del basis_0,basis_1,basis_2,basis_3,basis_4
            batch_output = torch.mm(af_basis,w)
            del af_basis
            if grad_temp==1:
                grad_S_batch = torch.autograd.grad(
                    outputs=batch_output,
                    inputs=batch_x,
                    grad_outputs=torch.ones_like(batch_output),
                    create_graph=False)[0]
                g_S_list.append(grad_S_batch.cpu().detach().numpy())
                del grad_S_batch
        elif af=="mix":
            basis_0=models0(batch_x,"Tanh",[],device).to(device)
            basis_1=model_sigmoid(batch_x,"sigmoid",Shape,device).to(device)
            af_basis=torch.cat((basis_0,basis_1),dim=1)
            del basis_0,basis_1
            batch_output = torch.mm(af_basis,w)
            del af_basis
            if grad_temp==1:
                grad_S_batch = torch.autograd.grad(
                    outputs=batch_output,
                    inputs=batch_x,
                    grad_outputs=torch.ones_like(batch_output),
                    create_graph=False)[0]
                g_S_list.append(grad_S_batch.cpu().detach().numpy())
                del grad_S_batch
        elif af=="Gauss":
            with torch.no_grad():
                basis_1=model_gauss1(batch_x,"Gauss",device).to(device)
                basis_2=model_gauss2(batch_x,"Gauss",device).to(device)
            af_basis=torch.cat((basis_1,basis_2),dim=1)
            del basis_1,basis_2
            batch_output = torch.mm(af_basis,w)
            if grad_temp==1:
                grad_S_batch = torch.autograd.grad(
                    outputs=batch_output,
                    inputs=batch_x,
                    grad_outputs=torch.ones_like(batch_output),
                    create_graph=False)[0]
                g_S_list.append(grad_S_batch.cpu().detach().numpy())
                del grad_S_batch
        elif af=="sigmoid" and Shape=="sweet":
            basis_1=model_sigmoid(batch_x,"sigmoid",Shape,device).to(device)
            af_basis=basis_1
            del basis_1
            batch_output = torch.mm(af_basis,w)
            del af_basis
            if grad_temp==1:
                grad_S_batch = torch.autograd.grad(
                    outputs=batch_output,
                    inputs=batch_x,
                    grad_outputs=torch.ones_like(batch_output),
                    create_graph=False)[0]
                g_S_list.append(grad_S_batch.cpu().detach().numpy())
                del grad_S_batch
        else:
            batch_output = torch.mm(models0(batch_x,af,Shape,device).to(device),w)
            if grad_temp==1:
                grad_S_batch = torch.autograd.grad(
                    outputs=batch_output,
                    inputs=batch_x,
                    grad_outputs=torch.ones_like(batch_output),
                    create_graph=False)[0]
                g_S_list.append(grad_S_batch.cpu().detach().numpy())
                del grad_S_batch
        S_num_pred[start_idx:end_idx, :] = batch_output
        del batch_output

    
    if grad_temp==1:
        g_S_num = np.concatenate(g_S_list, axis=0)
        g_S=np.linalg.norm(g_S_num,axis=-1)
        g_S=np.linalg.norm(g_S_num,axis=-1)
        g_S[g_S > 500] = 0
        g_S=g_S.reshape(test_Qx+1,test_Qy+1,test_Qz+1)
        g_S_x1_mid=g_S[int(0.5*g_S.shape[0])+1-1,:,:]
        g_S_x2_mid=g_S[:,int(0.5*g_S.shape[1])+1-1,:]
        fig, axs = plt.subplots(1, 2, figsize=(8, 8))
        heat_map_2D.improved_plot(g_S_x2_mid, title="gradient_S",x_min=y_min, x_max=y_max,
                        y_min=z_min, y_max=z_max, cmap="rainbow",shrink=0.3,ax=axs[0])
        heat_map_2D.improved_plot(g_S_x1_mid, title="gradient_S", x_min=y_min, x_max=y_max,
                        y_min=z_min, y_max=z_max, cmap="rainbow",shrink=0.3,ax=axs[1])
        plt.tight_layout()
        plt.show()
        del w,test_point

    else:
        g_S=[]
    
    



    S_num=(S_num_pred.cpu().detach().numpy()).reshape(test_Qx+1,test_Qy+1,test_Qz+1)
    S_true=ana_S(sample_s,center_true,R_true, r_true, S0).reshape(test_Qx+1,test_Qy+1,test_Qz+1)
    
    S_num_x1_mid=S_num[int(0.5*S_num.shape[0])+1-1,:,:]
    S_true_x1_mid=S_true[int(0.5*S_true.shape[0])+1-1,:,:]
    S_epsilon_x1_mid=np.abs(S_true_x1_mid-S_num_x1_mid)
    S_num_1D_x1=S_num[:,int(ratio1[1]*S_num.shape[1])+1-1,int(ratio1[2]*S_num.shape[2])+1-1]
    S_true_1D_x1=S_true[:,int(ratio1[1]*S_true.shape[1])+1-1,int(ratio1[2]*S_true.shape[2])+1-1]
    mid_row = int(0.5 * S_num.shape[0])
    flat_x1_left = np.abs(S_num_x1_mid[:mid_row, :]).ravel()
    flat_indices_x1_left = np.argsort(flat_x1_left)[-1:][::-1]
    indices_2d_x1_left = np.unravel_index(flat_indices_x1_left, (mid_row, S_num_x1_mid.shape[1]))

    flat_x1_right = np.abs(S_num_x1_mid[mid_row:, :]).ravel()
    flat_indices_x1_right = np.argsort(flat_x1_right)[-1:][::-1]
    indices_2d_x1_right_tmp = np.unravel_index(flat_indices_x1_right, (S_num_x1_mid.shape[0] - mid_row, S_num_x1_mid.shape[1]))
    indices_2d_x1_right = (indices_2d_x1_right_tmp[0] + mid_row, indices_2d_x1_right_tmp[1])
    indices_2d_x1 = (
        np.array([indices_2d_x1_left[0][0], indices_2d_x1_right[0][0]]),
        np.array([indices_2d_x1_left[1][0], indices_2d_x1_right[1][0]])
    )

    top2_with_x1 = S_num_x1_mid[indices_2d_x1]


    S_num_x2_mid=S_num[:,int(0.5*S_num.shape[2])+1-1,:]
    S_true_x2_mid=S_true[:,int(0.5*S_true.shape[2])+1-1,:]
    S_epsilon_x2_mid=np.abs(S_true_x2_mid-S_num_x2_mid)
    S_num_1D_x2=S_num[int(ratio1[0]*S_num.shape[0])+1-1,:,int(ratio1[2]*S_num.shape[2])+1-1]
    S_true_1D_x2=S_true[int(ratio1[0]*S_true.shape[0])+1-1,:,int(ratio1[2]*S_true.shape[2])+1-1]
    flat_x2_left = np.abs(S_num_x2_mid[:mid_row, :]).ravel()
    flat_indices_x2_left = np.argsort(flat_x2_left)[-1:][::-1]
    indices_2d_x2_left = np.unravel_index(flat_indices_x2_left, (mid_row, S_num_x2_mid.shape[1]))

    flat_x2_right = np.abs(S_num_x2_mid[mid_row:, :]).ravel()
    flat_indices_x2_right = np.argsort(flat_x2_right)[-1:][::-1]
    indices_2d_x2_right_tmp = np.unravel_index(flat_indices_x2_right, (S_num_x2_mid.shape[0] - mid_row, S_num_x2_mid.shape[1]))
    indices_2d_x2_right = (indices_2d_x2_right_tmp[0] + mid_row, indices_2d_x2_right_tmp[1])
    indices_2d_x2 = (
        np.array([indices_2d_x2_left[0][0], indices_2d_x2_right[0][0]]),
        np.array([indices_2d_x2_left[1][0], indices_2d_x2_right[1][0]])
    )

    top2_with_x2 = S_num_x2_mid[indices_2d_x2]
    
    S_num_x3_mid=S_num[:,:,int(0.5*S_num.shape[0])+1-1]
    S_true_x3_mid=S_true[:,:,int(0.5*S_true.shape[0])+1-1]
    S_epsilon_x3_mid=np.abs(S_true_x3_mid-S_num_x3_mid)
    S_num_1D_x3=S_num[int(ratio1[0]*S_num.shape[1])+1-1,int(ratio1[2]*S_num.shape[2])+1-1,:]
    S_true_1D_x3=S_true[int(ratio1[0]*S_true.shape[1])+1-1,int(ratio1[2]*S_true.shape[2])+1-1,:]
    flat_x3_left = np.abs(S_num_x3_mid[:mid_row, :]).ravel()
    flat_indices_x3_left = np.argsort(flat_x3_left)[-1:][::-1]
    indices_2d_x3_left = np.unravel_index(flat_indices_x3_left, (mid_row, S_num_x3_mid.shape[1]))

    flat_x3_right = np.abs(S_num_x3_mid[mid_row:, :]).ravel()
    flat_indices_x3_right = np.argsort(flat_x3_right)[-1:][::-1]
    indices_2d_x3_right_tmp = np.unravel_index(flat_indices_x3_right, (S_num_x3_mid.shape[0] - mid_row, S_num_x3_mid.shape[1]))
    indices_2d_x3_right = (indices_2d_x3_right_tmp[0] + mid_row, indices_2d_x3_right_tmp[1])
    indices_2d_x3 = (
        np.array([indices_2d_x3_left[0][0], indices_2d_x3_right[0][0]]),
        np.array([indices_2d_x3_left[1][0], indices_2d_x3_right[1][0]])
    )

    top2_with_x3 = S_num_x3_mid[indices_2d_x3]

   
    S_epsilon=np.abs((S_true-S_num))
    S_o=math.sqrt(np.sum(S_true**2)/length)
    S_l_inf=S_epsilon.max()/S_o
    S_l_2=math.sqrt(np.sum(S_epsilon**2)/length)/S_o
    print('S_l_inf=',S_l_inf,
          'S_L_2=',math.sqrt(np.sum(S_epsilon**2)/length),'S_l_2=',S_l_2)
        

    if temp:            

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        y_vals = np.linspace(y_min, y_max, S_num_x1_mid.shape[0])
        z_vals = np.linspace(z_min, z_max, S_num_x1_mid.shape[1])
        y_coord_1, z_coord_1 = y_vals[indices_2d_x1[0][0]], z_vals[indices_2d_x1[1][0]]
        y_coord_2, z_coord_2 = y_vals[indices_2d_x1[0][1]], z_vals[indices_2d_x1[1][1]]

        heat_map_2D.improved_plot(
            S_num_x1_mid,
            title="Num_S(0, x2, x3)",
            x_min=y_min, x_max=y_max,
            y_min=z_min, y_max=z_max,
            xlabel="x2", ylabel="x3",
            cmap="rainbow", shrink=1,
            ax=axs[0, 0]
        )
        axs[0, 0].text(indices_2d_x1[0][0], indices_2d_x1[1][0], f"max={top2_with_x1[0]:.2f},[{y_coord_1:.2f}, {z_coord_1:.2f}]", color="white", fontsize=8,
                    ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        axs[0, 0].text(indices_2d_x1[0][1], indices_2d_x1[1][1], f"max={top2_with_x1[1]:.2f},[{y_coord_2:.2f}, {z_coord_2:.2f}]", color="white", fontsize=8,
                    ha='center', va='top', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

        heat_map_2D.improved_plot(
            S_true_x1_mid,
            title="Exact_S(0, x2, x3)",
            x_min=y_min, x_max=y_max,
            y_min=z_min, y_max=z_max,
            xlabel="x2", ylabel="x3",
            cmap="rainbow",shrink=1,
            ax=axs[0, 1]
        )


        x_vals = np.linspace(x_min, x_max, S_num_x2_mid.shape[0])
        z_vals = np.linspace(z_min, z_max, S_num_x2_mid.shape[1])
        x_coord_1, z_coord_1 = x_vals[indices_2d_x2[0][0]], z_vals[indices_2d_x2[1][0]]
        x_coord_2, z_coord_2 = x_vals[indices_2d_x2[0][1]], z_vals[indices_2d_x2[1][1]]

        heat_map_2D.improved_plot(
            S_num_x2_mid,
            title="Num_S(x1, 0, x3)",
            x_min=x_min, x_max=x_max,
            y_min=z_min, y_max=z_max,
            xlabel="x1", ylabel="x3",
            cmap="rainbow",shrink=1,
            ax=axs[1, 0]
        )
        axs[1, 0].text(indices_2d_x2[0][0], indices_2d_x2[1][0], f"max={top2_with_x2[0]:.2f},[{x_coord_1:.2f}, {z_coord_1:.2f}]", color="white", fontsize=8,
                    ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        axs[1, 0].text(indices_2d_x2[0][1], indices_2d_x2[1][1], f"max={top2_with_x2[1]:.2f},[{x_coord_2:.2f}, {z_coord_2:.2f}]", color="white", fontsize=8,
                    ha='center', va='top', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

        heat_map_2D.improved_plot(
            S_true_x2_mid,
            title="Exact_S(x1, 0, x3)",
            x_min=x_min, x_max=x_max,
            y_min=z_min, y_max=z_max,
            xlabel="x1", ylabel="x3",
            cmap="rainbow",shrink=1,
            ax=axs[1, 1]
        )
        
        
        x_vals = np.linspace(x_min, x_max, S_num_x3_mid.shape[0])
        y_vals = np.linspace(y_min, y_max, S_num_x3_mid.shape[1])
        x_coord_1, y_coord_1 = x_vals[indices_2d_x3[0][0]], y_vals[indices_2d_x3[1][0]]
        x_coord_2, y_coord_2 = x_vals[indices_2d_x3[0][1]], y_vals[indices_2d_x3[1][1]]

        heat_map_2D.improved_plot(
            S_num_x3_mid,
            title="Num_S(x1, x2, 0)",
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            xlabel="x1", ylabel="x2",
            cmap="rainbow",shrink=1,
            ax=axs[2, 0]
        )
        axs[2, 0].text(indices_2d_x3[0][0], indices_2d_x3[1][0], f"max={top2_with_x3[0]:.2f},[{x_coord_1:.2f}, {y_coord_1:.2f}]", color="white", fontsize=8,
                    ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))
        axs[2, 0].text(indices_2d_x3[0][1], indices_2d_x3[1][1], f"max={top2_with_x3[1]:.2f},[{x_coord_2:.2f}, {y_coord_2:.2f}]", color="white", fontsize=8,
                    ha='center', va='top', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.2'))

        heat_map_2D.improved_plot(
            S_true_x3_mid,
            title="Exact_S(x1, x2, 0)",
            x_min=x_min, x_max=x_max,
            y_min=z_min, y_max=z_max,
            xlabel="x1", ylabel="x3",
            cmap="rainbow",shrink=1,
            ax=axs[2, 1]
        )


        plt.show()


    if jump:
          plt.figure(figsize=(5, 3), dpi=120)
          plt.plot(range(int(S_num.shape[0])), S_num_1D_x1, linestyle='--', color='red',label="Num_S(x1, 0.5, 0.3)")
          plt.plot(range(int(S_true.shape[0])), S_true_1D_x1,linestyle='-', color='blue' ,label="exact_S(x1, 0.5, 0.3)")
          n_ticks=7
          x_labels = np.linspace(x_min, x_max, n_ticks)
          y_labels = np.linspace(y_min, y_max, n_ticks)
          x_labels = np.round(x_labels, decimals=2)
          y_labels = np.round(y_labels, decimals=2)
          plt.xticks(ticks=np.linspace(0, S_num.shape[1]-1, n_ticks),labels=x_labels, rotation=0, fontsize=10)
          
          plt.figure(figsize=(5, 3), dpi=120)
          plt.plot(range(int(S_num.shape[0])), S_num_1D_x2, linestyle='--', color='red',label="Num_S(0.5, x2, 0.8)")
          plt.plot(range(int(S_true.shape[0])), S_true_1D_x2,linestyle='-', color='blue' ,label="exact_S(0.5, x2, 0.8)")
          n_ticks=7
          x_labels = np.linspace(x_min, x_max, n_ticks)
          y_labels = np.linspace(y_min, y_max, n_ticks)
          x_labels = np.round(x_labels, decimals=2)
          y_labels = np.round(y_labels, decimals=2)
          plt.xticks(ticks=np.linspace(0, S_num.shape[1]-1, n_ticks),labels=x_labels, rotation=0, fontsize=10)
          

          plt.legend(fontsize=6)
          plt.grid()
          plt.tight_layout()
          plt.show()
    return S_num,S_true,S_l_inf,S_l_2,g_S
            

