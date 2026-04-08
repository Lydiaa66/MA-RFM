import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sns
import heat_map_2D
import generate_data2,heat_map_2D
np.random.seed(2)


def test(models0,models1,models2,Qx,Qy,w,af,temp,jump,x_min,x_max,y_min,y_max,x_left,x_right,y_below,y_upper,center_true,r_true,ana_S,grad_temp,device):
    test_Qx = 2*Qx
    test_Qy = 2*Qy
    x_devide = np.linspace(x_min, x_max, test_Qx + 1)
    y_devide = np.linspace(y_min, y_max, test_Qy + 1)
    grid = np.array(list(itertools.product(x_devide,y_devide))).reshape(test_Qx+1,test_Qy+1,2)
    sample_s=grid.reshape(-1,2)
    length=sample_s.shape[0]
    test_point = torch.tensor(sample_s,requires_grad=True).to(device)

    S_num_pred=torch.zeros([length,1])
    batch_size=int(test_point.shape[0]/(10))
    num_batches=int(test_point.shape[0]/batch_size)
    w = torch.tensor(w.reshape(-1,1)).to(device)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_x = test_point[start_idx:end_idx, :]
        if af=="mix":
            with torch.no_grad():
                basis_0=models0[0][0](batch_x,"Tanh").to(device)
                basis_1=models1[0][0](batch_x,"Gauss").to(device)
                basis_2=models2[0][0](batch_x,"Gauss").to(device)
            af_basis=torch.cat((basis_0,basis_1,basis_2),dim=1)
            del basis_0,basis_1,basis_2
            batch_output = torch.mm(af_basis,w)
        elif af=="Gauss":
            with torch.no_grad():
                basis_1=models1[0][0](batch_x,"Gauss").to(device)
                basis_2=models2[0][0](batch_x,"Gauss").to(device)
            af_basis=torch.cat((basis_1,basis_2),dim=1)
            del basis_1,basis_2
            batch_output = torch.mm(af_basis,w)
        else:
            batch_output = torch.mm(models0[0][0](batch_x,af).to(device),w)

        S_num_pred[start_idx:end_idx, :] = batch_output
        del batch_output
    
    if grad_temp==1:
        grad_S_num = torch.autograd.grad(
            outputs=S_num_pred,
            inputs=test_point,
            grad_outputs=torch.ones_like(S_num_pred),
            create_graph=True)[0]
        
        g_S_num=grad_S_num.cpu().detach().numpy()
        g_S=np.linalg.norm(g_S_num,axis=-1)
        g_S[g_S > 500] = 0
        g_S=g_S.reshape(test_Qx+1,test_Qy+1)
        heat_map_2D.improved_plot(g_S, title="gradient_S", x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max, cmap="rainbow")
        del w,test_point,grad_S_num
    else:
        g_S=[]
    
    



    S_num=(S_num_pred.cpu().detach().numpy()).reshape(test_Qx+1,test_Qy+1)
    S_num_0=S_num[:,int(0.5*S_num.shape[1])+1-1]
    S_true=ana_S(sample_s,x_left,x_right,y_below,y_upper,center_true,r_true).reshape(test_Qx+1,test_Qy+1)
    S_true_0=S_true[:,int(0.5*S_true.shape[1])+1-1]
    S_epsilon=np.abs((S_true-S_num))
    S_o=math.sqrt(np.sum(S_true**2)/length)
    S_l_inf=S_epsilon.max()/S_o
    S_l_2=math.sqrt(np.sum(S_epsilon**2)/length)/S_o
    print('S_l_inf=',S_l_inf,
          'S_L_2=',math.sqrt(np.sum(S_epsilon**2)/length),'S_l_2=',S_l_2)
        

    if temp:            
        heat_map_2D.improved_plot(S_num, title="Numerical Solution of S", x_min=x_min, x_max=x_max,
                      y_min=y_min, y_max=y_max, cmap="rainbow")

    if jump:
          plt.figure(figsize=(5, 3), dpi=120)
          plt.plot(range(int(S_num.shape[1])), S_num_0, linestyle='--', color='red',label="Numerical Solution")
          plt.plot(range(int(S_true.shape[1])), S_true_0,linestyle='-', color='blue' ,label="Exact Solution")
          plt.legend(fontsize=6)
          plt.xlabel("x1", fontsize=14)
          n_ticks=7
          x_labels = np.linspace(x_min, x_max, n_ticks)
          y_labels = np.linspace(y_min, y_max, n_ticks)

          x_labels = np.round(x_labels, decimals=2)
          y_labels = np.round(y_labels, decimals=2)
          plt.xticks(ticks=np.linspace(0, S_num.shape[1]-1, n_ticks),labels=x_labels, rotation=0, fontsize=10)

          plt.grid()
          plt.tight_layout()
          plt.show()
    return S_num,S_true,S_l_inf,S_l_2,g_S
            

