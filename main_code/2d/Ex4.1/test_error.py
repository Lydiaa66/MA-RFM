import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sns
import heat_map_2D
import generate_data2,heat_map_2D
np.random.seed(2)


def test(models0,M,Qx,Qy,s_delta,s_star,af,temp,jump,x_min,x_max,y_min,y_max,device): #只限划分一个区域的情况
    test_Qx = 2*Qx #测试点的个数
    test_Qy = 2*Qy
    #关于源的
    x_devide = np.linspace(x_min, x_max, test_Qx + 1)
    y_devide = np.linspace(y_min, y_max, test_Qy + 1)
    grid = np.array(list(itertools.product(x_devide,y_devide))).reshape(test_Qx+1,test_Qy+1,2)
    sample_s=grid.reshape(-1,2)
    length=sample_s.shape[0] #测试点个数
    test_point = torch.tensor(sample_s,requires_grad=True).to(device)
    S_num_pred=torch.zeros([length,1])
    S_star=torch.zeros([length,1])
    batch_size=int(test_point.shape[0]/10)
    num_batches=int(test_point.shape[0]/batch_size)
    s_star = torch.tensor(s_star.reshape(-1,1)).to(device)
    s_delta = torch.tensor(s_delta.reshape(-1,1)).to(device)


    for i in range(num_batches):
        start_idx = i * batch_size
        if i<num_batches-1:
            end_idx =  (i + 1) * batch_size #防止越界
        else:
            end_idx =test_point.shape[0]+1
        batch_x = test_point[start_idx:end_idx, :]  # 取当前批次数据
        
        batch_output_num = torch.mm(models0[0][0](batch_x,af,[]).to(device),s_delta)  # 假设模型返回的输出是 (1, batch_size, 2)
        batch_output_star =torch.mm(models0[0][0](batch_x,af,[]).to(device),s_star)  
        # 将当前批次的输出合并到总输出中
        S_num_pred[start_idx:end_idx, :] = batch_output_num
        S_star[start_idx:end_idx, :] = batch_output_star
        del batch_output_num,batch_output_star


    S_num=(S_num_pred.cpu().detach().numpy()).reshape(test_Qx+1,test_Qy+1)
    S_num_0=S_num[:,int(0.5*S_num.shape[1])+1-1]
    S_star=(S_star.cpu().detach().numpy()).reshape(test_Qx+1,test_Qy+1)
    S_true_0=S_star[:,int(0.5*S_star.shape[1])+1-1]
    S_epsilon=np.abs((S_star-S_num))
    #re_S_epsilon=np.abs((S_star-S_num)/S_star)
    S_o=math.sqrt(np.sum(S_star**2)/length)
    S_l_inf=S_epsilon.max()/S_o
    S_l_2=math.sqrt(np.sum(S_epsilon**2)/length)/S_o
    print('S_l_inf=',S_l_inf,
          'S_L_2=',math.sqrt(np.sum(S_epsilon**2)/length),'S_l_2=',S_l_2)# *为矩阵逐元素乘法
        

    if temp:     
        heat_map_2D.improved_plot(S_star, title="Exact Solution of S", x_min=x_min, x_max=x_max,
                     y_min=y_min, y_max=y_max, cmap="hot")       
        heat_map_2D.improved_plot(S_num, title="Numerical Solution of S", x_min=x_min, x_max=x_max,
                      y_min=y_min, y_max=y_max, cmap='hot')
        heat_map_2D.improved_plot(S_epsilon, title="Numerical Solution of S", x_min=x_min, x_max=x_max,
                      y_min=y_min, y_max=y_max, cmap="hot")

    if jump:
          plt.figure(figsize=(5, 3), dpi=120)
          plt.plot(range(int(S_num.shape[1])), S_num_0, linestyle='--', color='red',label="Numerical Solution")
          plt.plot(range(int(S_star.shape[1])), S_true_0,linestyle='-', color='blue' ,label="Exact Solution")
          plt.legend(fontsize=6)
          plt.xlabel("x1", fontsize=14)
          n_ticks=7
          x_labels = np.linspace(x_min, x_max, n_ticks)
          y_labels = np.linspace(y_min, y_max, n_ticks)

          # 四舍五入处理
          x_labels = np.round(x_labels, decimals=2)  # 保留两位小数
          y_labels = np.round(y_labels, decimals=2)  # 保留两位小数
          plt.xticks(ticks=np.linspace(0, S_num.shape[1]-1, n_ticks),labels=x_labels, rotation=0, fontsize=10)

          plt.grid()
          plt.tight_layout()
          plt.show()
    return S_num,S_star,S_l_inf,S_l_2
            

