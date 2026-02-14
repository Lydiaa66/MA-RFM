import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
import seaborn as sns
import heat_map_2D
import generate_data2,heat_map_2D
np.random.seed(2)


def test(models0,models1,models2,M,Qx,Qy,w,af,Shape1,Shape2,temp,jump,x_min,x_max,y_min,y_max,x_left,x_right,y_below,y_upper,center_true,r_true,ana_S,grad_temp,device): #只限划分一个区域的情况
    test_Qx = 2*Qx #测试点的个数
    test_Qy = 2*Qy
    #关于源的
    x_devide = np.linspace(x_min, x_max, test_Qx + 1)
    y_devide = np.linspace(y_min, y_max, test_Qy + 1)
    grid = np.array(list(itertools.product(x_devide,y_devide))).reshape(test_Qx+1,test_Qy+1,2)
    sample_s=grid.reshape(-1,2)
    length=sample_s.shape[0] #测试点个数
    test_point = torch.tensor(sample_s,requires_grad=True).to(device)
    #out=models[0][0](test_point,af).reshape(-1,M)
    # w_tensor=torch.tensor(w,dtype=torch.float64).to(device)
    # S_num_pred=(out@w_tensor).reshape(-1,1)
    S_num_pred=torch.zeros([length,1])
    batch_size=int(test_point.shape[0]/10)
    num_batches=int(test_point.shape[0]/batch_size)
    w = torch.tensor(w.reshape(-1,1)).to(device)
    # 分批处理数据
    g_S_list=[]
    for i in range(num_batches):
        start_idx = i * batch_size
        if i<num_batches-1:
            end_idx =  (i + 1) * batch_size #防止越界
        else:
            end_idx =test_point.shape[0]+1
        batch_x = test_point[start_idx:end_idx, :]  # 取当前批次数据
        if af=="mix":
            #with torch.no_grad():
            basis_0=models0[0][0](batch_x,"Tanh",[]).to(device)
            basis_1=models1[0][0](batch_x,"sigmoid",Shape1).to(device)  # 假设模型返回的输出是 (1, batch_size, 2)
            basis_2=models2[0][0](batch_x,"sigmoid",Shape2).to(device)  # 假设模型返回的输出是 (1, batch_size, 2)
            af_basis=torch.cat((basis_0,basis_1,basis_2),dim=1)
            batch_output = torch.mm(af_basis,w)
            del af_basis,basis_0,basis_1,basis_2
            if grad_temp==1:
                grad_S_batch = torch.autograd.grad(
                    outputs=batch_output,  # S_num_tensor 是模型的输出
                    inputs=batch_x,  # 对 X_test_tensor 求梯度
                    grad_outputs=torch.ones_like(batch_output),  # 梯度传播的权重
                    create_graph=False)[0]  # 如果需要高阶梯度，设置为 True
                g_S_list.append(grad_S_batch.cpu().detach().numpy())
                del grad_S_batch
        elif af=="circle+rec":
            with torch.no_grad():
                basis_1=models1[0][0](batch_x,"sigmoid",Shape1).to(device)  # 假设模型返回的输出是 (1, batch_size, 2)
                basis_2=models2[0][0](batch_x,"sigmoid",Shape2).to(device)  # 假设模型返回的输出是 (1, batch_size, 2)
            af_basis=torch.cat((basis_1,basis_2),dim=1)
            batch_output = torch.mm(af_basis,w)
        else:
            batch_output = torch.mm(models0[0][0](batch_x,af,[]).to(device),w)  # 假设模型返回的输出是 (1, batch_size, 2)

        # 将当前批次的输出合并到总输出中
        S_num_pred[start_idx:end_idx, :] = batch_output
        del batch_output
    if af=="mix":
        g_S_num = np.concatenate(g_S_list, axis=0)
    
    if grad_temp==1:
        if af !="mix":
            grad_S_num = torch.autograd.grad(
                outputs=S_num_pred,  # S_num_tensor 是模型的输出
                inputs=test_point,  # 对 X_test_tensor 求梯度
                grad_outputs=torch.ones_like(S_num_pred),  # 梯度传播的权重
                create_graph=True)[0]  # 如果需要高阶梯度，设置为 True
            g_S_num=grad_S_num.cpu().detach().numpy()
        
        g_S=np.linalg.norm(g_S_num,axis=-1)
        g_S[g_S > 500] = 0 #剔除异常值
        g_S=g_S.reshape(test_Qx+1,test_Qy+1)
        heat_map_2D.improved_plot(g_S, title="gradient_S", x_min=x_min, x_max=x_max,
                        y_min=y_min, y_max=y_max, cmap="rainbow") #梯度热力图
        del w,test_point
    else:
        g_S=[]
    # g_S_0 = g_S[:,int(0.5*g_S.shape[1])+1-1] # 取中间切片的梯度(x1,0)
    # 绘制梯度结果
    # plt.figure(figsize=(4, 4), dpi=120)
    
    # plt.plot(range(int(g_S.shape[1])), g_S_0, linestyle='--', color='red', label="Numerical Solution")
    # plt.legend(fontsize=6)
    # plt.xlabel("x1", fontsize=14)
    # plt.title("g_S(x1,0)")
    # n_ticks = 7
    # x_labels = np.linspace(x_min, x_max, n_ticks)
    # y_labels = np.linspace(y_min, y_max, n_ticks)

    # # 四舍五入处理
    # x_labels = np.round(x_labels, decimals=2)  # 保留两位小数
    # y_labels = np.round(y_labels, decimals=2)  # 保留两位小数
    # #plt.xticks(ticks=np.linspace(0, g_S.shape[1] - 1, n_ticks), labels=x_labels, rotation=0, fontsize=10)

    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    S_num=(S_num_pred.cpu().detach().numpy()).reshape(test_Qx+1,test_Qy+1)
    S_num_0=S_num[:,int(0.5*S_num.shape[1])+1-1]
    S_true=ana_S(sample_s,x_left,x_right,y_below,y_upper,center_true,r_true).reshape(test_Qx+1,test_Qy+1)
    S_true_0=S_true[:,int(0.5*S_true.shape[1])+1-1]
    S_epsilon=np.abs((S_true-S_num))
    #re_S_epsilon=np.abs((S_true-S_num)/S_true)
    S_o=math.sqrt(np.sum(S_true**2)/length)
    S_l_inf=S_epsilon.max()/S_o
    S_l_2=math.sqrt(np.sum(S_epsilon**2)/length)/S_o
    print('S_l_inf=',S_l_inf,
          'S_L_2=',math.sqrt(np.sum(S_epsilon**2)/length),'S_l_2=',S_l_2)# *为矩阵逐元素乘法
        

    if temp:            
        # S 的热图
        # heat_map_2D.improved_plot(S_epsilon, title="Absolute Error of S", x_min=x_min, x_max=x_max,
        #               y_min=y_min, y_max=y_max, cmap="rainbow")
        #heat_map_2D.improved_plot(S_true, title="Exact Solution of S", x_min=x_min, x_max=x_max,
        #              y_min=y_min, y_max=y_max, cmap="rainbow")
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

          # 四舍五入处理
          x_labels = np.round(x_labels, decimals=2)  # 保留两位小数
          y_labels = np.round(y_labels, decimals=2)  # 保留两位小数
          plt.xticks(ticks=np.linspace(0, S_num.shape[1]-1, n_ticks),labels=x_labels, rotation=0, fontsize=10)

          plt.grid()
          plt.tight_layout()
          plt.show()
    return S_num,S_true,S_l_inf,S_l_2,g_S
            

