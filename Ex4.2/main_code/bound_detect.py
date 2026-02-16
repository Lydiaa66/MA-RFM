import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
from matplotlib import rcParams
import os 

rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def map_index_to_domain(points,tau_x_min,tau_x_max,tau_y_min,tau_y_max,shape):
    """
    将索引点映射到定义域
    :param points: 索引点数组，形状为 (n, 2)
    :param tau_x_min: x 轴最小值
    :param tau_x_max: x 轴最大值
    :param tau_y_min: y 轴最小值
    :param tau_y_max: y 轴最大值
    :param shape: 数组形状 (height, width)
    :return: 映射后的点数组
    """
    x_scale=(tau_x_max-tau_x_min)/shape[1]
    y_scale=(tau_y_max-tau_y_min)/shape[0]
    
    mapped_points=np.zeros_like(points,dtype=float)
    mapped_points[:,0] = tau_x_min + points[:,0]*x_scale
    mapped_points[:,1] = tau_y_min + points[:,1]*y_scale
    return mapped_points


def detect(grad,X,Y,tau_x_min,tau_x_max,tau_y_min,tau_y_max,para,output_directory=".", output_filename="plot.png"):
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")
    elif not output_directory:
        output_directory = "."

    full_output_path = os.path.join(output_directory, output_filename)
    
    shape=grad.shape[0]
    mask=np.where(grad>np.max(grad)/para)
    
    mask1=mask[0]
    mask2=mask[1]
    boolean=(mask[0]>=5) & (mask[0]<=shape-5)
    mask=(mask1[boolean],mask2[boolean])

    plt.figure(figsize=(6, 6))
    
    plt.scatter(mask[0], mask[1], color='red', marker='o', s=5,
                edgecolor='white', linewidth=0.5)
    
    mask=np.array(mask).T
    hull = ConvexHull(mask)
    hull_points = mask[hull.vertices]
    
    
    line=0.5*(np.min(mask[:,0])+np.max(mask[:,0]))
    s1=np.where(mask[:,0]<line)[0]
    s2=np.where(mask[:,0]>line)[0]
    area1=mask[s1,:]
    area2=mask[s2,:]
    
    aa1_hull=ConvexHull(area1)
    aa1=area1[aa1_hull.vertices]
    aa2_hull=ConvexHull(area2)
    aa2=area2[aa2_hull.vertices]
    
    plt.plot(aa1[:, 0], aa1[:, 1], 'b--', lw=2, label='Area1 Convex Hull')
    plt.plot(aa2[:, 0], aa2[:, 1], 'g--', lw=2, label='Area2 Convex Hull')
    
    xlabel="$x_1$"
    ylabel="$x_2$"
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    n_ticks=5
    x_labels = np.linspace(tau_x_min, tau_x_max, n_ticks)
    y_labels = np.linspace(tau_y_min, tau_y_max, n_ticks)

    x_labels = np.round(x_labels, decimals=2)
    y_labels = np.round(y_labels, decimals=2)
    plt.xticks(ticks=np.linspace(0, grad.shape[0], n_ticks), labels=x_labels, rotation=0, fontsize=16)
    plt.yticks(ticks=np.linspace(0, grad.shape[1], n_ticks), labels=y_labels, fontsize=16)
    plt.legend()
    try:
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {full_output_path}")
    except Exception as e:
        print(f"Error saving plot to {full_output_path}: {e}")
    plt.show()



    mapped_hull_points1 = map_index_to_domain(aa1, tau_x_min, tau_x_max, tau_y_min, tau_y_max, grad.shape)
    center1=np.array([0.5*(np.max(mapped_hull_points1[:,0])+np.min(mapped_hull_points1[:,0])),0.5*(np.max(mapped_hull_points1[:,1])+np.min(mapped_hull_points1[:,1]))])
    r1=max(0.5*(np.max(mapped_hull_points1[:,0])-np.min(mapped_hull_points1[:,0])),0.5*(np.max(mapped_hull_points1[:,1])-np.min(mapped_hull_points1[:,1])))
    
    mapped_hull_points2 = map_index_to_domain(aa2, tau_x_min, tau_x_max, tau_y_min, tau_y_max, grad.shape)
    center2=np.array([0.5*(np.max(mapped_hull_points2[:,0])+np.min(mapped_hull_points2[:,0])),0.5*(np.max(mapped_hull_points2[:,1])+np.min(mapped_hull_points2[:,1]))])
    r2=max(0.5*(np.max(mapped_hull_points2[:,0])-np.min(mapped_hull_points2[:,0])),0.5*(np.max(mapped_hull_points2[:,1])-np.min(mapped_hull_points2[:,1])))



    return center1,center2,r1,r2
