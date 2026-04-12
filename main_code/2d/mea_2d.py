import numpy as np

def p_b(x_min,x_max,y_min,y_max,n):
    l=np.linspace(y_min,y_max,n).reshape(-1,1)
    p_l=np.concatenate((x_min*np.ones_like(l),l),axis=1)
    p_r=np.concatenate((x_max*np.ones_like(l),l),axis=1)
    b=np.linspace(x_min,x_max,n).reshape(-1,1)
    p_b=np.concatenate((b,y_min*np.ones_like(b)),axis=1)
    p_u=np.concatenate((b,y_max*np.ones_like(b)),axis=1)
    p_bound=np.concatenate((p_l,p_r,p_b,p_u),axis=0)
    return p_bound


def p_b_circle(center_x=0, center_y=0, radius=0.5, alpha=1, n=100, temp=True):
    theta = np.linspace(0, 2 * alpha * np.pi, n, endpoint=temp)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return np.stack((x, y), axis=1)
