import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualize_3d_grid(cells, title="3D Adaptive Grid", views=[(0, 90), (90, 180), (0, 180)], alpha=0.9,
                      output_directory=".", output_filename="grid_visualization.png"):
    """
    Visualizes the 3D grid cells as wireframes.
    
    Args:
        cells: List of Cell objects (leaf cells).
    """
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    full_output_path = os.path.join(output_directory, output_filename)

    if not cells:
        print("No cells to visualize.")
        return

    leaf_cells = cells
    
    # 准备绘图数据：线段和颜色
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
    
    # 计算坐标范围
    if leaf_cells:
        all_x = [c.x0 for c in leaf_cells] + [c.x0+c.size for c in leaf_cells]
        all_y = [c.y0 for c in leaf_cells] + [c.y0+c.size for c in leaf_cells]
        all_z = [c.z0 for c in leaf_cells] + [c.z0+c.size for c in leaf_cells]
        xlims = (min(all_x), max(all_x))
        ylims = (min(all_y), max(all_y))
        zlims = (min(all_z), max(all_z))
    else:
        xlims = ylims = zlims = (0, 1)

    # 创建图形
    fig = plt.figure(figsize=(10, 10)) 
    
    # 使用第一个视角
    if views:   
        elev, azim = views[0]
    else:
        elev, azim = 30, -60

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_position([0.01, 0.02, 0.75, 0.96])
    
    lc = Line3DCollection(segments, colors=colors, linewidths=0.5) 
    ax.add_collection3d(lc)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    
    ax.tick_params(axis='both', which='major', labelsize=16, pad=6)
    ax.tick_params(axis='x', which='major', labelsize=16, pad=6)

    ax.set_xlabel(r'$x_1$', fontsize=25, labelpad=20)
    ax.set_ylabel(r'$x_2$', fontsize=25, labelpad=20)
    ax.set_zlabel(r'$x_3$', fontsize=25, labelpad=20)
    #ax.text( 0.5+0.1, -0.5-0.01 , 0.5+0.05, "$x_3$", fontsize=25, ha='center')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.dist = 10
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_level, vmax=max_level))
    sm.set_array([])
    # 添加公共颜色条
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.01)
    #bar 刻度字体变大
    cbar.ax.tick_params(labelsize=20)
    #cbar.set_label('Refinement Level', fontsize=20)
    try:
         cbar.set_ticks(range(min_level, max_level + 1))
    except:
         pass
    
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    