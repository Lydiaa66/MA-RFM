import numpy as np
import torch
import math
import pickle
import copy
class AdaptiveMeshRefinement:
    def __init__(self, cells, all_g_p, S_num, grad_S_num):
        """
        初始化自适应网格细分系统
        
        Args:
            cells: 初始网格单元列表
            all_g_p: 所有高斯点坐标 tensor
            S_num: 数值解在高斯点的值
            grad_S_num: 数值解在高斯点的梯度
        """
        self.cells = cells
        self.all_g_p = all_g_p
        self.S_num = S_num
        self.grad_S_num = grad_S_num
        
        if isinstance(all_g_p, torch.Tensor):
            self.all_g_p_np = all_g_p.cpu().detach().numpy()
        else:
            self.all_g_p_np = all_g_p
            
        if isinstance(S_num, torch.Tensor):
            self.S_num_np = S_num.cpu().detach().numpy()
        else:
            self.S_num_np = S_num
            
        if isinstance(grad_S_num, torch.Tensor):
            self.grad_S_num_np = grad_S_num.cpu().detach().numpy()
        else:
            self.grad_S_num_np = grad_S_num
    
    def map_gauss_points_to_cells(self):
        """
        将高斯点映射回对应的网格单元
        
        Returns:
            dict: {cell_index: [gauss_point_indices]}
        """
        cell_to_gauss_map = {}
        gauss_point_index = 0
        
        for cell_idx, cell in enumerate(self.cells):
            if not cell.children:
                num_gauss_points = cell.gauss_points.shape[0]
                gauss_indices = list(range(gauss_point_index, 
                                         gauss_point_index + num_gauss_points))
                cell_to_gauss_map[cell_idx] = gauss_indices
                gauss_point_index += num_gauss_points
        
        return cell_to_gauss_map
    
    def compute_cell_indicators(self, cell_to_gauss_map,iteration_count, indicator_type='gradient_norm'):
        """
        计算每个网格单元的细分指示器
        
        Args:
            cell_to_gauss_map: 网格到高斯点的映射
            indicator_type: 指示器类型 ('gradient_norm', 'solution_variation', 'mixed')
        
        Returns:
            dict: {cell_index: indicator_value}
        """
        cell_indicators = {}
        
        for cell_idx, gauss_indices in cell_to_gauss_map.items():
            if indicator_type == 'gradient_norm':
                gradients = self.grad_S_num_np[gauss_indices]
                grad_norms = np.linalg.norm(gradients, axis=1)
                indicator = np.max(grad_norms)
                
                
            elif indicator_type == 'mixed':
                gradients = self.grad_S_num_np[gauss_indices]
                solutions = self.S_num_np[gauss_indices]
                grad_norms = np.linalg.norm(gradients, axis=1)
                grad_indicator = np.mean(grad_norms)
                abs_indicator = np.mean(np.abs(solutions))
                A_const = 50.0
                B_const = 10.0
                abs_min_indicator = A_const * math.exp(-B_const * abs_indicator)
                indicator =  grad_indicator+80*abs_indicator
                if iteration_count==2:
                    indicator =  1.5*grad_indicator+30*abs_indicator
                if iteration_count>=3:
                    indicator =  2*grad_indicator+20*abs_indicator
            else:
                raise ValueError("Unsupported indicator type")
            
            cell_indicators[cell_idx] = indicator
        
        return cell_indicators
    
    def mark_cells_for_refinement(self, cell_indicators, threshold_strategy='fixed', 
                                iteration_count=0,threshold_value=None, refinement_ratio=0.3):
        """
        标记需要细分的网格单元
        
        Args:
            cell_indicators: 网格指示器值
            threshold_strategy: 阈值策略 ('fixed', 'adaptive', 'percentage')
            threshold_value: 固定阈值值
            refinement_ratio: 细分比例（用于percentage策略）
        
        Returns:
            list: 需要细分的网格索引列表
        """
        indicator_values = list(cell_indicators.values())
        
        if threshold_strategy == 'fixed':
            if threshold_value is None:
                threshold = np.mean(indicator_values) 

            cells_to_refine = [idx for idx, val in cell_indicators.items() 
                             if val > threshold]
            
        elif threshold_strategy == 'smart_fixed':
            if threshold_value is None:
                mean_val = np.mean(indicator_values)
            
            if iteration_count == 0:
                threshold = mean_val * 0.2
            elif iteration_count == 1:
                threshold = mean_val * 0.3
            else:
                threshold = mean_val * 1.4

            cells_to_refine = [idx for idx, val in cell_indicators.items() 
                         if val > threshold]
                             
        elif threshold_strategy == 'adaptive':
            threshold = np.percentile(indicator_values, 70)
            cells_to_refine = [idx for idx, val in cell_indicators.items() 
                             if val > threshold]
                             
        elif threshold_strategy == 'percentage':
            sorted_indices = sorted(cell_indicators.keys(), 
                                  key=lambda x: cell_indicators[x], reverse=True)
            num_to_refine = int(len(sorted_indices) * refinement_ratio)
            cells_to_refine = sorted_indices[:num_to_refine]
            
        else:
            raise ValueError("Unsupported threshold strategy")
        
        return cells_to_refine
    
    def refine_marked_cells(self, cells_to_refine, n):
        """
        细分标记的网格单元
        
        Args:
            cells_to_refine: 需要细分的网格索引列表
            n: 高斯点数量参数
        """
        refined_count = 0
        for cell_idx in cells_to_refine:
            if cell_idx < len(self.cells):
                cell = self.cells[cell_idx]
                if not cell.children:
                    cell.refine(n)
                    refined_count += 1
        
        print(f"成功细分了 {refined_count} 个网格单元")
        return refined_count
    
    def adaptive_refinement_cycle(self, n, max_iterations=5, 
                                indicator_type='gradient_norm',
                                threshold_strategy='adaptive',
                                threshold_value=None,
                                iteration_count=0,
                                refinement_ratio=0.3,
                                min_refinement_count=1):
        """
        执行完整的自适应细分循环
        
        Args:
            n: 高斯点数量参数
            max_iterations: 最大迭代次数
            indicator_type: 指示器类型
            threshold_strategy: 阈值策略
            threshold_value: 固定阈值
            refinement_ratio: 细分比例
            min_refinement_count: 最小细分数量（低于此数量停止迭代）
        
        Returns:
            dict: 细分统计信息
        """
        refinement_history = []
        
        for iteration in range(max_iterations):            
            cell_to_gauss_map = self.map_gauss_points_to_cells()
            
            
            cell_indicators = self.compute_cell_indicators(cell_to_gauss_map, iteration_count,
                                                         indicator_type)
            
            cells_to_refine = self.mark_cells_for_refinement(
                cell_indicators, threshold_strategy, iteration_count,
                threshold_value, refinement_ratio)
            print(f"当前活跃网格数量: {len(cell_to_gauss_map)}",f",标记细分网格数量: {len(cells_to_refine)}")
            
            if len(cells_to_refine) < min_refinement_count:
                print("细分网格数量过少，停止迭代")
                break
            
            refined_count = self.refine_marked_cells(cells_to_refine, n)
            
            refinement_info = {
                'iteration': iteration_count + 1,
                'active_cells': len(cell_to_gauss_map),
                'marked_cells': len(cells_to_refine),
                'refined_cells': refined_count,
                'max_indicator': max(cell_indicators.values()) if cell_indicators else 0,
                'mean_indicator': np.mean(list(cell_indicators.values())) if cell_indicators else 0,
                'final_cell_count': len(self.collect_leaf_cells()),
                'total_refinements': sum(info['refined_cells'] for info in refinement_history)
            }
            refinement_history.append(refinement_info)
            
            print(f"最大指示器值: {refinement_info['max_indicator']:.6f}",f",平均指示器值: {refinement_info['mean_indicator']:.6f}")

        
        return refinement_info
    
    def collect_leaf_cells(self):
        """收集所有叶子节点"""
        def collect_recursive(cells):
            leaf_cells = []
            for cell in cells:
                if not cell.children:
                    leaf_cells.append(cell)
                else:
                    leaf_cells.extend(collect_recursive(cell.children))
            return leaf_cells
        
        return collect_recursive(self.cells)
    
    def print_refinement_summary(self, refinement_stats):
        """打印细分总结"""
        print("\n" + "="*50)
        print("自适应网格细分总结")
        print("="*50)
        print(f"初始网格数量: {len(self.cells)}")
        print(f"最终网格数量: {refinement_stats['final_cell_count']}")
        print(f"总细分次数: {refinement_stats['total_refinements']}")
        print(f"细分轮数: {len(refinement_stats['history'])}")
        
        if refinement_stats['history']:
            print("\n各轮细分详情:")
            for info in refinement_stats['history']:
                print(f"第{info['iteration']:2d}轮: "
                      f"活跃网格={info['active_cells']:3d}, "
                      f"标记网格={info['marked_cells']:3d}, "
                      f"实际细分={info['refined_cells']:3d}, "
                      f"最大指示器={info['max_indicator']:.4f}")


def integrate_adaptive_refinement(cells, all_g_p, S_num, grad_S_num,indicator_type,threhold_strategy,iteration_count, n):
    """
    将自适应细分集成到你的现有代码中
    
    Args:
        cells: 你的网格单元列表
        all_g_p: 所有高斯点
        S_num: 数值解
        grad_S_num: 梯度
        n: 高斯点参数
    
    Returns:
        refined_cells: 细分后的网格列表
        refinement_stats: 细分统计信息
    """
    cells_copy = pickle.loads(pickle.dumps(cells))
    amr = AdaptiveMeshRefinement(cells_copy, all_g_p, S_num, grad_S_num)
    
    refinement_stats = amr.adaptive_refinement_cycle(
        n=n,
        max_iterations=1,
        indicator_type=indicator_type,
        threshold_strategy=threhold_strategy,
        iteration_count=iteration_count,
        refinement_ratio=1  
    )
    
    
    refined_cells = amr.collect_leaf_cells()
    
    return amr.cells,refined_cells, refinement_stats, amr

