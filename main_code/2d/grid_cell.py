import generate_data2
import numpy as np

class Cell:
    def __init__(self, x0, y0, size, x,level=0):
        self.x0 = x0
        self.y0 = y0
        self.x= x
        self.size = size
        self.level = level
        self.gauss_points = None
        self.w = None
        self.R=None
        self.sub_y1=None
        self.sub_y2=None
        self.children = []

    def pre_calculate(self, n):
        g=generate_data2.GaussLegendre2D_rec(n,self.x0,self.x0+self.size,self.y0,self.y0+self.size,self.x)
        self.gauss_points=g.points_int
        self.w=g.w
        self.X=g.X
        self.R=g.R
        self.sub_y1=g.sub_y1
        self.sub_y2=g.sub_y2

    def refine(self, n):
        h = self.size / 2
        self.children = []
        for dx in [0, h]:
            for dy in [0, h]:
                child = Cell(self.x0 + dx, self.y0 + dy, h, self.x,self.level + 1)
                child.pre_calculate(n)
                self.children.append(child)
    
    
def collect_all_gauss_points(cells):
    all_points = []
    all_weights = []
    for cell in cells:
        if not cell.children:
            all_points.append(cell.gauss_points)
            all_weights.append(cell.w)
    P = np.vstack(all_points)
    W = np.vstack(all_weights)
    return P, W

def create_initial_grid(x_min,x_max,y_min,y_max,N, n,x):
    cells = []
    h = (x_max-x_min) / N
    for i in range(N):
        for j in range(N):
            cell = Cell(i * h, j * h, h,x)
            cell.pre_calculate(n)
            cells.append(cell)
    return cells

def collect_leaf_cells(cells):
    leaf_cells = []
    for cell in cells:
        if not cell.children:
            leaf_cells.append(cell)
        else:
            leaf_cells.extend(collect_leaf_cells(cell.children))
    return leaf_cells


