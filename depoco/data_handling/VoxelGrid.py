import numpy as np

class VoxelGrid():
    """
    VoxelGrid类用于创建和管理体素网格。
    它将空间划分为规则的3D网格，并可以添加点到相应的体素中。
    """
    def __init__(self, VOXEL, center, grid_size, voxel_size, point_dim=3):
        """
        初始化VoxelGrid类。

        参数:
        - VOXEL: 体素类的类型。
        - center: 体素网格的中心点。
        - grid_size: 体素网格的总大小。
        - voxel_size: 单个体素的大小。
        - point_dim: 点的维度，默认为3。
        """
        self.grid_size = grid_size
        self.center = center
        self.voxel_size = voxel_size

        # 计算体素网格的维度
        self.grid_dim = np.ceil(grid_size / voxel_size).astype('int')
        # 计算体素的数量
        self.num_voxel = int(np.prod(self.grid_dim))
        # 计算体素网格的起始点偏移
        self.origin_offset = center - np.ceil(grid_size / voxel_size) / 2 * self.voxel_size
        # 初始化体素网格
        self.grid = [VOXEL(point_dim) for i in range(self.num_voxel)]
        # 用于存储已使用的体素的索引
        self.used_voxel = []

    def addPoint(self, point):
        """
        向体素网格中添加一个点。

        参数:
        - point: 要添加的点。
        """
        idx = self.xyz2index(point)
        if(idx is not None):
            if (self.grid[idx].isEmpty()):
                self.used_voxel.append(idx)
            self.grid[idx].addPoint(point)

    def xyz2index(self, point):
        """
        将点的xyz坐标转换为体素网格中的索引。

        参数:
        - point: 点的坐标。

        返回:
        - 点所在体素的索引，如果点不在网格内则返回None。
        """
        rcl = ((point - self.origin_offset)/self.voxel_size).astype("int")
        if np.any(rcl < 0) or np.any(rcl >= (self.grid_dim)):
            return None
        return rcl[0] + rcl[1] * self.grid_dim[0] + rcl[2] * self.grid_dim[0] * self.grid_dim[1]

    def cloud2indices(self, point_cld):
        """
        将点云转换为体素网格中的索引。

        参数:
        - point_cld: 点云数组。

        返回:
        - 点所在体素的索引数组和有效的点索引。
        """
        rcl = ((point_cld - self.origin_offset)/self.voxel_size).astype("int")
        valid= np.argwhere( np.all(rcl >= 0, axis=1) & np.all(rcl < (self.grid_dim),axis=1)).reshape(-1)
        idx = rcl[valid,0] + rcl[valid,1] * self.grid_dim[0] + rcl[valid,2] * self.grid_dim[0] * self.grid_dim[1]
        return idx, valid

    def addPointCloud(self, point_cld):
        """
        向体素网格中添加点云。

        参数:
        - point_cld: 要添加的点云。
        """
        grid_idx, cloud_idx = self.cloud2indices(point_cld[:, 0:3])
        for g_idx, c_idx in zip(grid_idx, cloud_idx):
            if (self.grid[g_idx].isEmpty()):
                self.used_voxel.append(g_idx)
            self.grid[g_idx].addPoint(point_cld[c_idx, :])

    def getPointCloud(self):
        """
        获取体素网格中非空体素的点。

        返回:
        - 非空体素的点数组。
        """
        return np.asarray([self.grid[i].getValue() for i in self.used_voxel])

class AverageVoxel():
    """
    AverageVoxel类用于存储体素中的点并计算它们的平均值。
    """
    def __init__(self,point_dim):
        """
        初始化AverageVoxel类。

        参数:
        - point_dim: 点的维度。
        """
        self.point = np.zeros((point_dim))
        self.weight = 0

    def addPoint(self, point):
        """
        向体素中添加一个点。

        参数:
        - point: 要添加的点。
        """
        self.point += point
        self.weight += 1

    def getValue(self):
        """
        获取体素中点的平均值和权重。

        返回:
        - 点的平均值和权重的数组。
        """
        dim = self.point.shape[0]
        val = np.ones( (dim+1),dtype= np.float32 ) *self.weight
        val[0:dim]= self.point/self.weight
        return val

    def isEmpty(self):
        """
        判断体素是否为空。

        返回:
        - 如果体素为空则返回True，否则返回False。
        """
        return self.weight == 0

class AverageGrid(VoxelGrid):
    """
    AverageGrid类继承自VoxelGrid类，用于创建和管理计算点平均值的体素网格。
    """
    def __init__(self, center, grid_size, voxel_size, point_dim=3):
        """
        初始化AverageGrid类。

        参数:
        - center: 体素网格的中心点。
        - grid_size: 体素网格的总大小。
        - voxel_size: 单个体素的大小。
        - point_dim: 点的维度，默认为3。
        """
        super().__init__(AverageVoxel, center, grid_size, voxel_size, point_dim=point_dim)

if __name__ == "__main__":
    center = np.array([0.0, 0.0, 0.0])
    grid_size = np.array([10.0, 10.0, 10.0])
    voxel_grid = AverageGrid(center, grid_size, 5)

    p1 = np.array([2.0, -1.0, -1.0])
    p2 = np.array([2.0, 1.0, 1.0])
    p3 = np.array([6.0, 1.2, 1.0])
    voxel_grid.addPoint(p1)
    voxel_grid.addPoint(p2)
    voxel_grid.addPoint(p3)
    voxel_grid.addPoint(p3)
    print('v1 used voxel',voxel_grid.used_voxel)
    p = voxel_grid.getPointCloud()
    print('v1 points',p.shape,p)

    voxel_grid2 = AverageGrid(center, grid_size, 5)
    pcs = (p1[np.newaxis,:],p2[np.newaxis,:],p3[np.newaxis,:],p3[np.newaxis,:])
    cld = np.concatenate(pcs)
    print('cld',cld.shape)
    voxel_grid2.addPointCloud(cld)
    p2 = voxel_grid2.getPointCloud()
    print('v2 points',p2.shape,p2)
    print('diff',p-p2)
