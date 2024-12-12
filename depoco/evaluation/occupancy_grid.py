import numpy as np


class OccupancyGrid():
    def __init__(self, center: np.array, resolution: np.array, size_meter: np.array):
        """
        初始化占用网格

        Args:
            center (np.array): 网格中心点 (1x3)
            resolution (np.array): 网格分辨率 (1x3)
            size_meter (np.array): 网格大小 (1x3)
        """
        self.center = center  # 网格中心点
        self.resolution = resolution  # 网格分辨率
        self.size_meter = size_meter  # 网格大小（米）
        self.size = np.ceil(size_meter / resolution)  # 网格尺寸（行x列x层）
        self.min_corner = self.center - self.size * self.resolution / 2  # 网格最小角点

        self.grid = np.zeros(np.squeeze(self.size.astype('int')), dtype='bool')  # 初始化网格为全零布尔数组

    def addPoints(self, points: np.array):
        """
        向网格中添加点

        Args:
            points (np.array): 点云数据 (Nx3)
        """
        points_l = np.floor((points - self.min_corner) / self.resolution).astype('int')  # 将点坐标转换为网格索引
        valids = np.all((points_l >= 0) & (points_l < self.size), axis=1)  # 检查点是否在网格范围内
        points_l = points_l[valids, :]  # 过滤出有效点
        self.grid[points_l[:, 0], points_l[:, 1], points_l[:, 2]] = True  # 在网格中标记有效点


def gridIOU(gt_grid: np.array, source_grid: np.array):
    """
    计算两个网格的交并比 (IOU)

    Args:
        gt_grid (np.array): 真实网格
        source_grid (np.array): 源网格

    Returns:
        float: 交并比 (IOU)
    """
    return np.sum((gt_grid & source_grid)) / np.sum((gt_grid | source_grid))


if __name__ == "__main__":
    # 创建一个占用网格实例
    occ_grid = OccupancyGrid(center=np.zeros((1, 3)), resolution=np.full((1, 3), 2), size_meter=np.full((1, 3), 6))
    print('初始占用网格 \n', occ_grid.grid)

    # 定义点云数据
    points = np.array([
        # [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7],
        [2.5, 2.5, 2.5],
        [-2.5, 2.5, 2.5],
        [100, 100, 100]
    ])
    # 向网格中添加点
    occ_grid.addPoints(points)
    print('添加点后的占用网格 \n', occ_grid.grid)

    # 创建另一个占用网格实例
    occ_grid2 = OccupancyGrid(center=np.zeros((1, 3)), resolution=np.full((1, 3), 2), size_meter=np.full((1, 3), 6))
    points = np.array([
        # [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7],
        [2.5, -2.5, 2.5],
        [-2.5, 2.5, 2.5],
        [100, 100, 100]
    ])
    # 向网格中添加点
    occ_grid2.addPoints(points)
    print('第二个占用网格 \n', occ_grid2.grid)

    # 计算两个网格的交并比 (IOU)
    iou = gridIOU(occ_grid.grid, occ_grid2.grid)
    print('交并比 (IOU)', iou)

    # 创建一个更大的占用网格实例
    big_grid = OccupancyGrid(center=np.zeros((1, 3)), resolution=np.array((0.2, 0.2, 0.1)), size_meter=np.array((40, 40, 15.0)))
    print('大网格的形状', big_grid.grid.shape)
