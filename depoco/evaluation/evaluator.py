import torch
import chamfer3D.dist_chamfer_3D
import depoco.evaluation.occupancy_grid as occupancy_grid
import numpy as np
from collections import defaultdict
import torch.nn as nn


class Evaluator():
    def __init__(self, config):
        self.config = config
        self.cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()  # Chamfer距离计算工具
        self.running_loss = 0.0  # 运行损失
        self.n = 0  # 样本计数
        self.eval_results = defaultdict(list)  # 评估结果存储
        self.l1_loss = nn.L1Loss()  # L1损失函数
        self.l2_loss = nn.MSELoss()  # L2损失函数

    def chamferDist(self, gt_points: torch.tensor, source_points: torch.tensor):
        """计算两个点云之间的Chamfer距离

        Args:
            gt_points (torch.tensor): 真实点云
            source_points (torch.tensor): 源点云

        Returns:
            float: Chamfer距离
        """
        gt_points = gt_points.cuda().detach()  # 将真实点云移动到GPU并分离梯度
        source_points = source_points.cuda().detach()  # 将源点云移动到GPU并分离梯度
        d_gt2source, d_source2gt, idx3, idx4 = self.cham_loss(
            gt_points.unsqueeze(0), source_points.unsqueeze(0))  # 计算Chamfer距离
        # 平均平方距离 (gt -> source) + 平均平方距离 (source -> gt)
        loss = (d_gt2source.mean() + d_source2gt.mean())  # /2 FIXME:
        self.running_loss += loss.cpu().item()  # 更新运行损失
        self.n += 1  # 更新样本计数
        return loss

    def evaluate(self, gt_points: torch.tensor, source_points: torch.tensor, gt_normals=None):
        """评估两个点云之间的Chamfer距离和其他指标

        Args:
            gt_points (torch.tensor): 真实点云
            source_points (torch.tensor): 源点云
            gt_normals (torch.tensor, optional): 真实点云的法线. Defaults to None.

        Returns:
            dict: 评估结果
        """
        ##### 计算Chamfer距离 ######
        gt_points = gt_points.cuda().detach()  # 将真实点云移动到GPU并分离梯度
        source_points = source_points.cuda().detach()  # 将源点云移动到GPU并分离梯度
        d_gt2source, d_source2gt, idx3, idx4 = self.cham_loss(
            gt_points.unsqueeze(0), source_points.unsqueeze(0))  # 计算Chamfer距离
        idx3 = idx3.long().squeeze()  # 转换索引为长整型并压缩维度
        idx4 = idx4.long().squeeze()  # 转换索引为长整型并压缩维度
        # 平均平方距离 (gt -> source) + 平均平方距离 (source -> gt)
        chamfer_dist = (d_gt2source.mean() + d_source2gt.mean()) / 2
        chamfer_dist_abs = (d_gt2source.sqrt().mean() + d_source2gt.sqrt().mean()) / 2
        out_dict = {}
        out_dict['chamfer_dist'] = chamfer_dist.cpu().item()  # 记录Chamfer距离
        self.eval_results['chamfer_dist'].append(out_dict['chamfer_dist'])  # 存储Chamfer距离
        out_dict['chamfer_dist_abs'] = chamfer_dist_abs.cpu().item()  # 记录绝对Chamfer距离
        self.eval_results['chamfer_dist_abs'].append(out_dict['chamfer_dist_abs'])  # 存储绝对Chamfer距离

        ############ 计算PSNR ##############
        if gt_normals is not None:  # 如果有法线，计算PSNR
            gt_normals = gt_normals.cuda().detach()  # 将法线移动到GPU并分离梯度
            d_plane_gt2source = torch.sum(
                (gt_points - source_points[idx3, :]) * gt_normals, dim=1)  # 计算平面距离 (gt -> source)
            d_plane_source2gt = torch.sum(
                (source_points - gt_points[idx4, :]) * gt_normals[idx4, :], dim=1)  # 计算平面距离 (source -> gt)
            chamfer_plane = (d_plane_gt2source.abs().mean() + d_plane_source2gt.abs().mean()) / 2
            out_dict['chamfer_dist_plane'] = chamfer_plane.cpu().item()  # 记录平面Chamfer距离
            self.eval_results['chamfer_dist_plane'].append(out_dict['chamfer_dist_plane'])  # 存储平面Chamfer距离

        ###### 计算IOU #######
        gt_points_np = gt_points.cpu().numpy()  # 将真实点云转换为NumPy数组
        source_points_np = source_points.cpu().numpy()  # 将源点云转换为NumPy数组

        center = (np.max(gt_points_np, axis=0, keepdims=True) + np.min(gt_points_np, axis=0, keepdims=True)) / 2  # 计算中心点
        resolution = np.array([self.config['evaluation']['iou_grid']['resolution']])  # 获取分辨率
        size_meter = np.array([self.config['grid']['size']])  # 获取网格大小
        gt_grid = occupancy_grid.OccupancyGrid(center=center, resolution=resolution, size_meter=size_meter)  # 创建真实点云的占用网格
        gt_grid.addPoints(gt_points_np)  # 添加真实点云
        source_grid = occupancy_grid.OccupancyGrid(center=center, resolution=resolution, size_meter=size_meter)  # 创建源点云的占用网格
        source_grid.addPoints(source_points_np)  # 添加源点云

        out_dict['iou'] = occupancy_grid.gridIOU(gt_grid.grid, source_grid.grid)  # 计算IOU
        self.eval_results['iou'].append(out_dict['iou'])  # 存储IOU

        return out_dict

    def getRunningLoss(self):
        """返回运行损失，并重置损失值

        Returns:
            float: 平均Chamfer距离
        """
        if self.n == 0:
            return None
        loss = self.running_loss / self.n  # 计算平均损失
        self.running_loss = 0.0  # 重置运行损失
        self.n = 0  # 重置样本计数
        return loss
