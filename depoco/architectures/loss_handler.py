import torch 
import depoco.architectures.original_kp_blocks as okp
import chamfer3D.dist_chamfer_3D
import depoco.architectures.network_blocks as network_blocks

def linDeconvRegularizer(net, weight, gt_points):
    """
    计算线性反卷积层的正则化损失。

    参数:
    net (torch.nn.Module): 网络模型。
    weight (float): 正则化项的权重。
    gt_points (torch.Tensor): 真实点云数据。

    返回:
    torch.Tensor: 计算得到的正则化损失。
    """
    cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()  # 初始化 Chamfer 距离计算对象
    loss = torch.tensor(0.0, dtype=torch.float32, device=gt_points.device)  # 初始化损失

    for m in net.modules():  # 遍历网络中的所有模块
        if isinstance(m, network_blocks.LinearDeconv) or isinstance(m, network_blocks.AdaptiveDeconv):
            # 计算真实点云与当前模块点云之间的 Chamfer 距离
            d_map2transf, d_transf2map, idx3, idx4 = cham_loss(gt_points.unsqueeze(0), m.points.unsqueeze(0))
            # 累加 Chamfer 距离损失
            loss += (0.5 * d_map2transf.mean() + 0.5 * d_transf2map.mean())

    return weight * loss  # 返回加权后的损失

# From KPCONV
def p2p_fitting_regularizer(net, deform_fitting_power=1.0, repulse_extent=1.2):
    """
    计算点到点拟合正则化损失和排斥正则化损失。

    参数:
    net (torch.nn.Module): 网络模型。
    deform_fitting_power (float): 拟合损失的权重，默认为1.0。
    repulse_extent (float): 排斥损失的距离阈值，默认为1.2。

    返回:
    torch.Tensor: 计算得到的正则化损失。
    """
    l1 = torch.nn.L1Loss()  # 初始化 L1 损失函数
    fitting_loss = 0  # 初始化拟合损失
    repulsive_loss = 0  # 初始化排斥损失

    for m in net.modules():  # 遍历网络中的所有模块
        if isinstance(m, okp.KPConv) and m.deformable:
            # 计算拟合损失
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)  # 归一化最小距离平方
            fitting_loss += l1(KP_min_d2, torch.zeros_like(KP_min_d2))  # 累加拟合损失

            # 计算排斥损失
            KP_locs = m.deformed_KP / m.KP_extent  # 归一化关键点位置
            for i in range(m.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()  # 其他关键点
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))  # 计算距离
                rep_loss = torch.sum(torch.clamp_max(distances - repulse_extent, max=0.0) ** 2, dim=1)  # 计算排斥损失
                repulsive_loss += l1(rep_loss, torch.zeros_like(rep_loss)) / m.K  # 累加排斥损失

    return deform_fitting_power * (2 * fitting_loss + repulsive_loss)  # 返回加权后的总损失
