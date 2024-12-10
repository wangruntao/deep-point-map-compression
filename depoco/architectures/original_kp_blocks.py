# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/

def gather(x, idx, method=0):
    """
    实现自定义的 gather 操作以加快反向传播速度。
    :param x: 输入张量，形状为 [N, D_1, ... D_d]
    :param idx: 索引张量，形状为 [n_1, ..., n_m]
    :param method: 选择的方法
    :return: x[idx]，形状为 [n_1, ..., n_m, D_1, ... D_d]
    """
    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)  # 增加一个维度
        x = x.expand((-1, idx.shape[-1], -1))  # 扩展维度
        idx = idx.unsqueeze(2)  # 增加一个维度
        idx = idx.expand((-1, -1, x.shape[-1]))  # 扩展维度
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)  # 增加一个维度
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)  # 扩展维度
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)  # 增加一个维度
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)  # 扩展维度
        return x.gather(0, idx)
    else:
        raise ValueError('未知方法')

def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    计算半径高斯（距离的高斯）。
    :param sq_r: 输入半径的平方 [dn, ..., d1, d0]
    :param sig: 高斯的范围 [d1, d0] 或 [d0] 或 float
    :return: 高斯值 [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))

def closest_pool(x, inds):
    """
    从最近的邻居中池化特征。注意：此函数假设邻居已排序。
    :param x: 特征矩阵 [n1, d]
    :param inds: 池化索引 [n2, max_num]，仅使用第一列进行池化
    :return: 池化后的特征矩阵 [n2, d]
    """
    # 添加一行最小特征作为阴影池
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    # 获取每个池化位置的特征 [n2, d]
    return gather(x, inds[:, 0])

def max_pool(x, inds):
    """
    使用最大值池化特征。
    :param x: 特征矩阵 [n1, d]
    :param inds: 池化索引 [n2, max_num]
    :return: 池化后的特征矩阵 [n2, d]
    """
    # 添加一行最小特征作为阴影池
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    # 获取每个池化位置的所有特征 [n2, max_num, d]
    pool_features = gather(x, inds)
    # 池化最大值 [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features

def global_average(x, batch_lengths):
    """
    在批次池化上执行全局平均。
    :param x: 输入特征 [N, D]
    :param batch_lengths: 批次长度列表 [B]
    :return: 平均特征 [B, D]
    """
    # 遍历批次中的每个点云
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):
        # 对每个批次点云计算平均特征
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))
        # 更新索引
        i0 += length
    # 在每个批次中平均特征
    return torch.stack(averaged_features)

# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#

class KPConv(nn.Module):

    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                 fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                 deformable=False, modulated=False):
        """
        初始化 KPConvDeformable 的参数。
        :param kernel_size: 核心点的数量
        :param p_dim: 点空间的维度
        :param in_channels: 输入特征的维度
        :param out_channels: 输出特征的维度
        :param KP_extent: 每个核心点的影响范围
        :param radius: 用于初始化核心点的半径
        :param fixed_kernel_points: 固定某些核心点的位置（'none', 'center' 或 'verticals'）
        :param KP_influence: 核心点的影响函数（'constant', 'linear', 'gaussian'）
        :param aggregation_mode: 选择求和影响或仅保留最近的（'closest', 'sum'）
        :param deformable: 是否可变形
        :param modulated: 是否调制权重
        """
        super(KPConv, self).__init__()

        # 保存参数
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated

        # 运行变量，包含变形核心点与输入点的距离（用于正则化损失）
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # 初始化权重
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32),
                                 requires_grad=True)

        # 初始化偏置权重
        if deformable:
            if modulated:
                self.offset_dim = (self.p_dim + 1) * self.K
            else:
                self.offset_dim = self.p_dim * self.K
            self.offset_conv = KPConv(self.K,
                                      self.p_dim,
                                      self.in_channels,
                                      self.offset_dim,
                                      KP_extent,
                                      radius,
                                      fixed_kernel_points=fixed_kernel_points,
                                      KP_influence=KP_influence,
                                      aggregation_mode=aggregation_mode)
            self.offset_bias = Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)

        else:
            self.offset_dim = None
            self.offset_conv = None
            self.offset_bias = None

        # 重置参数
        self.reset_parameters()

        # 初始化核心点
        self.kernel_points = self.init_KP()

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.deformable:
            nn.init.zeros_(self.offset_bias)
        return

    def init_KP(self):
        """
        在球体内初始化核心点位置。
        :return: 核心点张量
        """
        K_points_numpy = getKernelPoints(self.radius, self.K)
        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False)

    def forward(self, q_pts, s_pts, neighb_inds, x):
        """
        前向传播函数。
        :param q_pts: 查询点 [n_points, dim]
        :param s_pts: 支持点 [n_points, dim]
        :param neighb_inds: 邻居索引 [n_points, n_neighbors]
        :param x: 输入特征 [n_points, in_channels]
        :return: 输出特征 [n_points, out_channels]
        """
        ###################
        # Offset generation
        ###################
        if self.deformable:
            # 生成偏置
            self.offset_features = self.offset_conv(q_pts, s_pts, neighb_inds, x) + self.offset_bias
            if self.modulated:
                # 获取偏置（归一化尺度）和调制
                unscaled_offsets = self.offset_features[:, :self.p_dim * self.K]
                unscaled_offsets = unscaled_offsets.view(-1, self.K, self.p_dim)
                modulations = 2 * torch.sigmoid(self.offset_features[:, self.p_dim * self.K:])
            else:
                # 获取偏置（归一化尺度）
                unscaled_offsets = self.offset_features.view(-1, self.K, self.p_dim)
                modulations = None
            # 调整偏置尺度
            offsets = unscaled_offsets * self.KP_extent
        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################
        # 添加一个假点作为阴影邻居
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)
        # 获取邻居点 [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]
        # 中心化每个邻域
        neighbors = neighbors - q_pts.unsqueeze(1)
        # 应用偏置到核心点 [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points
        # 获取所有差值矩阵 [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points
        # 获取平方距离 [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)
        # 优化：忽略超出变形核心点范围的点
        if self.deformable:
            # 保存距离用于损失
            self.min_d2, _ = torch.min(sq_distances, dim=1)
            # 邻居在核心点范围内的布尔值 [n_points, n_neighbors]
            in_range = torch.any(sq_distances < self.KP_extent ** 2, dim=2).type(torch.int32)
            # 新的最大邻居数
            new_max_neighb = torch.max(torch.sum(in_range, dim=1))
            # 每行邻居中在范围内的索引 [n_points, new_max_neighb]
            neighb_row_bool, neighb_row_inds = torch.topk(in_range, new_max_neighb.item(), dim=1)
            # 获取新的邻居索引 [n_points, new_max_neighb]
            new_neighb_inds = neighb_inds.gather(1, neighb_row_inds, sparse_grad=False)
            # 获取新的距离到核心点 [n_points, new_max_neighb, n_kpoints]
            neighb_row_inds.unsqueeze_(2)
            neighb_row_inds = neighb_row_inds.expand(-1, -1, self.K)
            sq_distances = sq_distances.gather(1, neighb_row_inds, sparse_grad=False)
            # 新的阴影邻居必须指向最后一个阴影点
            new_neighb_inds *= neighb_row_bool
            new_neighb_inds -= (neighb_row_bool.type(torch.int64) - 1) * int(s_pts.shape[0] - 1)
        else:
            new_neighb_inds = neighb_inds

        # 获取核心点影响 [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)
        elif self.KP_influence == 'linear':
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)
        elif self.KP_influence == 'gaussian':
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('未知的影响函数类型 (config.KP_influence)')

        # 如果模式为最近，只有最近的核心点可以影响每个点
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)
        elif self.aggregation_mode != 'sum':
            raise ValueError("未知的卷积模式。应该是 'closest' 或 'sum'")

        # 添加一个零特征作为阴影邻居
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
        # 获取每个邻域的特征 [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds, method=0)
        # 应用距离权重 [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)
        # 应用调制
        if self.deformable and self.modulated:
            weighted_features *= modulations.unsqueeze(2)
        # 应用网络权重 [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        kernel_outputs = torch.matmul(weighted_features, self.weights)
        # 卷积求和 [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)

    def __repr__(self):
        return 'KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})'.format(self.radius,
                                                                              self.in_channels,
                                                                              self.out_channels)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

def block_decider(block_name, radius, in_dim, out_dim, layer_ind, config):
    """
    根据块名称决定使用哪种块。
    :param block_name: 块名称
    :param radius: 当前半径
    :param in_dim: 输入特征维度
    :param out_dim: 输出特征维度
    :param layer_ind: 层索引
    :param config: 参数配置
    :return: 块对象
    """
    if block_name == 'unary':
        return UnaryBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)
    elif block_name in ['simple', 'simple_deformable', 'simple_invariant', 'simple_equivariant', 'simple_strided', 'simple_deformable_strided', 'simple_invariant_strided', 'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    elif block_name in ['resnetb', 'resnetb_invariant', 'resnetb_equivariant', 'resnetb_deformable', 'resnetb_strided', 'resnetb_deformable_strided', 'resnetb_equivariant_strided', 'resnetb_invariant_strided']:
        return ResnetBottleneckBlock(block_name, in_dim, out_dim, radius, layer_ind, config)
    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)
    elif block_name == 'global_average':
        return GlobalAverageBlock()
    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)
    else:
        raise ValueError('架构定义中未知的块名称 : ' + block_name)

class BatchNormBlock(nn.Module):
    """
    初始化一个批归一化块。如果网络不使用批归一化，替换为偏差。
    :param in_dim: 输入特征维度
    :param use_bn: 是否使用批归一化
    :param bn_momentum: 批归一化动量
    """
    def __init__(self, in_dim, use_bn, bn_momentum):
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:
            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose