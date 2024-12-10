import torch.nn as nn
import torch
import numpy as np
import octree_handler
import depoco.architectures.original_kp_blocks as o_kp_conv

##################################################
############ NETWORK BLOCKS Dictionary############
##################################################

def printNan(bla: torch.tensor, pre=''):
    """
    检查张量中是否存在NaN值，并打印相关信息。
    
    参数:
    - bla: 要检查的张量
    - pre: 打印前缀，用于标识张量来源
    """
    if (bla != bla).any():
        print(pre, 'NAN')

class Network(nn.Module):
    """
    定义一个神经网络模型，由多个网络块组成。
    """
    def __init__(self, config_list: list):
        """
        初始化Network类。
        
        参数:
        - config_list: 包含网络块配置的列表
        """
        super().__init__()
        blocks = []
        for config in config_list:
            blocks.append(getBlocks(config))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input_dict: dict):
        """
        前向传播函数。
        
        参数:
        - input_dict: 输入字典，包含点云数据和特征
        """
        return self.blocks(input_dict)

def getBlocks(config: dict):
    """
    根据配置生成网络块。
    
    参数:
    - config: 包含网络块配置的字典
    """
    config_it, block_type = blockConfig2Params(config)
    blocks = []
    for c in config_it:
        blocks.append(eval(block_type)(c))
    if len(config_it) == 1:
        return blocks[0]
    return nn.Sequential(*blocks)

def blockConfig2Params(config: dict):
    """
    将配置转换为参数列表和块类型。
    
    参数:
    - config: 包含网络块配置的字典
    
    返回:
    - 参数列表
    - 块类型
    """
    nr_blocks = config['number_blocks']
    if nr_blocks == 1:
        return [config['parameters']], config['type']

    config_list = []
    for i in range(nr_blocks):
        new_config = config["parameters"].copy()
        for k, v in zip(config["parameters"].keys(), config["parameters"].values()):
            if isinstance(v, list):
                if len(v) == nr_blocks:
                    new_config[k] = v[i]
                if k == 'subsampling_ratio':
                    new_config['cum_subsampling_ratio'] = np.cumprod([1.0] + v)[i]
        config_list.append(new_config)
    return config_list, config['type']

def dict2initParams(dict_, class_):
    """
    从字典中提取类初始化所需的参数。
    
    参数:
    - dict_: 包含参数的字典
    - class_: 目标类
    """
    init_params = class_.__init__.__code__.co_varnames
    print(f'init vars: \n {init_params}')
    params = {k: dict_[k] for k in dict_ if k in init_params}
    print(params)
    return params

def gridSampling(pcd: torch.tensor, resolution_meter=1.0, map_size=40):
    """
    对点云进行网格采样。
    
    参数:
    - pcd: 点云数据
    - resolution_meter: 网格分辨率
    - map_size: 地图大小
    """
    resolution = resolution_meter / map_size
    grid = torch.floor(pcd / resolution)
    center = (grid + 0.5) * resolution
    dist = ((pcd - center) ** 2).sum(dim=1)
    dist = dist / dist.max() * 0.7
    v_size = np.ceil(1 / resolution)
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size
    grid_d = grid_idx + dist
    idx_orig = torch.argsort(grid_d)
    unique, inverse = torch.unique_consecutive(grid_idx[idx_orig], return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    p = perm.cpu()
    i = inverse.cpu()
    idx = torch.empty(unique.shape, dtype=p.dtype).scatter_(0, i, p)
    return idx_orig[idx].tolist()

class GridSampleConv(nn.Module):
    """
    定义一个网格采样卷积层。
    """
    def __init__(self, config: dict):
        """
        初始化GridSampleConv类。
        
        参数:
        - config: 包含网络块配置的字典
        """
        super().__init__()
        self.relu = nn.LeakyReLU()
        in_fdim = config['in_fdim']
        out_fdim = config['out_fdim']
        self.preactivation = nn.Identity()
        if in_fdim > 1:
            pre_blocks = [nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if config['batchnorm']:
                pre_blocks.append(nn.BatchNorm1d(out_fdim))
            if config['relu']:
                pre_blocks.append(self.relu)
            self.preactivation = nn.Sequential(*pre_blocks)
        conf_in_fdim = out_fdim if in_fdim > 1 else in_fdim
        self.subsampling_dist = config['subsampling_dist'] * config['subsampling_factor']
        self.kernel_radius = max(config['min_kernel_radius'], config['kernel_radius'] * self.subsampling_dist) / 40
        KP_extent = self.kernel_radius / (config['num_kernel_points'] ** (1/3) - 1) * 1.5
        self.kp_conv = o_kp_conv.KPConv(kernel_size=config['num_kernel_points'],
                                        p_dim=3, in_channels=conf_in_fdim,
                                        out_channels=out_fdim,
                                        KP_extent=KP_extent, radius=self.kernel_radius,
                                        deformable=config['deformable'])
        self.max_nr_neighbors = config['max_nr_neighbors']
        self.map_size = config['map_size']
        self.octree = octree_handler.Octree()
        print('kernel radius', self.kernel_radius)
        post_layer = []
        if config['batchnorm']:
            post_layer.append(nn.BatchNorm1d(out_fdim))
        if config['relu']:
            post_layer.append(self.relu)
        post_layer.append(nn.Linear(in_features=out_fdim, out_features=out_fdim))
        if config['batchnorm']:
            post_layer.append(nn.BatchNorm1d(out_fdim))
        self.post_layer = nn.Sequential(*post_layer)
        self.shortcut = nn.Identity()
        if in_fdim != out_fdim:
            sc_blocks = [nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if config['batchnorm']:
                sc_blocks.append(nn.BatchNorm1d(out_fdim))
            self.shortcut = nn.Sequential(*sc_blocks)

    def forward(self, input_dict: dict) -> dict:
        """
        前向传播函数。
        
        参数:
        - input_dict: 输入字典，包含点云数据和特征
        """
        source = input_dict['points']
        source_np = source.detach().cpu().numpy()
        sample_idx = gridSampling(source, resolution_meter=self.subsampling_dist, map_size=self.map_size)
        self.octree.setInput(source_np)
        neighbors_index = self.octree.radiusSearchIndices(sample_idx, self.max_nr_neighbors, self.kernel_radius)
        neighbors_index = torch.from_numpy(neighbors_index).long().to(source.device)
        features = self.preactivation(input_dict['features'])
        features = self.kp_conv.forward(q_pts=input_dict['points'][sample_idx, :],
                                        s_pts=input_dict['points'],
                                        neighb_inds=neighbors_index,
                                        x=features)
        features = self.post_layer(features)
        input_dict['features'] = self.relu(self.shortcut(input_dict['features'][sample_idx, :]) + features)
        input_dict['points'] = input_dict['points'][sample_idx, :]
        return input_dict

class LinearLayer(nn.Module):
    """
    定义一个线性层。
    """
    def __init__(self, config: dict):
        """
        初始化LinearLayer类。
        
        参数:
        - config: 包含网络块配置的字典
        """
        super().__init__()
        blocks = [nn.Linear(in_features=config['in_fdim'], out_features=config['out_fdim'])]
        if 'relu' in config and config['relu']:
            blocks.append(nn.LeakyReLU())
        if 'batchnorm' in config and config['batchnorm']:
            blocks.append(nn.BatchNorm1d(num_features=config['out_fdim']))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input_dict: dict):
        """
        前向传播函数。
        
        参数:
        - input_dict: 输入字典，包含点云数据和特征
        """
        input_dict['features'] = self.blocks(input_dict['features'])
        return input_dict

class LinearDeconv(nn.Module):
    """
    定义一个线性反卷积层。
    """
    def __init__(self, config: dict):
        """
        初始化LinearDeconv类。
        
        参数:
        - config: 包含网络块配置的字典
        """
        super().__init__()
        self.config = config
        if config['estimate_radius']:
            self.kernel_radius = nn.Parameter(torch.tensor([config['kernel_radius']]), requires_grad=True).float()
        else:
            self.kernel_radius = config['kernel_radius']
        feature_space = config['inter_fdim'] if 'inter_fdim' in config.keys() else 128
        self.upsampling_rate = config['upsampling_rate']
        trans_blocks = [nn.Linear(config['in_fdim'], out_features=feature_space),
                        nn.LeakyReLU(),
                        nn.Linear(in_features=feature_space, out_features=3 * self.upsampling_rate),
                        nn.Tanh()]
        self.transl_nn = nn.Sequential(*trans_blocks)
        feature_blocks = [nn.Linear(config['in_fdim'], out_features=feature_space),
                          nn.LeakyReLU(),
                          nn.Linear(in_features=feature_space, out_features=config['out_fdim'] * self.upsampling_rate),
                          nn.LeakyReLU()]
        if config['use_batch_norm']:
            feature_blocks.append(nn.BatchNorm1d(config['out_fdim'] * self.upsampling_rate))
        self.feature_nn = nn.Sequential(*feature_blocks)
        self.tmp_i = 0
        self.points = None

    def forward(self, input_dict: dict):
        """
        前向传播函数。
        
        参数:
        - input_dict: 输入字典，包含点云数据和特征
        """
        p = input_dict['points']
        f = input_dict['features']
        delta = self.transl_nn(f)
        delta = delta.reshape((delta.shape[0], self.upsampling_rate, 3)) * self.kernel_radius
        p_new = (p.unsqueeze(1) + delta).reshape((delta.shape[0] * self.upsampling_rate, 3))
        f_new = self.feature_nn(f).reshape((delta.shape[0] * self.upsampling_rate, self.config['out_fdim']))
        self.tmp_i += 1
        self.points = p_new
        input_dict['points'] = p_new
        input_dict['features'] = f_new
        return input_dict

def getScalingFactor(upsampling_rate, nr_layer, layer=0):
    """
    计算每层的上采样因子。
    
    参数:
    - upsampling_rate: 总上采样率
    - nr_layer: 层数
    - layer: 当前层索引
    """
    sf = upsampling_rate ** (1 / nr_layer)
    factors = nr_layer * [round(sf)]
    sampling_factor = np.prod(factors)
    print(f'factors {factors}, upsampling rate {sampling_factor}, should: {upsampling_rate}')
    return factors[layer]

class AdaptiveDeconv(nn.Module):
    """
    定义一个自适应反卷积层。
    """
    def __init__(self, config: dict):
        """
        初始化AdaptiveDeconv类。
        
        参数:
        - config: 包含网络块配置的字典
        """
        super().__init__()
        self.config = config
        if config['estimate_radius']:
            self.kernel_radius = nn.Parameter(torch.tensor([config['kernel_radius']]), requires_grad=True).float()
        else:
            self.kernel_radius = config['kernel_radius']
        feature_space = config['inter_fdim'] if 'inter_fdim' in config.keys() else 128
        sub_rate = config['subsampling_fct_p1'] * config['subsampling_dist'] ** (-config['subsampling_fct_p2'])
        print('sub rate', sub_rate)
        self.upsampling_rate = getScalingFactor(upsampling_rate=1 / sub_rate, nr_layer=config['number_blocks'], layer=config['block_id'])
        trans_blocks = [nn.Linear(config['in