#!/usr/bin/env python3

# 导入必要的模块
from depoco.trainer import DepocoNetTrainer  # 导入训练器类
from ruamel import yaml  # 导入YAML解析库
import argparse  # 导入命令行参数解析库
import time  # 导入时间模块
import depoco.utils.point_cloud_utils as pcu  # 导入点云实用工具
import os  # 导入操作系统接口模块

if __name__ == "__main__":
    print('Hello')  # 打印欢迎信息

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("./evaluate.py")

    # 添加命令行参数
    parser.add_argument(
        '--config_cfg', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='配置文件的路径。参见 /config/depoco.yaml 示例。默认值为 config/depoco.yaml',
    )
    parser.add_argument(
        '--file_ext', '-fe',
        type=str,
        required=False,
        default='',
        help='在输出文件名后附加的字符串',
    )

    # 解析命令行参数
    FLAGS, unparsed = parser.parse_known_args()

    print('传递的参数')  # 打印提示信息
    # 读取配置文件
    config = yaml.safe_load(open(FLAGS.config_cfg, 'r'))
    print('已加载 YAML 配置文件')  # 打印提示信息
    print('配置文件路径:', FLAGS.config_cfg)  # 打印配置文件路径

    # 初始化训练器
    trainer = DepocoNetTrainer(config)
    print('训练器初始化完成')  # 打印提示信息

    # 记录开始时间
    ts = time.time()

    # 进行测试
    test_dict = trainer.test(best=True)
    print('评估时间:', time.time() - ts)  # 打印评估时间

    # 打印部分评估结果
    print('重建误差:', test_dict['mapwise_reconstruction_error'][0:10])
    print('内存使用:', test_dict['memory'][0:10])

    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(config['evaluation']['out_dir']):
        os.makedirs(config['evaluation']['out_dir'])

    # 构建输出文件路径
    file = config['evaluation']['out_dir'] + trainer.experiment_id + FLAGS.file_ext + ".pkl"

    # 保存评估结果
    pcu.save_obj(test_dict, file)
