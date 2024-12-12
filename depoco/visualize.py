import depoco.utils.point_cloud_utils as pcu
import argparse
import ruamel.yaml as yaml
from depoco.trainer import DepocoNetTrainer
import torch

if __name__ == "__main__":
    print('Hello')

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser("./sample_net_trainer.py")

    # 添加配置文件路径参数
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='配置文件路径。参见 /config 目录下的示例文件',
    )

    # 添加要可视化的地图数量参数
    parser.add_argument(
        '--number', '-n',
        type=int,
        default=5,
        help='要可视化的地图数量',
    )

    # 解析命令行参数
    FLAGS, unparsed = parser.parse_known_args()

    print('传递的参数')

    # 从配置文件中加载配置
    config = yaml.safe_load(open(FLAGS.config, 'r'))
    print('已加载 YAML 配置')

    # 初始化训练器
    trainer = DepocoNetTrainer(config)
    trainer.loadModel(best=False)  # 加载模型，不加载最佳模型
    print('已初始化训练器')

    # 获取训练集中的子地图
    for i, batch in enumerate(trainer.submaps.getOrderedTrainSet()):
        with torch.no_grad():  # 不计算梯度，用于推理
            points_est, nr_emb_points = trainer.encodeDecode(batch)  # 编码和解码点云
            print(f'嵌入点数: {nr_emb_points}, 输出点数: {points_est.shape[0]}')

            # 可视化点云
            pcu.visPointCloud(points_est.detach().cpu().numpy())

        # 如果已经处理了指定数量的地图，则退出循环
        if i + 1 >= FLAGS.number:
            break
