import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
from ruamel import yaml
import depoco.utils.point_cloud_utils as pcu

def plotResults(files, x_key, y_key, ax, draw_line=False, label=None, set_lim=True):
    """
    绘制结果图。

    参数:
    files (list): 包含评估结果文件路径的列表。
    x_key (str): X轴数据对应的键。
    y_key (str): Y轴数据对应的键。
    ax (matplotlib.axes.Axes): 子图对象。
    draw_line (bool): 是否绘制折线图，默认为False。
    label (str): 图例标签，默认为None。
    set_lim (bool): 是否设置坐标轴范围，默认为True。
    """
    x = []
    y = []
    for f in files:
        eval_dict = pcu.load_obj(f)  # 加载评估结果文件
        if ((x_key in eval_dict.keys()) & (y_key in eval_dict.keys())):
            # 确保 x_key 和 y_key 在评估结果中存在
            for v in eval_dict.values():
                v = np.array(v)  # 将值转换为numpy数组

            if not draw_line:
                # 绘制散点图并标注文件名
                ax.plot(np.mean(eval_dict[x_key]), np.mean(eval_dict[y_key]), '.')
                ax.text(np.mean(eval_dict[x_key]), np.mean(eval_dict[y_key]), f.split('/')[-1][:-4])

            x.append(np.mean(eval_dict[x_key]))  # 记录X轴数据的平均值
            y.append(np.mean(eval_dict[y_key]))  # 记录Y轴数据的平均值

    if draw_line:
        # 绘制折线图
        line, = ax.plot(x, y, '-x', label=label)
        # line.set_label(label)

    ax.set_xlabel(x_key)  # 设置X轴标签
    ax.set_ylabel(y_key)  # 设置Y轴标签

    if set_lim:
        # 设置坐标轴范围
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

def genPlots(files, f, ax, draw_line=False, label=None, x_key='memory'):
    """
    生成多个子图。

    参数:
    files (list): 包含评估结果文件路径的列表。
    f (matplotlib.figure.Figure): 图形对象。
    ax (numpy.ndarray): 子图对象数组。
    draw_line (bool): 是否绘制折线图，默认为False。
    label (str): 图例标签，默认为None。
    x_key (str): X轴数据对应的键，默认为'memory'。
    """
    # 绘制不同指标的子图
    plotResults(files, x_key=x_key, y_key='chamfer_dist_abs', ax=ax[0], draw_line=draw_line, label=label)
    plotResults(files, x_key=x_key, y_key='chamfer_dist_plane', ax=ax[1], draw_line=draw_line, label=label)
    plotResults(files, x_key=x_key, y_key='iou', ax=ax[2], draw_line=draw_line, label=label)

if __name__ == "__main__":
    ####### radius fct ##############
    path = 'experiments/results/kitti/'  # 设置结果文件路径
    files = sorted(glob.glob(path + '*.pkl'))  # 获取所有评估结果文件

    f, ax = plt.subplots(1, 3)  # 创建包含3个子图的图形
    f.suptitle('Radius FPS')  # 设置图形标题

    # 生成子图
    genPlots(files, f, ax, draw_line=True, label='our', x_key='bpp')

    for a in ax:
        a.grid()  # 显示网格
        a.legend()  # 显示图例

    plt.show()  # 显示图形
