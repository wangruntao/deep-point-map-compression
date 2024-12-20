import depoco.datasets.kitti2voxel as kitti2voxel
import depoco.datasets.submap_handler as submap_handler
import depoco.utils.point_cloud_utils as pcu
import ruamel.yaml as yaml
import argparse
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil

def nearestPoint(points, qx, qy):
    """
    找到给定坐标最近的点的索引。

    参数:
    - points: 点的数组。
    - qx: X 坐标。
    - qy: Y 坐标。

    返回:
    - min_idx: 最近点的索引。
    """
    dx = points[:, 0] - qx
    dy = points[:, 1] - qy
    d = (dx**2 + dy**2)
    min_idx = np.argmin(d)
    return min_idx

def drawSubmap(submap: submap_handler.SubMap):
    """
    绘制子地图的点云。

    参数:
    - submap: 子地图对象。
    """
    pcu.visPointCloud(submap.getPoints())

def saveFiles(files, source_path):
    """
    保存文件列表到指定路径。

    参数:
    - files: 文件列表。
    - source_path: 源路径。
    """
    seq = files[0].split('/')[-2]
    print('seq', seq)
    filenames = [f.split("/")[-1] for f in files]
    print(filenames)
    myfile = open(source_path + 'validation_files.txt', 'w')
    myfile.write('#source_file target_file\n')
    for f in filenames:
        myfile.write(f + ' ' + seq + '_' + f + ' \n')
    myfile.close()

def moveFiles(validation_files_txt, source_path, target_path, undo=False):
    """
    移动文件。

    参数:
    - validation_files_txt: 包含文件列表的文本文件。
    - source_path: 源路径。
    - target_path: 目标路径。
    - undo: 是否撤销移动操作。
    """
    f = open(validation_files_txt, "r")
    for i, x in enumerate(f):
        if i > 0:
            print(x)
            source = source_path + x.split(' ')[0]
            target = target_path + x.split(' ')[1]
            if undo:
                shutil.move(target, source)
            else:
                shutil.move(source, target)
    f.close()

class Splitter():
    def __init__(self, path):
        """
        初始化Splitter类。

        参数:
        - path: 数据路径。
        """
        self.path = path
        self.submaps = submap_handler.createSubmaps([path])
        print('submaps', len(self.submaps))
        self.poses = np.loadtxt(path + 'key_poses.txt')
        self.poses = np.reshape(self.poses, (self.poses.shape[0], 4, 4))
        self.xy = self.poses[:, 0:2, -1]
        self.current_idx = None
        self.validation_idx = []
        print('xy shape', self.xy.shape)

    def onclick(self, event):
        """
        鼠标点击事件处理函数。

        参数:
        - event: 鼠标事件。
        """
        x = event.xdata
        y = event.ydata
        print('x', x, 'y', y)
        min_idx = nearestPoint(self.xy, x, y)
        print('min_idx', min_idx, 'pt', self.xy[min_idx, :])
        self.ax.plot(self.xy[min_idx, 0], self.xy[min_idx, 1], 'xr')
        self.fig.canvas.draw()
        drawSubmap(self.submaps[min_idx])
        self.current_idx = min_idx
        print('Use in validation set, press y')

    def keyPressed(self, event):
        """
        键盘按键事件处理函数。

        参数:
        - event: 键盘事件。
        """
        if event.key == 'y':
            print('hey you pressed yes')
            if self.current_idx not in self.validation_idx:
                self.validation_idx.append(int(self.current_idx))
                print('Aded map', self.current_idx, 'to the validation set')
                self.ax.plot(self.xy[self.current_idx, 0], self.xy[self.current_idx, 1], 'og')
                self.fig.canvas.draw()

    def draw(self):
        """
        绘制子地图并处理用户交互。
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        bit = self.fig.canvas.mpl_connect('key_press_event', self.keyPressed)
        plt.grid()
        self.ax.plot(self.xy[:, 0], self.xy[:, 1], '.b')
        plt.axis('equal')
        plt.show()
        print('the following:', len(self.validation_idx), 'submaps can be used for validation. For saving the locations press y')
        if input() == 'y':
            print('saved')
            validation_poses = np.reshape(self.poses[self.validation_idx, :], (len(self.validation_idx), 16))
            np.savetxt(self.path + 'validation_poses.txt', validation_poses)
            files = [self.submaps[i].file for i in self.validation_idx]
            print('files', files)
            saveFiles(files, self.path)
        else:
            print('Not saved.')

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser("./train_test_splitter.py")
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=False,
        default='../config/arch/depoco.yaml',
        help='架构配置文件的 YAML 路径。参见 /config/arch 示例。'
    )

    FLAGS, unparsed = parser.parse_known_args()
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))

    # input_folder = FLAGS.dataset + '/sequences/00/'
    # calibration = kitti2voxel.parse_calibration(os.path.join(input_folder, "calib.txt"))
    # poses = kitti2voxel.parse_poses(os.path.join(input_folder, "poses.txt"), calibration)
    # idx, keypose, d= kitti2voxel.getKeyPoses(poses,delta=ARCH["grid"]["pose_distance"])

    # xy = np.asarray(poses)
    # xy = xy[:,0:2,-1]
    # x = xy[:,0]
    # y = xy[:,1]
    # print(xy.shape)
    # plt.figure
    # plt.plot(xy[:,0], xy[:,1])
    # plt.plot(xy[idx,0], xy[idx,1],'xr')
    # plt.axis('equal')

    ## TODO: 修改路径
    target_path = '/media/lwiesmann/WiesmannIPB/data/data_kitti/dataset/submaps/40m_ILEN/validation/'
    # splitter = Splitter(path)
    # splitter.draw()

    for i in range(10):
        try:
            path = '/media/lwiesmann/WiesmannIPB/data/data_kitti/dataset/submaps/40m_ILEN/0' + str(i) + '/'
            moveFiles(path + 'validation_files.txt', source_path=path, target_path=target_path, undo=False)
        except:
            print('Kitti {i} file not found')

    # moveFiles(path + 'validation_files.txt', source_path=path, target_path=target_path, undo=False)
