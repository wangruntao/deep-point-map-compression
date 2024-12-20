#!/usr/bin/env python3

import depoco.datasets.submap_handler as submap_handler
import depoco.evaluation.evaluator as evaluator
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim

import os
from torch.utils.tensorboard import SummaryWriter
from ruamel import yaml
import argparse
import depoco.architectures.network_blocks as network
import chamfer3D.dist_chamfer_3D

import depoco.utils.point_cloud_utils as pcu
import subprocess

# import depoco.utils.checkpoint as chkpt
import depoco.architectures.loss_handler as loss_handler
from tqdm.auto import trange, tqdm


class DepocoNetTrainer():
    def __init__(self, config):
        t_start = time.time()
        # 初始化参数
        config['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())
        self.config = config
        self.experiment_id = self.config["train"]["experiment_id"]

        # 加载子图数据
        t_sm = time.time()
        self.submaps = submap_handler.SubMapParser(config)
        print(f'Loaded Submaps ({time.time() - t_sm}s')

        # 设置最大点数和设备
        self.max_nr_pts = self.config["train"]["max_nr_pts"]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # 加载模型
        t_model = time.time()
        self.encoder_model = None
        self.decoder_model = None
        self.getModel(config)
        print(f'Loaded Model ({time.time() - t_model}s)')

        # 损失函数属性
        self.cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        self.pairwise_dist = nn.PairwiseDistance()
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.w_transf2map = self.config['train']['loss_weights']['transf2map']
        self.w_map2transf = self.config['train']['loss_weights']['map2transf']

        print(f'Init Trainer ({time.time() - t_start}s)')

    def getModel(self, config: dict):
        """加载指定的模型"""
        arch = self.config["network"]
        print(f"网络架构: {arch}")
        self.encoder_model = network.Network(
            config['network']['encoder_blocks'])
        self.decoder_model = network.Network(
            config['network']['decoder_blocks'])
        print(self.encoder_model)
        print(self.decoder_model)

    def loadModel(self, best: bool = True, device='cuda', out_dir=None):
        """加载预训练模型"""
        if out_dir is None:
            out_dir = self.config["network"]['out_dir']
        enc_path = out_dir + self.experiment_id + '/enc'
        dec_path = out_dir + self.experiment_id + '/dec'
        enc_path += '_best.pth' if best else '.pth'
        dec_path += '_best.pth' if best else '.pth'
        print("加载模型:", enc_path, ",", dec_path)
        if os.path.isfile(enc_path) and os.path.isfile(dec_path):
            self.encoder_model.load_state_dict(torch.load(
                enc_path, map_location=lambda storage, loc: storage))
            self.decoder_model.load_state_dict(torch.load(
                dec_path, map_location=lambda storage, loc: storage))
            self.encoder_model.to(device)
            self.decoder_model.to(device)
        else:
            print(10 * '!', '无法加载模型', 10 * '!')

    def saveModel(self, best: bool = False):
        """保存模型"""
        out_dir = self.config["network"]['out_dir']
        enc_path = out_dir + self.experiment_id + '/enc'
        dec_path = out_dir + self.experiment_id + '/dec'
        enc_path += '_best.pth' if best else '.pth'
        dec_path += '_best.pth' if best else '.pth'
        torch.save(self.encoder_model.state_dict(), enc_path)
        torch.save(self.decoder_model.state_dict(), dec_path)

    def saveYaml(self, out_dir="network_files/"):
        """保存配置文件"""
        config_path = out_dir + self.experiment_id + '/' + self.experiment_id + '.yaml'
        if not os.path.exists(out_dir + self.experiment_id):
            os.makedirs(out_dir + self.experiment_id, exist_ok=True)
        with open(config_path, 'w') as f:
            saver = yaml.YAML()
            saver.dump(self.config, f)

    def test(self, best=True):
        """测试模型"""
        return self.evaluate(self.submaps.getTestSet(),
                             load_model=True,
                             best_model=best,
                             reference_points='map',
                             compute_memory=True, evaluate=True)

    def getNetworkParams(self):
        """获取网络参数"""
        return list(self.encoder_model.parameters()) + list(self.decoder_model.parameters())

    def getScheduler(self, optimizer, len_data_loader, batch_size):
        """获取学习率调度器"""
        number_epochs = self.config['train']['max_epochs']
        steps_per_epoch = int(len_data_loader / batch_size)
        max_lr = self.config["train"]["optimizer"]["max_lr"]
        div_factor = max_lr / self.config["train"]["optimizer"]["start_lr"]
        final_div_factor = self.config["train"]["optimizer"]["start_lr"] / self.config["train"]["optimizer"]["end_lr"]
        pct_start = self.config["train"]['optimizer']["pct_incr_cycle"]
        anneal_strategy = self.config["train"]['optimizer']["anneal_strategy"]
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch,
                                             epochs=number_epochs, pct_start=pct_start, anneal_strategy=anneal_strategy,
                                             div_factor=div_factor, final_div_factor=final_div_factor)

    def getLogWriter(self, logdir):
        """获取日志记录器"""
        if os.path.isdir(logdir):
            filelist = [f for f in os.listdir(logdir)]
            for f in filelist:
                os.remove(os.path.join(logdir, f))
        return SummaryWriter(logdir)

    def train(self, verbose=True):
        """训练模型"""
        # 设置
        if self.config['train']['load_pretrained']:
            self.loadModel(best=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder_model.to(self.device)
        self.decoder_model.to(self.device)
        number_epochs = self.config['train']['max_epochs']
        output_path = self.config['network']['out_dir']
        writer = self.getLogWriter(output_path + 'log/' + self.experiment_id + "/")
        batch_size = min((self.config["train"]["batch_size"], self.submaps.getTrainSize()))

        # 初始化优化器
        lr = self.config["train"]["optimizer"]["max_lr"]
        optimizer = optim.Adam(self.getNetworkParams(), lr=lr, amsgrad=False)
        scheduler = self.getScheduler(optimizer, self.submaps.getTrainSize(), batch_size)
        self.saveYaml(out_dir=output_path)

        time_start = time.time()
        optimizer.zero_grad()
        r_batch = 0
        best_loss = 1e10
        n_pct_time = time.time()

        validation_it = 0
        nr_batches = int(self.submaps.getTrainSize() / batch_size)
        print('开始训练: #子图数量: %d, #批次: %f' % (self.submaps.getTrainSize(), nr_batches))
        for epoch in range(number_epochs):
            running_loss = 0
            batch = 0
            for i, input_dict in enumerate(self.submaps.getTrainSet()):
                if i >= (nr_batches * batch_size):  # 舍弃最后一个批次
                    continue
                # 预处理
                input_dict['points'] = input_dict['points'].to(self.device)
                input_dict['features'] = input_dict['features'].to(self.device)
                input_points = input_dict['points']

                # 编码和解码
                t1 = time.time()
                out_dict = self.encoder_model(input_dict.copy())
                out_dict = self.decoder_model(out_dict)
                translation = out_dict['features'][:, :3]

                samples = out_dict['points']
                samples_transf = samples + translation

                # 计算损失
                loss = self.getTrainLoss(input_points, samples, translation)
                loss += loss_handler.linDeconvRegularizer(
                    self.decoder_model,
                    weight=self.config['train']['loss_weights']['upsampling_reg'],
                    gt_points=input_points)

                loss.backward()
                running_loss += loss.item()

                # 梯度累积
                if (i % batch_size) == (batch_size - 1):
                    r_batch += 1
                    batch += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss /= batch_size
                    scheduler.step()

                    curr_lr = scheduler.get_lr()[0]

                    # 写入日志
                    writer.add_scalar('学习损失', running_loss, r_batch)
                    writer.add_scalar('学习率', curr_lr, r_batch)
                    if verbose:
                        print('[%d, %5d] 损失: %.5f, 时间: %.1f lr: %.5f' %
                              (epoch + 1, batch, running_loss, time.time() - time_start, curr_lr))

                    time_start = time.time()
                    self.saveModel(best=False)

                    running_loss = 0

            # 验证
            if (epoch % self.config['train']['validation']['report_rate']) == (
                    self.config['train']['validation']['report_rate'] - 1):
                ts_val = time.time()
                valid_dict = self.evaluate(dataloader=self.submaps.getValidSet())
                chamf_dist = valid_dict['reconstruction_error']
                writer.add_scalar('验证: Chamfer距离', chamf_dist, r_batch)
                if chamf_dist < best_loss:
                    self.saveModel(best=True)
                    best_loss = chamf_dist
                if verbose:
                    print('[%d, 验证] 重建误差: %.5f, 时间: %.1f' %
                          (epoch + 1, chamf_dist, time.time() - ts_val))

                validation_it += 1

            # 每10%打印一次进度
            if pcu.isEveryNPercent((epoch), max_it=number_epochs, percent=10):
                n_pct = (epoch + 1) / number_epochs * 100
                time_est = (time.time() - n_pct_time) / n_pct * (100 - n_pct)
                print("%4d%% in %ds, 估计剩余时间: %ds (%d分钟), 最佳损失: %.5f" % (
                    n_pct, "%", time.time() - n_pct_time, time_est, time_est / 60, best_loss))

    def getTrainLoss(self, gt_points: torch.tensor, samples, translations):
        """计算训练损失"""
        loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)  # 初始化损失
        samples_transf = samples + translations

        # Chamfer损失
        d_map2transf, d_transf2map, idx3, idx4 = self.cham_loss(gt_points.unsqueeze(0), samples_transf.unsqueeze(0))
        loss += (self.w_map2transf * d_map2transf.mean() + self.w_transf2map * d_transf2map.mean())
        return loss

    def evaluate(self, dataloader,
                 load_model=False,
                 best_model=False,
                 reference_points='points',
                 compute_memory=False,
                 evaluate=False):
        """评估模型"""
        loss_evaluator = evaluator.Evaluator(self.config)
        self.encoder_model.eval()
        self.decoder_model.eval()
        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.encoder_model.to(self.device)
                self.decoder_model.to(self.device)
                print('加载最佳模型:', best_model)
            for i, input_dict in enumerate(tqdm(dataloader)):
                map_idx = input_dict['idx']
                scale = input_dict['scale']
                input_dict['features'] = input_dict['features'].to(self.device)
                input_dict['points'] = input_dict['points'].to(self.device)

                # 转换为float16
                out_dict = self.encoder_model(input_dict.copy())
                if self.config['evaluation']['float16']:
                    out_dict['points'] = out_dict['points'].half().float()
                    out_dict['features'] = out_dict['features'].half().float()

                # 计算内存
                if compute_memory:
                    nbytes = 2 if self.config['evaluation']['float16'] else 4
                    mem = (out_dict['points'].numel() + out_dict['features'].numel()) * nbytes
                    loss_evaluator.eval_results['memory'].append(mem)
                    loss_evaluator.eval_results['bpp'].append(mem / input_dict['map'].shape[0] * 8)

                # 解码
                out_dict = self.decoder_model(out_dict)
                translation = out_dict['features'][:, :3]
                samples = out_dict['points']
                samples_transf = samples + translation

                # 缩放到度量空间
                samples_transf *= scale
                samples *= scale
                translation *= scale
                gt_points = input_dict[reference_points].to(self.device)

                # 重建误差
                reconstruction_error = loss_evaluator.chamferDist(
                    gt_points=gt_points, source_points=samples_transf)
                loss_evaluator.eval_results['mapwise_reconstruction_error'].append(
                    reconstruction_error.item())

                if evaluate:  # 测试时的完整评估
                    feat_ind = np.cumsum([0] + self.config['grid']['feature_dim'])
                    normal_idx = pcu.findList(self.config['grid']['features'], value='normals')
                    normal_idx = (feat_ind[normal_idx], feat_ind[normal_idx + 1])
                    gt_normals = input_dict[reference_points + '_attributes'][:, normal_idx[0]:normal_idx[1]].cuda()
                    loss_evaluator.evaluate(
                        gt_points=gt_points, source_points=samples_transf, gt_normals=gt_normals)
            chamfer_dist = loss_evaluator.getRunningLoss()
            loss_evaluator.eval_results['reconstruction_error'] = chamfer_dist

        self.encoder_model.train()
        self.decoder_model.train()

        return loss_evaluator.eval_results

    def encodeDecode(self, input_dict, float_16=True):
        """编码和解码"""
        map_idx = input_dict['idx']
        scale = input_dict['scale']
        input_dict['features'] = input_dict['features'].to(self.device)
        input_dict['points'] = input_dict['points'].to(self.device)

        # 转换为float16
        out_dict = self.encoder_model(input_dict.copy())
        if float_16:
            out_dict['points'] = out_dict['points'].half().float()
            out_dict['features'] = out_dict['features'].half().float()
        nr_emb = out_dict['points'].shape

        # 解码
        out_dict = self.decoder_model(out_dict)
        translation = out_dict['features'][:, :3]
        samples = out_dict['points']
        samples_transf = samples + translation

        samples_transf *= scale
        return samples_transf, nr_emb


if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./sample_net_trainer.py")
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='配置文件路径',
    )
    FLAGS, unparsed = parser.parse_known_args()

    print('传递的标志')
    config = yaml.safe_load(open(FLAGS.config, 'r'))
    print('加载的配置文件')
    trainer = DepocoNetTrainer(config)
    print('初始化训练器')
    trainer.train(verbose=True)
