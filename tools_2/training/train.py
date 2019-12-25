#!/usr/bin/env python
# encoding:utf-8
"""
author: sunchongjing
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: train.py
@time: 2019/9/6 16:37
@desc:
"""
from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed, train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector




class TrainMMDet(object):

    def __init__(self,
                 config_file,
                 work_dir,
                 visible_gpus='0,1,2,3,4',
                 gpus_num=5,
                 seed=None,
                 validate=True,
                 resume_from=None,
                 autoscale_lr=True,
                 launcher='pytorch',
                 local_rank=0):
        """

        :param config_file: train config file path
        :param work_dir: the dir to save logs and models
        :param visible_gpus: the gpu indexes can be used by this process, can be set as '0,1', '3,5'
        :param gpus_num: number of gpus to use,.only applicable to non-distributed training
        :param seed: random seed, int
        :param validate: whether to evaluate the checkpoint during training
        :param resume_from: the checkpoint file to resume from
        :param autoscale_lr: automatically scale lr with the number of gpus
        :param launcher: job launcher, ['none', 'pytorch', 'slurm', 'mpi']
        """
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(local_rank)

        os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus

        cfg = Config.fromfile(config_file)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        # update configs according to CLI args
        if work_dir is not None:
            cfg.work_dir = work_dir
        if resume_from is not None:
            cfg.resume_from = resume_from
        cfg.gpus = gpus_num

        if autoscale_lr:
            # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

        self.cfg = cfg

        # init distributed env first, since logger depends on the dist info.
        assert launcher in ['none', 'pytorch', 'slurm', 'mpi']
        if launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(launcher, **cfg.dist_params)
        self.distributed = distributed

        # init logger before other steps
        self.logger = get_root_logger(cfg.log_level)
        self.logger.info('Distributed training: {}'.format(distributed))

        # set random seeds
        if seed is not None:
            self.logger.info('Set random seed to {}'.format(seed))
            set_random_seed(seed)

        self.model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        self.train_dataset = build_dataset(cfg.data.train)
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.text,
                CLASSES=self.train_dataset.CLASSES)
        # add an attribute for visualization convenience
        self.model.CLASSES = self.train_dataset.CLASSES
        self.validate = validate

    def start_training(self):
        train_detector(
            self.model,
            self.train_dataset,
            self.cfg,
            distributed=self.distributed,
            validate=self.validate,
            logger=self.logger)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training(only applicable to distributed training)')
    parser.add_argument(
        '--visible_gpus',
        # opzealot
        default='0,1,2,3,4',
        help="str such as '0,1', '2,0,3'")
    parser.add_argument(
        '--gpus',
        type=int,
        default=5,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        # opzealot
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale_lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    config_file = args.config
    work_dir = args.work_dir
    resume_from = args.resume_from
    validate = args.validate
    visible_gpus = args.visible_gpus
    gpus_num = args.gpus
    seed = args.seed
    launcher = args.launcher
    local_rank = args.local_rank
    autoscale_lr = args.autoscale_lr

    train = TrainMMDet(config_file,
                       work_dir,
                       visible_gpus,
                       gpus_num,
                       seed,
                       validate,
                       resume_from,
                       autoscale_lr,
                       launcher,
                       local_rank)

    train.start_training()


if __name__ == '__main__':
    main()

    # config_file_ = '/home/Visionox/Visionox_code/13620_code/configs/model_base_configs/config_v2.py'
    # work_dir_ = '/home/Visionox/V2/work_dir/20191022_1'
    # visible_gpus_ = '0,1,2,3,4'
    # gpus_num_ = 5
    # seed_ = 34
    #
    # train = TrainMMDet(config_file=config_file_, work_dir=work_dir_, visible_gpus=visible_gpus_,
    #                    gpus_num=gpus_num_, seed = seed_, validate=True)
    # train.start_training()

    # config_file_ = '/home/scj/mm_detection_proj/stations/visionox_v3/work_dir/20190911_154938.py'
    # work_dir_ = '/home/scj/mm_detection_proj/stations/visionox_v3/training_models/20190911'
    # gpus_num_ = 4
    # seed_ = 34
    #
    # train = TrainMMDet(config_file_, work_dir_, gpus_num_, seed_)
    # train.start_training()
