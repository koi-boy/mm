from __future__ import division
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# def parse_args():
#     parser = argparse.ArgumentParser(description='Train a detector')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument('--work_dir', help='the dir to save logs and models')
#     parser.add_argument(
#         '--resume_from', help='the checkpoint file to resume from')
#     parser.add_argument(
#         '--validate',
#         action='store_true',
#         help='whether to evaluate the checkpoint during training')
#     parser.add_argument(
#         '--gpus',
#         type=int,
#         default=1,
#         help='number of gpus to use '
#         '(only applicable to non-distributed training)')
#     parser.add_argument('--seed', type=int, default=None, help='random seed')
#     parser.add_argument(
#         '--launcher',
#         choices=['none', 'pytorch', 'slurm', 'mpi'],
#         default='none',
#         help='job launcher')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument(
#         '--autoscale-lr',
#         action='store_true',
#         help='automatically scale lr with the number of gpus')
#     args = parser.parse_args()
#     if 'LOCAL_RANK' not in os.environ:
#         os.environ['LOCAL_RANK'] = str(args.local_rank)
#
#     return args

def main(config,
         work_dir=None,
         resume_from=None,
         validate=False,
         visible_gpus='0,1,2,3',
         gpus=1,
         seed=None,
         launcher='none',
         local_rank=0,
         autoscale_lr=False):

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)

    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus

    cfg = Config.fromfile(config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if work_dir is not None:
        cfg.work_dir = work_dir
    if resume_from is not None:
        cfg.resume_from = resume_from
    cfg.gpus = gpus

    if autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if seed is not None:
        logger.info('Set random seed to {}'.format(seed))
        set_random_seed(seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=validate,
        logger=logger)


if __name__ == '__main__':
    config = r'/data/sdv1/whtm/mmdet_cq/CQ_cfg/HU_cfg/ga_faster_x101_32x4d_fpn_1x_all.py'
    main(config, validate=True)
