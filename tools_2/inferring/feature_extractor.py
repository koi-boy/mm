#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: liusili@unionbigdata.com
@software:
@file: feature_extractor
@time: 2019/12/6
@desc:
"""
import warnings

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from pipeline_lib import ImageCrop
from pipeline_lib import minIOFRandomCrop
from mmdet.datasets.registry import PIPELINES
PIPELINES.register_module(ImageCrop)
PIPELINES.register_module(minIOFRandomCrop)


def init_detector(config, checkpoint=None, device='cuda:0'):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def preprocess(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data



if __name__ == '__main__':
    config_file = '/home/Visionox/Visionox_code/V3_OLED_code/configs/model_base_configs/config_OLED.py'
    # config_file = '/home/Visionox/V3/OLED_deploy/config_OLED.py'
    checkpoint = '/home/Visionox/V3/OLED_deploy/v3_oled_deploy.pth'
    img_path = '/home/Visionox/V3/OLED_deploy/OLED_test/Others/L2E9A23A7091BT_617986_187956_BIG_A_BF_BL_R_REV.jpg'
    img_path = np.array(img_path)
    model = init_detector(config_file, checkpoint, device='cuda:0')
    data = preprocess(model, img_path)
    print(len(data['img']))
    print(data['img'][0].shape)
    print(type(model))
    for name, module in model._modules.items():
        if name == 'backbone':
            data_feature = module(data['img'][0])
            print(data_feature[0].shape)
            # torch.Size([1, 256, 256, 312])