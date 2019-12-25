#!/usr/bin/env python
# encoding: utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: auto_gen_config_file.py
@time: 2019/9/6 16:09
@desc: 
"""
# -*- coding: utf-8 -*-
import time, os


def generate_configs(base_config_file, pre_train_model, data_root_, train_name,
                     test_name, num_classes, work_dir, log_interval, checkpoint_interval, total_epoch,
                     lr_config_step):
    """


    :param base_config_file: the config file
    :param pre_train_model: pre_train_model
    :param data_root_: the root path of data
    :param train_name: the train json name
    :param test_name: the test json name
    :param num_classes: the number of class/category
    :param work_dir: the directory for saving training results
    :param log_interval: the interval for logging
    :param checkpoint_interval: the interval for saving checkpoint
    :param total_epoch: the total number of training epoch
    :param lr_config_step: the step list for adjusting learning rate
    :return:
    """
    f = open(base_config_file, "r")
    file_name = str(time.strftime("%Y%m%d_%H%M%S", time.localtime())) + '.py'
    file_name = os.path.join(work_dir, file_name)
    w = open(file_name, 'w')
    stat = None
    for line in f.readlines():
        # print(line)
        prefix_line = line.split('=')[0]
        if 'pretrained' in prefix_line:
            line = line.split('=')[0] + "='" + pre_train_model + "',\n"
        elif 'num_classes' in prefix_line:
            line = line.split('=')[0] + "=" + str(num_classes+1) + ",\n"
        elif 'data_root' in prefix_line:
            line = line.split('=')[0] + "= '" + str(data_root_) + "'\n"
        elif ('interval' in line) and ('dict(' not in line):
            line = line.split('=')[0] + '=' + str(log_interval) + ',\n'
        elif 'checkpoint_config' in prefix_line:
            line = line.split('=')[0] + '= dict(interval=' + str(checkpoint_interval) + ')\n'
        elif 'work_dir' in prefix_line:
            line = line.split('=')[0] + "= '" + work_dir + "'\n"
        elif 'total_epochs' in prefix_line:
            line = line.split('=')[0] + '= ' + str(total_epoch) + '\n'
        elif 'step=' in line:
            line = line.split('=')[0] + '=' + str(lr_config_step) + ')\n' # 可能出问题

        if 'train=dict' in line:
            stat = 'train'
        elif 'val=dict' in line:
            stat = 'val'
        elif 'test=dict' in line:
            stat = 'test'

        if 'img_prefix' in line:
            line = line.split('=')[0] + '=data_root,\n'

        if (stat == 'train') and ('ann_file' in line):
            line = line.split('=')[0] + "=data_root + '" + train_name + "',\n"
        elif (stat == 'val') and ('ann_file' in line):
            line = line.split('=')[0] + "=data_root + '" + test_name + "',\n"
        elif (stat == 'test') and ('ann_file' in line):
            line = line.split('=')[0] + "=data_root + '" + test_name + "',\n"

        # print(line)
        w.write(line)

    f.close()
    w.close()
    return file_name


# if __name__ == '__main__':
#     base_config_file = '/home/scj/mm_detection_proj/configs/base_configs/visonox_v3.py'
#     pretrain_model = '/home/scj/mm_detection_proj/pre_train_models/resnext101_32x4d-a5af3160.pth'
#     data_root = '/home/scj/mm_detection_proj/stations/boe_b2/trainData'
#     train_name = 'train.json'
#     test_name = 'test.json'
#     num_classes = '31'
#     work_dir = '/home/scj/mm_detection_proj/train_models/visonox_v3'
#     log_interval = 5
#     total_epoch = 12
#     lr_config_step = [4, 8]
#     generate_configs(base_config_file, pretrain_model, data_root, train_name,
#                      test_name, num_classes, work_dir, log_interval, total_epoch, lr_config_step)


