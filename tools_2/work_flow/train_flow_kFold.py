#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: liusili@unionbigdata.com
@software:
@file: train_flow
@time: 10/30/19
@desc:
"""
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from preprocess.datasets.voc2coco import Voc2Coco
from data_explore.category_distribution import CategoryDist
from preprocess.datasets.rename_coco_categories import RenameCategory
from training.train import TrainMMDet
from configs.auto_gen_config_file import generate_configs
from testing.test import TestMMDetModel
from metrics.test_result_analysis import ResultAnalysis
from configs.global_info import main_info

import torch


class MainFlow(object):

    def __init__(self, sample_root, xml_img_in_same_folder):
        self.sample_root = sample_root
        self.xml_img_in_same_folder = xml_img_in_same_folder
        self.model_classes_num = 0

    def __generate_coco_json(self, cate2img_count, need_img, drop_or_others, test_size, random_state):
        """

        :param cate2img_count:
        :param need_img:
        :param drop_or_others:
        :param test_size:
        :param random_state:
        :return:
        """

        if drop_or_others:
            model_categories = [x for x in cate2img_count.keys() if cate2img_count[x] >= need_img]
            data_convert = Voc2Coco(self.sample_root, self.xml_img_in_same_folder)
            data_convert.get_train_test_json(test_size=test_size, random_state=random_state,
                                             categories=model_categories)
            self.model_classes_num = len(model_categories)
        else:
            data_convert = Voc2Coco(self.sample_root, self.xml_img_in_same_folder)
            data_convert.get_train_test_json(test_size=test_size, random_state=random_state)
            new2olds = dict()
            for cate in cate2img_count.keys():
                if cate2img_count[cate] < need_img:
                    if main_info.others_code not in new2olds.keys():
                        new2olds[main_info.others_code] = [cate]
                    else:
                        new2olds[main_info.others_code].append(cate)
                else:
                    new2olds[cate] = cate
            self.model_classes_num = len(new2olds.keys())
            
            redefine_code = RenameCategory(new2olds, os.path.join(self.sample_root, 'train.json'))
            redefine_code.convert(os.path.join(self.sample_root, 'train_others.json'))
            
            redefine_code = RenameCategory(new2olds, os.path.join(self.sample_root, 'test.json'))
            redefine_code.convert(os.path.join(self.sample_root, 'test_others.json'))

    # opzealot start
    def __generate_merge_json(self, drop_or_others=True, merge=False, merge_dict={}, sample_root='/'):
        if merge and len(merge_dict) > 0:
            if drop_or_others:
                renameCate = RenameCategory(merge_dict, sample_root+'train.json')
                renameCate.convert(sample_root+'train_merge.json')
                renameCate = RenameCategory(merge_dict, sample_root+'test.json')
                renameCate.convert(sample_root+'test_merge.json')
            else:
                renameCate = RenameCategory(merge_dict, sample_root+'train_others.json')
                renameCate.convert(sample_root+'train_merge_others.json')
                renameCate = RenameCategory(merge_dict, sample_root+'test_others.json')
                renameCate.convert(sample_root+'test_merge_others.json')
    # opzealot end

    @staticmethod
    def select_last_k_epoches(work_dir, k):
        epoch_id = []
        for file_ in os.listdir(work_dir):
            if file_.startswith('epoch') and file_.endswith('.pth'):
                temp = file_.split('_')[1]
                id_ = int(temp.split('.')[0])
                epoch_id.append(id_)
        sorted_epoch_ids = sorted(epoch_id)
        start_ = len(sorted_epoch_ids) - k
        if start_ < 0:
            start_ = 0
        select_epoches = sorted_epoch_ids[start_:]
        
        return ['epoch_' + str(x) + '.pth' for x in select_epoches]

    def __generate_k_fold_json(self, cate2img_count, need_img, drop_or_others, repeats, k_fold, random_state):
        if drop_or_others:
            model_categories = [x for x in cate2img_count.keys() if cate2img_count[x] >= need_img]
            data_convert = Voc2Coco(self.sample_root, self.xml_img_in_same_folder)
            data_convert.repeated_k_fold(repeats, k_fold, random_state)
            self.model_classes_num = len(model_categories)

        else:
            assert False, 'K folds mode has not supported drop images yet.'

    def generate_k_fold_json(self,
                             repeats=1,
                             k_fold=5,
                             random_state=34,
                             need_img=400,
                             drop_or_others=True,
                             work_dir=None,
                             ):

        if not os.path.exists(work_dir):
            os.mkdir(work_dir)
        # voc to coco
        print('voc to coco...')
        cate_dist = CategoryDist(self.sample_root, self.xml_img_in_same_folder)
        cate2img_count = cate_dist.get_dist_on_img_xml()

        from data_explore import plotting

        # opzealot
        plotting.bar_plot(cate2img_count, os.path.join(work_dir, 'dist.png'), control_line=need_img)
        self.__generate_k_fold_json(cate2img_count, need_img, drop_or_others, repeats, k_fold, random_state)

    def train_test_flow(self,
                        fold_index = 1,
                        random_state=34,
                        drop_or_others=True,
                        visible_gpus='0',
                        gpu_num=1,
                        distributed_train=True,
                        work_dir=None,
                        base_config=None,
                        pretrain_model=None,
                        log_interval=5,
                        checkpoint_interval=1,
                        total_epoch=24,
                        lr_config_step=[8, 11]
                        ):

        assert work_dir is not None, 'The work directory should not be None.'
        assert base_config is not None, 'The base config should not be None.'
        assert pretrain_model is not None, "The pretrain model should not be None."

        work_dir = os.path.join(work_dir, 'fold_{}'.format(fold_index))
        if not os.path.exists(work_dir):
            os.mkdir(work_dir)

        # training
        print('start to train...')

        if drop_or_others:
            train_name = 'train_{}.json'.format(fold_index)
            test_name = 'test_{}.json'.format(fold_index)
        else:
            assert False, 'K folds mode has not supported drop images yet.'

        num_classes = self.model_classes_num

        # train_name = 'train.json' if drop_or_others else 'train_others.json'
        # test_name = 'test.json' if drop_or_others else 'test_others.json'

        print(num_classes, total_epoch)
        num_classes = 14
        config_file_ = generate_configs(base_config, pretrain_model, self.sample_root, train_name, test_name,
                                        num_classes, work_dir, log_interval, checkpoint_interval,
                                        total_epoch, lr_config_step)
        
        if not distributed_train:
            # non distributed training
            train = TrainMMDet(config_file_, work_dir, visible_gpus=visible_gpus, gpus_num=gpu_num,
                               seed=random_state, launcher='none')
            print('start to training..')
            train.start_training()
        else:
            # distributed training
            from training import dist_train
            master_port = '25200'
            training_args = [config_file_, '--validate']
            dist_train.one_node_multi_gpu(master_port, gpu_num, training_args, visible_gpus)
        
        # testing
        print('start to test...')
        # config_file_ = '/home/Visionox/V2/work_dir/20191018_1/20191018_054738.py'
        
        last_k_epoch = self.select_last_k_epoches(work_dir, 1)
        out_dir = os.path.join(work_dir, 'out_dir')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for epoch_ck in last_k_epoch:
            out_pkl = os.path.join(out_dir, epoch_ck + '.pkl')
            epoch_ck_path = os.path.join(work_dir, epoch_ck)
            test_model = TestMMDetModel(config_file_, epoch_ck_path, out_pkl)
            test_model.start_testing()
            
            out_confusion_matrix_csv = os.path.join(out_dir, epoch_ck + '.csv')
            analysis = ResultAnalysis(self.sample_root, os.path.join(self.sample_root, test_name), out_pkl,
                                      merge, merge_dict)
            analysis.save_confusion_matrix(out_confusion_matrix_csv, rule_method='MaxConf', alpha=0.7)
            
            out_wrong_classified_img_path = os.path.join(out_dir, epoch_ck.split('.pth')[0])
            if not os.path.exists(out_wrong_classified_img_path):
                os.makedirs(out_wrong_classified_img_path)
            analysis.draw_bounding_boxes(out_wrong_classified_img_path, 0.3, rule_method='MaxConf', alpha=0.7)


if __name__ == '__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')

    xml_img_in_same_folder = True
    sample_root_ = '/root/WHTM/data/TEST/1x1B2/'
    base_conf = '/root/WHTM/data/TEST/config_normal.py'
    pretrain_model = '/root/WHTM/pretrained model/resnet50_trained.pth'

    merge = False
    merge_dict = {}

    work_dir ='/root/WHTM/work_dirs/TEST'
    process_flow = MainFlow(sample_root_, xml_img_in_same_folder)
    process_flow.generate_k_fold_json(repeats=1,
                                      k_fold=5,
                                      random_state=16,
                                      need_img=20,
                                      drop_or_others=True,
                                      work_dir=work_dir)

    for i in range(1, 6):
        process_flow.train_test_flow(fold_index=i,
                                     random_state=16,
                                     drop_or_others=True,
                                     work_dir=work_dir,
                                     base_config=base_conf,
                                     pretrain_model=pretrain_model,
                                     visible_gpus='0,1',
                                     distributed_train=False,
                                     gpu_num=2,
                                     log_interval=10,
                                     checkpoint_interval=10,
                                     total_epoch=20,
                                     lr_config_step=[15, 18]
                                     )

