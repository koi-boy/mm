#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: liusili@unionbigdata.com
@software:
@file: test_flow
@time: 10/18/19
@desc: test the model
"""
from preprocess.datasets.voc2coco import Voc2Coco
from data_explore.category_distribution import CategoryDist
from preprocess.datasets.rename_coco_categories import RenameCategory
from training.train import TrainMMDet
from configs.auto_gen_config_file import generate_configs
from testing.test import TestMMDetModel
from metrics.test_result_analysis import ResultAnalysis
import os
from configs.global_info import main_info


class TestFlow(object):
    def __init__(self, sample_root, k):
        self.sample_root = sample_root
        self.k = k

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
    
    @staticmethod
    def get_latest_config(work_dir):
        for file in os.listdir(work_dir):
            if file.endswith('.py'):
                return os.path.join(work_dir, file)
    
    def test_flow(self,
                  drop_or_others=True,
                  work_dir=None,
                  merge=False,
                  merge_dict=None):

        if drop_or_others:
            test_name = 'test_merge.json' if merge else 'test.json'
        else:
            test_name = 'test_merge_others.json' if merge else 'test_others.json'

        # config_file_ = self.get_latest_config(work_dir)
        config_file_ = '/home/Visionox/Visionox_code/V3_POC_code/configs/model_base_configs/config_V3_deployment.py'

        print('start to test...')
        last_k_epoch = self.select_last_k_epoches(work_dir, self.k)
        out_dir = os.path.join(work_dir, 'out_dir')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 测试改这里就行
        last_k_epoch = ['v3_ultimate3.pth']
        for epoch_ck in last_k_epoch:
            out_pkl = os.path.join(out_dir, epoch_ck + '.pkl')
            epoch_ck_path = os.path.join(work_dir, epoch_ck)
            test_model = TestMMDetModel(config_file_, epoch_ck_path, out_pkl)
            test_model.start_testing()
    
            out_confusion_matrix_csv = os.path.join(out_dir, epoch_ck + '.csv')
            analysis = ResultAnalysis(self.sample_root, os.path.join(self.sample_root, test_name),
                                      out_pkl,
                                      merge,
                                      merge_dict)
            analysis.save_confusion_matrix(out_confusion_matrix_csv, rule_method='MaxConf', alpha=0.7)
    
            out_wrong_classified_img_path = os.path.join(out_dir, epoch_ck.split('.pth')[0])
            if not os.path.exists(out_wrong_classified_img_path):
                os.makedirs(out_wrong_classified_img_path)
            analysis.draw_bounding_boxes(out_wrong_classified_img_path, 0.3, rule_method='MaxConf', alpha=0.7)

if __name__ == '__main__':
    work_dir = '/home/Visionox/V3/work_dir/v3_ultimate3/'
    sample_root_ = '/home/Visionox/V3/poc_data/V3_ultimate_test/'
    drop_or_others = True
    test_last_k_epoch = 1
    merge = True
    merge_dict = {"A2DMR": ['A2DMR_G', 'A2DMR_P', 'A2DMR_S', 'A2DMR'],
                  "A2SIP": ['A2SIP_G', 'A2SIP_P', 'A2SIP_S', 'A2SIP'],
                  "A2WBD": ["A2WBD_G", 'A2WBD'],
                  "Others": ['A2WOP_G', 'A2PPL_P', 'A2SFB_N', 'A2CFB_S', 'A2CFB_G',
                             'A2SFB_G', 'A2DBE_G', 'A2SSP_N', 'A2PPL_S', 'A2PMR_P',
                             'A2PMR_S', 'A2PMR_G', 'A2SSP_G', 'A2SFB_P', 'A2WOR_G',
                             'A2PPL_G', 'A2WSR_G', 'A2WSC_S', 'A2CIP_P', 'A2CIP_G',
                             'A2SFB_S', 'A2PPL_N', 'A2WTR_G', 'A2CFB_P', 'A2DOE_G']
                  }
    process_test = TestFlow(sample_root=sample_root_,
                            k=test_last_k_epoch
                            )
    process_test.test_flow(drop_or_others=drop_or_others,
                           work_dir=work_dir,
                           merge=merge,
                           merge_dict=merge_dict)
    
        
        


