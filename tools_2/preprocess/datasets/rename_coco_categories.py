#!/usr/bin/env python
# encoding:utf-8
"""
author: sunchongjing
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: rename_coco_categories.py
@time: 2019/9/6 16:37
@desc:
"""
import json


class RenameCategory(object):

    def __init__(self, new2olds, coco_json_file):
        """

        :param new2olds: one new category may contains more than one old categories
        """
        self.new2olds = new2olds
        with open(coco_json_file, 'r') as f:
            self.coco_dataset = json.load(f)
        self.old2new = dict()

        self.new_cates = []
        for new_cate in new2olds.keys():
            self.new_cates.append(new_cate)
            old_cates = new2olds[new_cate]
            for old_cate in old_cates:
                self.old2new[old_cate] = new_cate
        categories = self.coco_dataset['categories']

        self.old_cates = [None for _ in range(len(categories))]
        for category in categories:
            category['id']
            category['name']
            self.old_cates[category['id']-1] = category['name']

    def old_cate_id_2_new_id(self, old_cate_id):
        """

        :param old_cate_id:
        :return:
        """
        old_cate = self.old_cates[old_cate_id - 1]
        new_cate = self.old2new[old_cate]
        new_cate_id = self.new_cates.index(new_cate) + 1
        return new_cate_id

    def convert(self, save_file):
        new_annotations = []
        for annotation in self.coco_dataset['annotations']:
            new_anno = annotation
            new_anno['category_id'] = self.old_cate_id_2_new_id(annotation['category_id'])
            new_annotations.append(new_anno)

        new_categories = []
        for i, category in enumerate(self.new_cates):
            new_category = dict()
            new_category['supercategory'] = "FWW"
            new_category['id'] = i + 1
            new_category['name'] = category
            new_categories.append(new_category)

        new_coco_json = dict()
        new_coco_json['images'] = self.coco_dataset['images']
        new_coco_json['type'] = self.coco_dataset['type']
        new_coco_json['annotations'] = new_annotations
        new_coco_json['categories'] = new_categories

        with open(save_file, 'w') as f:
            json.dump(new_coco_json, f, indent=4)


if __name__ == '__main__':
    merge_dict = {"A2DMR": ['A2DMR_G', 'A2DMR_P', 'A2DMR_S', 'A2DMR'],
                  "A2SIP": ['A2SIP_G', 'A2SIP_P', 'A2SIP_S', 'A2SIP'],
                  "A2WBD": ["A2WBD_G", 'A2WBD'],
                  "Others": ['A2WOP_G', 'A2PPL_P', 'A2SFB_N', 'A2CFB_S', 'A2CFB_G',
                             'A2SFB_G', 'A2DBE_G', 'A2SSP_N', 'A2PPL_S', 'A2PMR_P',
                             'A2PMR_S', 'A2PMR_G', 'A2SSP_G', 'A2SFB_P', 'A2WOR_G',
                             'A2PPL_G', 'A2WSR_G', 'A2WSC_S', 'A2CIP_P', 'A2CIP_G',
                             'A2SFB_S', 'A2PPL_N', 'A2WTR_G', 'A2CFB_P', 'A2DOE_G']
                  }

    json_path = '/home/Visionox/V3/poc_data/V3_ultimate_test/test.json'
    renameCate = RenameCategory(merge_dict, json_path)
    renameCate.convert('/home/Visionox/V3/poc_data/V3_ultimate_test/test_merge.json')


