#!/usr/bin/env python
# encoding:utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: category_distribution.py
@time: 2019/9/6 16:20
@desc: Analyze the distribution of each category.
This can be used after voc2coco, especially when we wanna analyze the coco json file
"""

import os
import json
from tools_2.utils import file_name_ext


class CategoryDist(object):

    def __init__(self, sample_root, xml_img_in_same_folder=True):
        self.sample_root = sample_root
        self.xml_img_in_same_folder = xml_img_in_same_folder

    def get_dist_on_folder(self):
        """
        count the number of img in each category folder
        :return:
        """
        cate2count = dict()
        for dir_ in os.listdir(self.sample_root):
            path_ = os.path.join(self.sample_root, dir_)
            if os.path.isdir(path_):
                count = 0
                for root, dirs, files in os.walk(path_):
                    for file in files:
                        if not file_name_ext.is_supported_img_ext(file):
                            continue
                        else:
                            count += 1
                cate2count[dir_] = count
        return cate2count

    def get_dist_on_img_xml(self):
        """
        get the count of images having xml for each category
        :return:
        """
        cate2count = dict()
        for dir_ in os.listdir(self.sample_root):
            path_ = os.path.join(self.sample_root, dir_)
            if os.path.isdir(path_):
                count = 0
                for root, dirs, files in os.walk(path_):
                    for file in files:
                        if not file_name_ext.is_supported_img_ext(file):
                            continue
                        else:
                            jpg_path = os.path.join(root, file)
                            xml_path = os.path.splitext(jpg_path)[0] + '.xml'
                            if not self.xml_img_in_same_folder:
                                xml_path = xml_path.replace('images', 'labels')

                            if os.path.exists(xml_path):
                                count += 1
                            else:
                                # self.xml_relative_paths.append(None)
                                print('NOT FIND XML for image -> {}'.format(jpg_path))
                cate2count[dir_] = count
        return cate2count

    def __load_coco_data(self, json_file_name):
        """
        load coco
        :param json_file_name:
        :return:
        """
        assert json_file_name.endswith(".json"), 'the json file ends with: {}'.format(json_file_name.split('.')[1])
        file_path = os.path.join(self.sample_root, json_file_name)
        assert os.path.exists(file_path), 'the file not exist: {}'.format(file_path)

        with open(file_path) as f:
            coco_dict = json.load(f)
        return coco_dict

    def get_cate_dist_on_coco_json(self, json_file_name):
        """
        get the category distribution on coco dataset
        :param json_file_name:
        :return:
        """
        coco_dict = self.__load_coco_data(json_file_name)

        cate2count = dict()
        cate_id2name = dict()
        categories = coco_dict['categories']
        for category in categories:
            cate_name = category['name']
            cate_id = category['id']
            cate2count[cate_name] = 0
            cate_id2name[cate_id] = cate_name

        for anno in coco_dict['annotations']:
            if anno['category_id'] in cate_id2name.keys():
                cate_name = cate_id2name[anno['category_id']]
                cate2count[cate_name] += 1

        return cate2count

    def get_area_list_on_coco(self, json_file_name):
        """
        get the area list for each bbox
        :param json_file_name:
        :return:
        """
        coco_dict = self.__load_coco_data(json_file_name)

        cate2area_list = dict()
        cate_id2name = dict()
        categories = coco_dict['categories']
        for category in categories:
            cate_name = category['name']
            cate_id = category['id']
            cate2area_list[cate_name] = []
            cate_id2name[cate_id] = cate_name

        for anno in coco_dict['annotations']:
            if anno['category_id'] in cate_id2name.keys():
                cate_name = cate_id2name[anno['category_id']]
                cate_area = anno['area']
                cate2area_list[cate_name].append(cate_area)

        return cate2area_list

    def get_w_h_list_on_coco(self, json_file_name):
        """
        get the (width, height) list for all the bbox
        :param json_file_name:
        :return:
        """
        coco_dict = self.__load_coco_data(json_file_name)

        cate2w_h_list = dict()
        cate_id2name = dict()
        categories = coco_dict['categories']
        for category in categories:
            cate_name = category['name']
            cate_id = category['id']
            # cate2area_list[cate_name] = []
            cate_id2name[cate_id] = cate_name

        for anno in coco_dict['annotations']:
            if anno['category_id'] in cate_id2name.keys():
                cate_name = cate_id2name[anno['category_id']]
                cate_w = anno['bbox'][2]
                cate_h = anno['bbox'][3]
                cate2w_h_list[cate_name].append((cate_w, cate_h))

        return cate2w_h_list


# if __name__ == "__main__":
#
#     from data_explore import plotting
#     categoryDist = CategoryDist('/home/scj/mm_detection_proj/stations/boe_b2/trainData')
#     cate2count = categoryDist.get_dist_on_img_xml()
#     cate2count = categoryDist.get_cate_dist_on_coco_json('train.json')
#
#     cate2area_list = categoryDist.get_area_list_on_coco('train.json')
#     for cate in cate2area_list.keys():
#         v_list = cate2area_list[cate]
#         plotting.hist_plot(v_list, '{}.png'.format(cate), cate, bins=30)
#
#     # plotting.bar_plot(cate2count, 'test_2.png')
#     # plotting.pie_plot(cate2count, 'test1.png')






















