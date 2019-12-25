#!/usr/bin/env python
# encoding:utf-8
"""
author: sunchongjing
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software:
@file: voc2coco.py
@time: 2019/9/6 16:37
@desc:
"""
from sklearn.model_selection import train_test_split, RepeatedKFold
import xml.etree.ElementTree as ET
import os, json
from utils import file_name_ext


class Voc2Coco(object):

    def __init__(self, sample_root, xml_img_in_same_folder=True):
        """

        :param sample_root: root of images with voc xml files
        """
        self.sample_root = sample_root
        self.xml_img_in_same_folder = xml_img_in_same_folder
        self.img_relative_paths = []
        self.xml_relative_paths = []

        self.get_img_relative_paths()
        self.get_xml_relative_paths()
        self.image_labels = sorted([item.split('/')[0] for item in self.xml_relative_paths])  # first part as image code name

    def get_img_relative_paths(self):
        """
        get the relative paths of all jpg/JPG images
        :return:
        """
        # filename = os.path.join(path, 'img_list.txt')
        # f = open(filename, 'w')

        for root, dirs, files in os.walk(self.sample_root):
            for file in files:
                if not file_name_ext.is_supported_img_ext(file):
                    continue
                filepath = os.path.join(root, file)
                file_rel_path = os.path.relpath(filepath, self.sample_root)
                self.img_relative_paths.append(file_rel_path)

    def get_xml_relative_paths(self):
        """
        get the xml relative path for each image in img_relative_paths .

        :return:
        """
        can_used_img_rel_path = []
        for jpg_rel_path in self.img_relative_paths:
            xml_rel_path = os.path.splitext(jpg_rel_path)[0] + '.xml'
            if not self.xml_img_in_same_folder:
                xml_rel_path = xml_rel_path.replace('images', 'labels')

            if os.path.exists(os.path.join(self.sample_root, xml_rel_path)):
                self.xml_relative_paths.append(xml_rel_path)
                can_used_img_rel_path.append(jpg_rel_path)
            else:
                # self.xml_relative_paths.append(None)
                print('NOT FIND XML for image -> {}'.format(jpg_rel_path))
        self.img_relative_paths = can_used_img_rel_path

    fww1030_image_id = 10000000
    fww1030_bounding_box_id = 10000000

    @staticmethod
    def get_current_image_id():
        """
        :return: the global image id
        """
        # global fww1030_image_id
        Voc2Coco.fww1030_image_id += 1
        return Voc2Coco.fww1030_image_id

    @staticmethod
    def get_current_annotation_id():
        """
        get the global annotation id
        :return:
        """
        # global fww1030_bounding_box_id
        Voc2Coco.fww1030_bounding_box_id += 1
        return Voc2Coco.fww1030_bounding_box_id

    @staticmethod
    def get_and_check(root, name, length):
        """

        :param root: the element of ElementTree
        :param name: the name of sub-element
        :param length: the number of sub-element with name as parameter name
        :return:
        """
        var_lst = root.findall(name)
        if len(var_lst) == 0:
            raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
        if (length > 0) and (len(var_lst) != length):
            raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (
                name, length, len(var_lst)))
        if length == 1:
            var_lst = var_lst[0]
        return var_lst

    def convert(self, img_idxes, json_file, categories=None):
        """
        convert the voc format into coco format.

        :param img_idxes: list of index for image in self.img_relative_paths
        :param categories: the category list, that we want to train model with
        :param json_file: the name of saved coco json file
        :return:
        """
        json_dict = {"images": [], "type": "instances", "annotations": [],
                     "categories": []}
        if categories is None:
            categories = list(set(self.image_labels))

        for idx in img_idxes:
            xml_file = os.path.join(self.sample_root, self.xml_relative_paths[idx])

            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                image_id = self.get_current_image_id()
                size = self.get_and_check(root, 'size', 1)
                width = int(self.get_and_check(size, 'width', 1).text)
                height = int(self.get_and_check(size, 'height', 1).text)

                # 构造image
                image = {'file_name': self.img_relative_paths[idx], 'height': height, 'width': width,
                         'id': image_id}
                json_dict['images'].append(image)

                for obj in root.findall('object'):
                    category = self.get_and_check(obj, 'name', 1).text

                    if category == 'GO_1_GI Open':  # for B2 wrong labeled xml
                        category = 'GO_1_GI_Open'

                    if category not in categories:
                        print('skip annotation {}'.format(category))
                        continue

                    category_id = categories.index(category) + 1

                    bndbox = self.get_and_check(obj, 'bndbox', 1)
                    xmin = int(self.get_and_check(bndbox, 'xmin', 1).text)
                    ymin = int(self.get_and_check(bndbox, 'ymin', 1).text)
                    xmax = int(self.get_and_check(bndbox, 'xmax', 1).text)
                    ymax = int(self.get_and_check(bndbox, 'ymax', 1).text)
                    if (xmax <= xmin) or (ymax <= ymin):
                        print('{} error'.format(xml_file))
                        continue
                    o_width = (xmax - xmin) + 1
                    o_height = (ymax - ymin) + 1
                    ### opzealot 进行魔改
                    ### 对面积小于3000像素的bdbox进行扩大
                    # area = o_width * o_height
                    # if area < 100000:
                    #     xmin = max(xmin - 100, 0)
                    #     ymin = max(ymin - 100, 0)
                    #     xmax = min(xmax + 100, width)
                    #     ymax = min(ymax + 100, height)
                    #     o_width = (xmax - xmin) + 1
                    #     o_height = (ymax - ymin) + 1
                    
                    
                    
                    ### opzealot 改完了
                    
                    ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': image_id,
                           'bbox': [xmin, ymin, o_width, o_height],
                           'category_id': category_id, 'id': self.get_current_annotation_id(), 'ignore': 0,
                           'segmentation': []}
                    json_dict['annotations'].append(ann)
            except NotImplementedError:
                print('xml {} file error!'.format(self.xml_relative_paths[idx]))

        for cid, cate in enumerate(categories):
            cat = {'supercategory': 'FWW', 'id': cid + 1, 'name': cate}
            json_dict['categories'].append(cat)

        json_file = os.path.join(self.sample_root, json_file)
        with open(json_file, 'w') as f:
            json.dump(json_dict, f, indent=4)

    def get_train_test_json(self, test_size=0.1, random_state=666, categories=None):
        """

        :param test_size:
        :param random_state:
        :param categories:
        :return:
        """

        train_idxes, test_idxes = train_test_split(
            list(range(len(self.img_relative_paths))), test_size=test_size, random_state=random_state,
            stratify=self.image_labels)
        self.convert(train_idxes, 'train.json', categories)
        self.convert(test_idxes, 'test.json', categories)

    def repeated_k_fold(self, repeats=1, k_fold=5, random_state=666):
        kf = RepeatedKFold(n_repeats=repeats, n_splits=k_fold, random_state=random_state)
        count = 0
        for train_idx, test_idx in kf.split(self.img_relative_paths):
            count += 1
            self.convert(train_idx, 'train_{}.json'.format(count))
            self.convert(test_idx, 'test_{}.json'.format(count))

    def get_test_json(self, categories=None):
        """

        :param test_size:
        :param random_state:
        :param categories:
        :return:
        """

        test_idxes = list(range(len(self.img_relative_paths)))
        self.convert(test_idxes, 'test.json', categories)


if __name__ == '__main__':
    sample_root_ = '/home/Visionox/V3/poc_data/OLED_ultimate'

    data_convert = Voc2Coco(sample_root_)
    data_convert.get_test_json()


    # data_convert.repeated_k_fold()





