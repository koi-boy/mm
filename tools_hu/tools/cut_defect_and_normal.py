# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:02:56 2019

@author: hkf
"""

import xml.etree.ElementTree as ET
import os
import cv2
from tools_hu.utils.utils import bboxes_iou
from tools_hu.utils.Code_dictionary import CodeDictionary
import numpy as np
import random


def cut_defect(images, annotations, output_dir, code_dict=None, cut_normal=False):
    for root_dir, _, files in os.walk(images):
        for file in files:
            if not (file.endswith('jpg') or (file.endswith('JPG'))):
                continue
            img_path = os.path.join(root_dir, file)
            xml = file.replace('jpg', 'xml')
            xml_path = os.path.join(annotations, xml)
            try:
                tree = ET.parse(xml_path)
            except Exception as e:
                print('no file named {}'.format(xml_path))
                continue
            root = tree.getroot()
            objs = root.findall('object')
            img = cv2.imread(img_path)
            H, W, D = img.shape
            start_id = 0
            gt_bbox = []
            for obj in objs:
                category = obj[0].text
                if code_dict is not None:
                    category = str(code_dict.code2id(category))
                bbox = [int(float(obj[4][i].text)) for i in range(4)]
                cut = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                gt_bbox.append(np.array(bbox))
                save_dir_code = os.path.join(output_dir, category)
                if not os.path.exists(save_dir_code):
                    os.makedirs(save_dir_code)
                cut_name = '__'.join(
                    [file[:-4], category, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]) + '.jpg'
                start_id += 1
                cut_path = os.path.join(save_dir_code, cut_name)
                cv2.imwrite(cut_path, cut)
            if cut_normal:
                samples = 5
                save_dir_code = os.path.join(output_dir, 'normal')
                if not os.path.exists(save_dir_code):
                    os.makedirs(save_dir_code)
                for _ in range(samples):
                    size = random.sample(ANCHOR_SCALE, 1)[0]
                    ratio = random.sample(ANCHOR_RATIO, 1)[0]
                    width = int(size*ratio)
                    height = int(width/ratio)
                    if width >= W or height >= H:
                        continue
                    xmin = random.randint(0, W - width)
                    ymin = random.randint(0, H - height)
                    xmax = xmin + width
                    ymax = ymin + height
                    bbox = [xmin, ymin, xmax, ymax]
                    if np.all(bboxes_iou(np.array(bbox), np.array(gt_bbox))<0.3):
                        cut = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                        cut_name = '__'.join(
                            [file[:-4], 'normal', str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]) + '.jpg'
                        cut_path = os.path.join(save_dir_code, cut_name)
                        cv2.imwrite(cut_path, cut)
                    else:
                        print('skip')
                        continue


if __name__ == '__main__':
    defect_dir = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\top\train'
    xml_dir = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\top\train'
    save_dir = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\top\cut'
    category_file = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\top\classes.txt'
    id_file = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\top\id.txt'
    cd = CodeDictionary(category_file, id_file)

    ANCHOR_SCALE = [32, 64, 128, 256]
    ANCHOR_RATIO = [0.2, 0.5, 1.0, 2.0, 5.0]

    cut_defect(defect_dir, xml_dir, save_dir, cd, cut_normal=True)
