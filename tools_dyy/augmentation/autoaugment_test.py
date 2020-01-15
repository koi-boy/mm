#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, cv2
import numpy as np
from autoaugment import *

if __name__ == '__main__':

    root = '/Users/dyy/Desktop/CQ/'
    img_dir = os.path.join(root, 'train')
    json_file = os.path.join(root, 'train.json')
    save_dir = os.path.join(root, 'aug')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(json_file) as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    for img in images:
        img_name = img['file_name']
        img_id = img['id']
        height = img['height']
        width = img['width']
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        bboxes = []
        for anno in annotations:
            if anno['image_id'] == img_id:
                x, y, w, h = anno['bbox']
                min_y = y / height
                min_x = x / width
                max_y = (y+h) / height
                max_x = (x+w) / width
                bboxes.append([min_y, min_x, max_y, max_x])
        bboxes = np.array(bboxes)

        augmented_image, augmented_bboxes = distort_image_with_autoaugment(
            image, bboxes, 'test')
        for bbox in augmented_bboxes:
            ymin, xmin, ymax, xmax = int(bbox[0] * height), int(bbox[1] * width), int(bbox[2] * height), int(bbox[3] * width)
            cv2.rectangle(augmented_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_dir, img_name), augmented_image)


