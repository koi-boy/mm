#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json
import numpy as np


def get_bboxes(result_json, is_gt, score_thr=0.1):
    with open(result_json, 'r') as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    img_dict = {}
    for img in images:
        img_name = img['file_name']
        img_id = img['id']
        img_dict[img_id] = img_name

    bbox_dict = {}
    for anno in annotations:
        img_id = anno['image_id']
        img_name = img_dict[img_id]
        category_id = anno['category_id']
        bbox = anno['bbox']
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        if is_gt:
            bbox_dict.setdefault(img_name, []).append(bbox + [category_id])
        else:
            score = anno['score']
            if score > score_thr:
                bbox_dict.setdefault(img_name, []).append(bbox + [category_id] + [score])
    return bbox_dict


def results_plot(det_dict, gt_dict, img_dir, save_dir):
    imgs = os.listdir(img_dir)
    for i, img_name in enumerate(imgs):
        print(i, img_name)
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if gt_dict is not None:
            for gt_bbox in np.array(gt_dict[img_name]):
                bbox_int = gt_bbox[:4].astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                label_txt = str(int(gt_bbox[4]))
                cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 1)
                cv2.putText(img, label_txt, (bbox_int[2], max(bbox_int[3] + 2, 0)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
        try:
            for bbox in np.array(det_dict[img_name]):
                bbox_int = bbox[:4].astype(np.int32)
                left_top = (bbox_int[0], bbox_int[1])
                right_bottom = (bbox_int[2], bbox_int[3])
                label_txt = str(int(bbox[4])) + ': ' + str(bbox[5])
                cv2.rectangle(img, left_top, right_bottom, (0, 0, 255), 1)
                cv2.putText(img, label_txt, (bbox_int[0], max(bbox_int[1] - 2, 0)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
            cv2.imwrite(os.path.join(save_dir, img_name), img)
        except KeyError:
            cv2.imwrite(os.path.join(save_dir, img_name), img)


if __name__ == '__main__':
    # det_json = '/Users/dyy/Desktop/test.json'
    # gt_json = '/Users/dyy/Desktop/CQ/val.json'
    #
    # det_dict = get_bboxes(det_json, is_gt=False, score_thr=0.01)
    # gt_dict = get_bboxes(gt_json, is_gt=True)
    #
    # img_dir = '/Users/dyy/Desktop/CQ/val'
    # save_dir = '/Users/dyy/Desktop/output'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # results_plot(det_dict, gt_dict, img_dir, save_dir)

    det_json = '/Users/dyy/Desktop/cascade_garpn_0211.json'
    det_dict = get_bboxes(det_json, is_gt=False, score_thr=0.01)

    img_dir = '/Users/dyy/Desktop/chongqing1_round1_testB_20200210/images'
    save_dir = '/Users/dyy/Desktop/testB_output2'
    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    results_plot(det_dict, None, img_dir, save_dir)