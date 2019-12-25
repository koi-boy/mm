#!/usr/bin/env python
# encoding:utf-8
"""
author: liusili
@l@icense: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: liusili@unionbigdata.com
@software:
@file: flask_infer
@time: 2019/12/2
@desc:
"""
import os
import cv2
import numpy as np
import time, datetime
from mmdet.apis import init_detector, inference_detector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
basedir = '/home/Visionox/V3/OLED_deploy/'
model = None
labels = []
# NMS_THD = 0.55
NMS_THD = 0.3

def initModel():
    global model
    global labels
    config_file = basedir + 'config_OLED.py'
    checkpoint_file = basedir + 'v3_oled_deploy.pth'
    for line in open(basedir + 'classes.txt', "r"):
        lineTemp = line.strip()
        if lineTemp:
            labels.append(lineTemp)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')


def NMS(bboxes, score, thresh):
    """Pure Python NMS baseline."""
    # bounding box and score
    boxes = np.array(bboxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = np.array(score)
    # the area of candidate
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # score in descending order
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Calculate the intersection between current box and other boxes
        # using numpy->broadcast, obtain vector
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # intersection area, return zero if no intersection
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # IOU：intersection area /（area1+area2-intersection area）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # find out box the overlap ratio smaller than threshold
        inds = np.where(ovr <= thresh)[0]
        # update order
        order = order[inds + 1]
    return keep


def selectClsScoreBoxFromResult(result, cls_names):
    assert isinstance(cls_names, (tuple, list))

    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)
    selectedCls = []
    selectedScore = []
    selectedBox = []
    assert (len(labels) == len(bboxes))
    for i in range(0, len(labels)):
        # selectedResult.append([cls_names[labels[i]], bboxes[i][-1]])
        selectedCls.append(cls_names[labels[i]])
        selectedScore.append(bboxes[i][-1])
        tempBox = []
        tempBox = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
        selectedBox.append(tempBox)
    return selectedCls, selectedScore, selectedBox


def infer(sample_root, outpath):
    global model
    for code in os.listdir(sample_root):
        code_path = os.path.join(sample_root, code)
        for img_name in os.listdir(code_path):
            imagepath = os.path.join(code_path, img_name)
            img = open(imagepath, 'rb').read()
            if img == None:
                print('img is none')
            nparr = np.fromstring(img, np.uint8)
            img_np = cv2.imdecode(nparr, 1)
            # 边缘裁剪
            img_np = img_np[:, :1228, :]
            # opzealot
            height = img_np.shape[0]
            width = img_np.shape[1]

            sys_time = int(int(round(time.time() * 1000)))
            cur_dir = os.getcwd()
            localtime = time.localtime(time.time())
            result = {}
            result['defect'] = 0
            out = inference_detector(model, img_np)

            log_codes = []
            log_scores = []
            bboxs = []
            log_codes, log_scores, bboxs = selectClsScoreBoxFromResult(out, labels)
            if len(log_codes) != 0:
                result['defect'] = 1
                validResult = np.arange(0, len(bboxs))
                if len(bboxs) > 1:
                    validResult = NMS(bboxs, log_scores, NMS_THD)

                for index in validResult:
                    # ignore edges codes
                    xmin = bboxs[index][0]
                    ymin = bboxs[index][1]
                    xmax = bboxs[index][2]
                    ymax = bboxs[index][3]

                    center_x = (xmin + xmax) // 2
                    center_y = (ymin + ymax) // 2

                    if center_x < 100 or center_y < 100 or center_x > width - 100 \
                            or center_y > height - 100:
                        log_scores[index] = 0

                    if log_scores[index] > 0:
                        cv2.rectangle(img_np, (bboxs[index][0], bboxs[index][1]),
                                      (bboxs[index][2], bboxs[index][3]), (0, 255, 255), thickness=2)
                        strText = str(code) + ': ' + str(log_scores[index])
                        cv2.putText(img_np, strText, (bboxs[index][0], bboxs[index][1]),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

                target_img_dir = outpath
                os.makedirs(target_img_dir, exist_ok=True)
                target_img_file_path = os.path.join(target_img_dir, img_name)
                cv2.imwrite(target_img_file_path, img_np)
                print('save img {}'.format(img_name))

                result['log_codes'] = log_codes
                result['log_score'] = str(log_scores)

                out_label = None
                out_score = None
                out_bbox = None
                if len(log_scores) == 0:
                    out_label = 'Others'
                    out_score = str(0.0)
                    out_bbox = None
                else:
                    out_score = max(log_scores)
                    out_label = log_codes[log_scores.index(out_score)]
                    out_bbox = bboxs[log_scores.index(out_score)]
                    if out_score < 0.4:
                        out_label = 'Others'

                    # opzealot set the background threshold
                    if out_score < 0.2:
                        out_label = 'OK'
                        out_score = 0.99
                result['img_cls'] = out_label
                result['img_score'] = str(out_score)
                result['detect_begin_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result['detect_cost_time'] = '{:.2f}'.format(int(int(round(time.time() * 1000))) - sys_time)
                result['savepath'] = imagepath.replace('input', 'result')

            else:
                target_img_dir = outpath
                os.makedirs(target_img_dir, exist_ok=True)
                target_img_file_path = os.path.join(target_img_dir, img_name)
                cv2.imwrite(target_img_file_path, img_np)
                print('save image to {}'.format(target_img_file_path))

                result['defect'] = 1
                result['img_cls'] = 'OK'
                result['img_score'] = str(0.99)
                result['detect_begin_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                result['detect_cost_time'] = '{:.2f}'.format(int(int(round(time.time() * 1000))) - sys_time)
                result['savepath'] = target_img_file_path


if __name__ == '__main__':
    initModel()
    imagepath = '/home/Visionox/V3/OLED_deploy/OLED_test'
    outpath = '/home/Visionox/V3/OLED_deploy/output'
    infer(imagepath, outpath)