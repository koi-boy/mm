#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import time
import glob
import json
import numpy as np
from mmdet.apis import inference_detector, init_detector
from progressbar import ProgressBar


def model_test(imgs, cfg_file, ckpt_file,
               score_thr=0.1, device='cuda:0'):
    model = init_detector(cfg_file, ckpt_file, device=device)
    pbar = ProgressBar().start()
    img_dict, anno_dict = [], []
    for i, img in enumerate(imgs):  # loop for images
        # st = time.time()
        pbar.update(int(i / (len(imgs) - 1) * 100))

        img_name = img.split('/')[-1]
        img_id = i + 1
        img_dict.append({"file_name": img_name, "id": img_id})

        img_data = cv2.imread(img)
        h, w, _ = img_data.shape
        if w < 1000:
            class_dict = [1, 2, 3, 4, 5, 9, 10]
        else:
            class_dict = [6, 7, 8]

        result = inference_detector(model, img_data)
        total_bbox = []
        for idx, bboxes in enumerate(result):  # loop for categories
            category_id = idx + 1
            if len(bboxes) != 0 and category_id in class_dict:
                for bbox in bboxes:  # loop for bbox
                    conf = bbox[4]
                    if conf > score_thr:
                        total_bbox.append(list(bbox) + [category_id])

        for bbox in np.array(total_bbox):
            xmin, ymin, xmax, ymax = bbox[:4]
            coord = [xmin, ymin, xmax - xmin, ymax - ymin]
            coord = [round(x, 2) for x in coord]
            conf = round(bbox[4], 2)
            category_id = int(bbox[5])
            anno_dict.append({'image_id': img_id, 'bbox': coord, 'category_id': category_id, 'score': conf})
        # print(i, img_name, time.time() - st)
    pbar.finish()
    return img_dict, anno_dict


if __name__ == '__main__':
    imgs = glob.glob('/data/sdv1/whtm/data/cq/test/images/*.jpg')
    cfg_file = '/data/sdv1/whtm/mmdet_cq/CQ_cfg/HU_cfg/ga_faster_x101_32x4d_fpn_1x_all.py'
    ckpt_file = '/data/sdv1/whtm/a/CQ_work_dirs/ga_faster_rcnn_x101_32x4d_mdconv_fpn_all/epoch_6.pth'

    img_dict, anno_dict = model_test(imgs, cfg_file, ckpt_file, score_thr=0.0)
    predictions = {"images": img_dict, "annotations": anno_dict}
    with open('/data/sdv1/whtm/result/cq/test/GA_FASTER_0209_01.json', 'w') as f:
        json.dump(predictions, f, indent=4)
