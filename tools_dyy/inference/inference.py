#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    for i, img in enumerate(imgs):   # loop for images
        # st = time.time()
        pbar.update(int(i / (len(imgs) - 1) * 100))
        img_name = img.split('/')[-1]
        img_id = i + 1
        img_dict.append({"file_name": img_name, "id": img_id})
        
        result = inference_detector(model, img)
        total_bbox = []
        for idx, bboxes in enumerate(result):  # loop for categories
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox in bboxes:  # loop for bbox
                    conf = bbox[4]
                    if conf > score_thr:
                        total_bbox.append(list(bbox) + [category_id])
                    
        for bbox in np.array(total_bbox):
            xmin, ymin, xmax, ymax = bbox[:4]
            coord = [xmin, ymin, xmax-xmin, ymax-ymin]
            coord = [round(x, 2) for x in coord]
            conf = round(bbox[4], 2)
            category_id = int(bbox[5])
            anno_dict.append({'image_id': img_id, 'bbox': coord, 'category_id': category_id, 'score': conf})
        # print(i, img_name, time.time() - st)
    pbar.finish()
    return img_dict, anno_dict


if __name__ == '__main__':
    imgs = glob.glob('../a/CQ/test/*.jpg')
    cfg_file = 'cfg/1224_baseline.py'
    ckpt_file = '../a/CQ_work_dirs/baseline/epoch_24.pth'
    
    img_dict, anno_dict = model_test(imgs, cfg_file, ckpt_file, score_thr=0.)
    predictions = {"images": img_dict, "annotations": anno_dict}
    with open('baseline_keep15.json', 'w') as f:
        json.dump(predictions, f, indent=4)