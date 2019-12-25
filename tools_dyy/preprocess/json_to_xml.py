#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, shutil
from pascal_voc_io import PascalVocWriter


if __name__ == '__main__':

    root = r'C:\Users\EDZ\Desktop\chongqing1_round1_train1_20191223'
    img_dir = os.path.join(root, 'images')
    json_file = os.path.join(root, 'annotations.json')

    with open(json_file) as f:
        data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    class_d = {}
    for cate in categories:
        id = cate['id']
        name = cate['name']
        class_d[id] = name
    # print(class_d)

    for img in images:
        img_name = img['file_name']
        img_id = img['id']
        height = img['height']
        width = img['width']
        localImgPath = os.path.join(img_dir, img_name)
        imgSize = [height, width, 3]
        XMLWriter = PascalVocWriter(img_dir, img_name, imgSize, localImgPath)
        count = 0
        for anno in annotations:
            if anno['image_id'] == img_id:
                x, y, w, h = anno['bbox']
                category = anno['category_id']
                if category != 0:
                    XMLWriter.addBndBox(x, y, x+w, y+h, class_d[category])
                    count += 1
        if count == 0:
            save_dir = os.path.join(root, 'normal')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            shutil.copy(localImgPath, os.path.join(save_dir, img_name))
        else:
            save_dir = os.path.join(root, 'defect')
            xml_dir = os.path.join(root, 'defect_xmls')
            for dir in [save_dir, xml_dir]:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            shutil.copy(localImgPath, os.path.join(save_dir, img_name))
            XMLWriter.save(os.path.join(xml_dir, img_name[:-4]+'.xml'))












