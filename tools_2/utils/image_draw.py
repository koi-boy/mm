#!/usr/bin/env python
# encoding: utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software: 
@file: image_draw.py
@time: 2019/9/30 13:24
@desc:
"""

import cv2

# Text
TEXT_FONT = cv2.FONT_HERSHEY_TRIPLEX
TEXT_FONT_SCALE = 1
TEXT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)

# BBox
BBOX_COLOR = (0, 255, 0)
BBOX_THICKNESS = 1


def draw_bbox_with_code(img, code, bbox, xmin_plus=0, ymin_plus=-10):
    """

    :param img:
    :param code:
    :param bbox:
    :param xmin_plus: text on bbox, width direction
    :param ymin_plus: text on bbox, height direction
    :return:
    """
    img = draw_rectangle(img, bbox)

    return put_text_on_img(img, str(code), (int(bbox[0]) + xmin_plus, int(bbox[1]) + ymin_plus))


def draw_bbox_with_code_conf(img, code, conf, bbox, xmin_plus=0, ymin_plus=-10):
    """
    draw bbox with text as 'code: conf'
    :param img:
    :param code:
    :param conf:
    :param bbox:
    :param xmin_plus: text on bbox, width direction
    :param ymin_plus: text on bbox, height direction
    :return:
    """
    img = draw_rectangle(img, bbox)
    text = code + ':{:.3f}'.format(conf)
    return put_text_on_img(img, text, (int(bbox[0]) + xmin_plus, int(bbox[1]) + ymin_plus))


def put_text_on_img(img, text, org):
    """
    
    :param img: 
    :param text: text
    :param org: the left-up point coordination
    :return: 
    """
    return cv2.putText(img, text, org,
                       TEXT_FONT, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)  # 字体，字体大小，颜色，字体粗细


def draw_rectangle(img, bbox):
    """

    :param img:
    :param bbox:
    :return:
    """
    assert len(bbox) == 4, 'the bbox should contains 4 elements'
    return cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                         BBOX_COLOR, thickness=BBOX_THICKNESS)


if __name__ == '__main__':

    img = cv2.imread('/home/adc_work_dir/OLED_1/C5DH/L2E9601A8061AA_-676831_390842_OVER_A_DF_BL_R_REV_C5DH_RS.jpg')
    # img = put_text_on_img(img, 'test:0.9865', (600, 600))
    # cv2.imwrite('test.jpg', img)

    bbox = [600, 600, 800, 900]
    code = 'CODE_A'
    conf = 0.93568
    img = draw_bbox_with_code_conf(img, code, conf, bbox)
    cv2.imwrite('test.jpg', img)

