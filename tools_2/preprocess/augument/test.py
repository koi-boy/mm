#!/usr/bin/env python
# encoding: utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software: 
@file: test.py
@time: 2019/9/9 18:20
@desc:
"""

import numpy as np
import cv2
import random
from preprocess.augument.bbox_util import draw_rect

img = cv2.imread('/home/scj/mm_detection_proj/stations/boe_b2/trainData/Active_Remain(N)/A090960001BDN1_A_NA_106_4_201906201438_001.jpg')
bboxes = np.array([[300, 400, 500, 600]], dtype=float)

cv2.imwrite('test.jpg', draw_rect(img, bboxes))

from preprocess.augument.data_aug import HorizontalFlip

rotate = HorizontalFlip()
new_img, new_bbox = rotate.__call__(img, bboxes)
#


# import matplotlib
# matplotlib.use('Agg')
#
# import matplotlib.pyplot as plt
# plt.imshow(draw_rect(img, bboxes))
# plt.show()


cv2.imwrite('test_1.jpg', draw_rect(new_img, new_bbox))

# cv2.imwrite('test.jpg', draw_rect(img, bboxes))

# img = draw_rect(img, bboxes)
# cv2.
# plt.savefig(img, 'test.jpg')
# # cv2.imshow(img)
# # plt.imshow(draw_rect(new_img, new_bbox))
