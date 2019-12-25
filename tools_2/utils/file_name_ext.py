#!/usr/bin/env python
# encoding: utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software: 
@file: file_name_ext.py
@time: 2019/9/30 11:11
@desc:
"""
import os


def is_supported_img_ext(img_name):
    """

    :param img_name:
    :return:
    """
    return get_file_ext(img_name) in ['JPG', 'jpg', 'JPEG', 'jpeg']


def get_file_ext(file_name):
    """
    get the extension name of file
    :param file_name:
    :return:
    """

    return os.path.splitext(file_name)[-1][1:]


def get_file_name(file_name):
    """
    get the name of file
    :param file_name: 
    :return: 
    """
    return os.path.splitext(file_name)[0]

