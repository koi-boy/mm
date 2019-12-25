#!/usr/bin/env python
# encoding: utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software: 
@file: combine_two_sets.py
@time: 2019/9/21 17:22
@desc:
"""
import os
import shutil


def combine(first_root, second_root, combine_root):
    """
    Combine the two dataset into one
    :param first_root:
    :param second_root:
    :param combine_root:
    :return:
    """

    first_codes = os.listdir(first_root)
    second_codes = os.listdir(second_root)
    all_codes = set(first_codes + second_codes)

    if not os.path.exists(combine_root):
        os.makedirs(combine_root)
    else:
        shutil.rmtree(combine_root, True)
        os.makedirs(combine_root)

    for code in all_codes:
        print('codes is: ' + code)
        if not os.path.exists(os.path.join(combine_root, code)):
            os.makedirs(os.path.join(combine_root, code))

        if code in first_codes:
            code_path = os.path.join(first_root, code)
            print('in {} contains {} files.'.format(first_root, len(os.listdir(code_path))))
            for file_ in os.listdir(code_path):
                if os.path.isfile(os.path.join(code_path, file_)):
                    shutil.copy(os.path.join(code_path, file_), os.path.join(combine_root, code))

        if code in second_codes:
            code_path = os.path.join(second_root, code)
            print('in {} contains {} files.'.format(second_root, len(os.listdir(code_path))))
            for file_ in os.listdir(code_path):
                if os.path.isfile(os.path.join(code_path, file_)):
                    shutil.copy(os.path.join(code_path, file_), os.path.join(combine_root, code))


if __name__ == '__main__':

    first_root = '/home/WXN_V3/ARRAY/1'
    second_root = '/home/WXN_V3/ARRAY/2'
    combine_root = '/home/WXN_V3/ARRAY/combine'
    combine(first_root, second_root, combine_root)

