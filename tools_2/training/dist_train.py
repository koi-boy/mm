#!/usr/bin/env python
# encoding: utf-8
"""
@author: sunchongjing
@license: (C) Copyright 2019, Union Big Data Co. Ltd. All rights reserved.
@contact: sunchongjing@unionbigdata.com
@software: 
@file: dist_train.py
@time: 2019/10/9 17:59
@desc:
"""

# import torch.distributed.launch
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import subprocess
import os
from argparse import ArgumentParser, REMAINDER
import training.train


def one_node_multi_gpu(master_port, nproc_per_node, training_script_args, use_gpus='0'):
    """

    :param master_port:
    :param nproc_per_node: the number of processes
    :param training_script_args: the args for training
    :param use_gpus: str such as '0,1', '2,0,3'
    :return:
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = use_gpus
    # world size in terms of number of processes
    dist_world_size = nproc_per_node * 1

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = str(master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, nproc_per_node):
        # each process's rank
        dist_rank = local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        print(training_script_args)
        # print(sys.path[0])
        script_path = os.path.dirname(os.path.abspath(__file__))
        # print(os.path.dirname(os.path.abspath(__file__)))
        cmd = [sys.executable,
               "-u",
               os.path.join(script_path, 'train.py'),
               "--local_rank={}".format(local_rank),
               "--launcher={}".format("pytorch"),
               "--visible_gpus={}".format(use_gpus)] + training_script_args

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utilty that will spawn up "
                                        "multiple distributed processes")

    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for GPU training, this is recommended to be set "
                             "to the number of GPUs in your system so that "
                             "each process can be bound to a single GPU.")

    parser.add_argument("--use_gpus", default='0',
                        help="the gpu indexes can be used by this process, can be set as '0,1', '3,5'")

    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communciation during distributed "
                             "training")

    # rest from the training program
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()


def one_node_multi_gpu_main():
    args = parse_args()
    nproc_per_node = args.nproc_per_node
    use_gpus = args.use_gpus
    master_port = args.master_port
    training_script_args = args.training_script_args
    one_node_multi_gpu(master_port, nproc_per_node, training_script_args, use_gpus)


if __name__ == "__main__":

    one_node_multi_gpu_main()


