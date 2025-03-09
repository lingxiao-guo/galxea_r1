#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
import math
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle as pkl
import matplotlib.pyplot as plt
import PIL
import yaml
from glob import glob
from tqdm import tqdm
import re
from PIL import Image

VIS = False

def _init(q):
    gpu_id = q.get()
    # DO NOT "import open3d as o3d" since o3d init gpu stream
    gpu_id = int(gpu_id)
    if gpu_id not in visible_gpu_id_list:
        list_id = gpu_id % len(visible_gpu_id_list)
        gpu_id = visible_gpu_id_list[list_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def gpu_warpper(handle, jobs, gpu_num=len(visible_gpu_id_list)):
    # DO NOT import open3d before
    num_proc = min(gpu_num, len(jobs))
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    for i in range(num_proc):
        q.put(i % gpu_num)
    cnt = 0
    total = 0
    with ProcessPoolExecutor(num_proc, ctx, _init, (q,)) as ex:
        for res in ex.map(handle, jobs):
            # if res:
            #     break
            pass
            # cnt += res
            # total += 1
            # print(cnt, total)

def extract_number(filename):
    filename = os.path.basename(filename)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def main_fun(seq_idx):
    import open3d as o3d
    from src.lidar2cam import CAM_PARAS_DICT, LiDAR2Ego, calculate_pose
    
    seq_name = f'{str(seq_idx).zfill(3)}'


def main():
    parser = argparse.ArgumentParser(description='',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=798)
    parser.add_argument('--gpu', type=int, default=8, )
    args = parser.parse_args()

    jobs = parse_list # [i for i in range(args.start, args.end)]
    gpu_warpper(main_fun, jobs, gpu_num=args.gpu)

def debug():
    for seq_idx in parse_list:
        main_fun(seq_idx)

if __name__ == "__main__":
    main()