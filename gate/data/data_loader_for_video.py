# -*- coding: utf-8 -*-
""" updated: 2017/3/28
"""
import os
import math
import random

import numpy as np
from PIL import Image


def compress_multi_imgs_to_one(foldpath, channels=16, is_training=True):
    """
        channels: how many pictures will be compressed.
    """
    fold_path_abs = str(foldpath, encoding='utf-8')
    img_list = [fs for fs in os.listdir(fold_path_abs) if len(fs.split('.jpg')) > 1]

    # generate idx without reptitious
    # img_indice = random.sample([i for i in range(len(img_list))], channels)
    invl = math.floor(len(img_list) / float(channels))
    start = 0
    img_indice = []

    for _ in range(channels):
        end = start + invl
        if is_training:
            img_indice.append(random.randint(start, end - 1))
        else:
            img_indice.append(start)
        start = end

    # generate
    img_selected_list = []
    for idx in range(channels):
        img_path = os.path.join(fold_path_abs, img_list[img_indice[idx]])
        img_selected_list.append(img_path)
    img_selected_list.sort()

    # compression to (256,256,3*16)
    combine = np.asarray(Image.open(img_selected_list[0]))
    for idx, img in enumerate(img_selected_list):
        if idx == 0:
            continue
        img_content = np.asarray(Image.open(img))
        combine = np.dstack((combine, img_content))

    return combine


def compress_pair_multi_imgs_to_one(img_fold, flow_fold, channels=16, is_training=True):
    """
        assemble images into pair sequence data
    """
    img_fold_path_abs = str(img_fold, encoding='utf-8')
    flow_fold_path_abs = str(flow_fold, encoding='utf-8')

    img_list = [fs for fs in os.listdir(img_fold_path_abs) if len(fs.split('.jpg')) > 1]
    flow_list = [fs for fs in os.listdir(flow_fold_path_abs) if len(fs.split('.jpg')) > 1]

    # pay attention, please keep the image and flow images
    #   is same number(frame id) in same people in a folder
    invl = math.floor(len(img_list) / float(channels))
    start = 0
    indice = []

    # for trainset, random to choose None
    # for testset, choose fixed point
    for _ in range(channels):
        end = start + invl
        if is_training:
            indice.append(random.randint(start, end - 1))
        else:
            indice.append(start)
        start = end

    # acquire actual image path according to indice
    img_selected_list = []
    flow_selected_list = []
    for idx in range(channels):
        img_path = os.path.join(img_fold_path_abs, img_list[indice[idx]])
        flow_path = os.path.join(flow_fold_path_abs, flow_list[indice[idx]])

        img_selected_list.append(img_path)
        flow_selected_list.append(flow_path)

    img_selected_list.sort()
    flow_selected_list.sort()

    # combine channels into one image
    combine_img = np.asarray(Image.open(img_selected_list[0]))
    combine_flow = np.asarray(Image.open(flow_selected_list[0]))
    for idx in range(channels):
        if idx == 0:
            continue
        img_content = np.asarray(Image.open(img_selected_list[idx]))
        flow_content = np.asarray(Image.open(flow_selected_list[idx]))
        combine_img = np.dstack((combine_img, img_content))
        combine_flow = np.dstack((combine_flow, flow_content))

    return combine_img, combine_flow
