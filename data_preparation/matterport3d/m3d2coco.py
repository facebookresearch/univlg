# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import json

from PIL import Image
from pycococreatortools import pycococreatortools
import pycocotools.mask as mask_util

from univlg.global_vars import MATTERPORT_NAME_MAP
from data_preparation.matterport3d.global_dirs import DATA_DIR, SPLITS

import ipdb
st = ipdb.set_trace


INFO = {
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40'}
    for key, item in MATTERPORT_NAME_MAP.items()
]


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def polygons_to_bitmask(polygons, height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


def convert_scannet_to_coco(path, phase):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "depths": [],
        "poses": [],
        "intrinsics": [],
    }

    # get list
    split = SPLITS[phase]
    scene_ids = read_txt(split)
    image_ids = []
    for scene_id in scene_ids:
        for image_id in os.listdir(os.path.join(path, scene_id, 'color')):
            image_ids.append(os.path.join(scene_id, image_id.split('.')[0]))
    print("images number in {}: {}".format(path, len(image_ids)))

    coco_image_id = 1
    for index in range(len(image_ids)):
        print("{}/{}".format(index, len(image_ids)), end='\r')

        scene_id = image_ids[index].split('/')[0]
        image_id = image_ids[index].split('/')[1]
        image_size = Image.open(os.path.join(path, scene_id, 'color', image_id + '.jpg')).size 

        image_filename = os.path.join(scene_id, 'color', image_id + '.jpg')
        image_info = pycococreatortools.create_image_info(
            coco_image_id, image_filename, image_size)
        coco_output['images'].append(image_info)

        depth_filename = os.path.join(scene_id, 'depth', image_id + '.png')
        depth_info = pycococreatortools.create_image_info(
            coco_image_id, depth_filename, image_size)
        coco_output['depths'].append(depth_info)

        pose_filename = os.path.join(scene_id, 'pose', image_id + '.txt')
        pose_info = pycococreatortools.create_image_info(
            coco_image_id, pose_filename, image_size)
        coco_output['poses'].append(pose_info)

        intrinsic_filename = os.path.join(scene_id, 'intrinsic', image_id + '.txt')
        pose_info = pycococreatortools.create_image_info(
            coco_image_id, intrinsic_filename, image_size)
        coco_output['intrinsics'].append(pose_info)

        coco_image_id += 1

    parent_dir = os.path.dirname(path)
    json.dump(
        coco_output, open(f'{parent_dir}/m3d_{phase}.coco.json', 'w'))


if __name__ == '__main__':
    phases = ['train', 'val', 'ten_scene', 'two_scene']
    for phase in phases:
        convert_scannet_to_coco(DATA_DIR, phase)
