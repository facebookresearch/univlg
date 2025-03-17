# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# TODO: Make path generic.
DATA_DIR = '/path/to/SEMSEG_100k/matterport_frames'
SPLITS_PATH = 'splits/m3d_splits'
PC_DATA_DIR = "/path/to/SEMSEG_100k/matterport3d_meshes/matterport"
PC_PROCESSED_PATH = "/path/to/mask3d_processed/matterport"

SPLITS = {
    'train':   f'{SPLITS_PATH}/m3d_train.txt',
    'val':     f'{SPLITS_PATH}/m3d_val.txt',
    'two_scene': f'{SPLITS_PATH}/two_scene.txt',
    'ten_scene': f'{SPLITS_PATH}/ten_scene.txt'
}
