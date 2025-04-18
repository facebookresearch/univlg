# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os

import ipdb
st = ipdb.set_trace

DATA_PATH = "/path/to/SEMSEG_100k/matterport_3d"
FRAME_DIR = "/path/to/SEMSEG_100k/matterport_frames"

train = os.listdir(os.path.join(DATA_PATH, "train"))
train = [x.split('.')[0] for x in train]

train = [t for t in train if t in os.listdir(FRAME_DIR)]
val = os.listdir(os.path.join(DATA_PATH, "val"))
val = [x.split('.')[0] for x in val]

val = [t for t in val if t in os.listdir(FRAME_DIR)]

train_eval = np.random.choice(train, 10, replace=False)
debug_split = np.random.choice(train_eval, 2, replace=False)

data_path = 'm3d_splits'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# write all the splits to a file
train_file_path = f'{data_path}/m3d_train.txt'
val_file_path = f'{data_path}/m3d_val.txt'
debug_file_path = f'{data_path}/two_scene.txt'
train_eval_file_path = f'{data_path}/ten_scene.txt'

with open(train_file_path, 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open(val_file_path, 'w') as f:
    for item in val:
        f.write("%s\n" % item)

with open(debug_file_path, 'w') as f:
    for item in debug_split:
        f.write("%s\n" % item)

with open(train_eval_file_path, 'w') as f:
    for item in train_eval:
        f.write("%s\n" % item)

