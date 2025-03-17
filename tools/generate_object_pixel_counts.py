# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from natsort import natsorted as sorted
import yaml
import numpy as np
import torch
from pytorch3d.ops import knn_points
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from univlg.modeling.backproject.backproject import backprojector
import pickle
import torch.nn.functional as F
import ipdb
from pathlib import Path

st = ipdb.set_trace

split = ['train', 'validation']

UNIVLG_DATA_PATH = Path(os.getenv('UNIVLG_DATA_PATH'))
REF_DATASET = Path(os.getenv('REF_DATASET'))
MASK3D_processed = UNIVLG_DATA_PATH / 'mask3d_processed' / 'scannet200'
FRAME_DIR = UNIVLG_DATA_PATH / 'SEMSEG_100k' / 'frames_square_highres'

data = {}
total = 0
for s in split:
    with open(os.path.join(MASK3D_processed, f'{s}_database.yaml')) as f:
        data_ = yaml.load(f, Loader=yaml.FullLoader)
        # data_ is a list of dicts, with a key 'filepath'
        # eg: '/path/to/mask3d_processed/scannet/train/0000_00.npy'
        # we need to get 0000_00 and use it as the key in data
        for d in data_:
            key = d['filepath'].split('/')[-1].split('.')[0]
            data[key] = d
            total += 1

print(f'Loaded {total} scenes')

def get_2d_labels(rgb_coords, mask3d_coords, labels, segments, depths, poses, fnames):
    N, H, W, _ = rgb_coords.shape

    # unproject our pc
    rgb_coords = backprojector([rgb_coords.permute(0, 3, 1, 2).shape], depths[None].float(), poses[None].float())[0][0]

    rgb_coords = F.interpolate(
        rgb_coords[0].permute(0, 3, 1, 2), scale_factor=0.25, mode="nearest"
    ).permute(0, 2, 3, 1)[None]
    H, W = H // 4, W // 4

    rgb_coords = rgb_coords.reshape(1, -1, 3)

    # do 1 NN with mask3d points
    dists_idxs_ = knn_points(mask3d_coords[None], rgb_coords)

    valids = torch.zeros_like(rgb_coords[:, :, 0], dtype=torch.bool)
    valids[:, dists_idxs_.idx.squeeze(-1)[dists_idxs_.dists.squeeze(-1) < 1e-3]] = True

    rgb_coord_labels = torch.ones_like(rgb_coords[:, :, 0], dtype=torch.float32) * -1
    labels = torch.from_numpy(labels).cuda()
    rgb_coord_labels[:, dists_idxs_.idx.squeeze(-1)[dists_idxs_.dists.squeeze(-1) < 1e-3]] = labels[:, 1][(dists_idxs_.dists.squeeze(-1) < 1e-3).squeeze()]
    valids = valids.reshape(N, H, W)

    rgb_coord_labels = rgb_coord_labels.reshape(N, H, W)

    return rgb_coord_labels.cpu().numpy(), valids.cpu().numpy()


count = 0
scene_object_frame_map = {}
for scene in tqdm(os.listdir(FRAME_DIR)):
    scene_id = scene.split('scene')[1]

    if scene_id not in data:
        continue

    if scene_id != '0309_01':
        continue

    # load labels from data
    points = np.load(data[scene_id]['filepath'], allow_pickle=True)
    coordinates, mask3d_colors, _, segments, labels = (
        points[:, :3],
        points[:, 3:6],
        points[:, 6:9],
        points[:, 9],
        points[:, 10:12],
    )

    coordinates = torch.from_numpy(coordinates).cuda()
    segments = torch.from_numpy(segments).cuda()

    if labels.shape[0] != coordinates.shape[0]:
        print(f'{labels.shape[0]} != {coordinates.shape[0]} for {scene_id}')
        print(scene_id)
        continue

    fnames = sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))

    images = np.stack(
        [np.array(Image.open(os.path.join(FRAME_DIR, scene, 'color', frame))) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))])
    images = torch.from_numpy(images).cuda().float()

    depths = np.stack(
        [np.array(Image.open(os.path.join(FRAME_DIR, scene, 'depth_inpainted', frame))) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'depth')))])
    depths = torch.from_numpy(depths).cuda().float() / 1000.0

    poses = np.stack(
        [np.loadtxt(os.path.join(FRAME_DIR, scene, 'pose', frame)) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'pose')))]
    )

    poses = torch.from_numpy(poses).cuda().float()

    mask = torch.isinf(poses).any([1, 2])

    depths = depths[~mask]
    images = images[~mask]
    poses = poses[~mask]
    fnames = [fname for i, fname in enumerate(fnames) if not mask[i]]

    # images is [N, H, W, 3]
    labels_2d, valids_2d = get_2d_labels(images, coordinates, labels, segments, depths, poses, fnames)
    tensor = torch.from_numpy(labels_2d).long()
    tensor = rearrange(tensor, 'n h w -> n (h w)')
    unique_vals = torch.unique(tensor)
    unique_vals = unique_vals[unique_vals != -1]
    object_frame_map = {}
    shuffle = True
    for val in unique_vals:
        rows, _ = torch.where(tensor == val)
        indices, counts = torch.unique(rows, return_counts=True)
        counts, sorting_indices = torch.sort(counts, descending=True)
        indices = indices[sorting_indices]
        object_frame_map[val.item()] = torch.stack([indices, counts])

    scene_object_frame_map[scene] = object_frame_map
    print(f'Processed {scene}')

output_file = REF_DATASET / 'scannet_object_id_frame_map_fixed.pth'
torch.save(scene_object_frame_map, output_file)
print(f"Saved to {output_file}")

count = 0
scene_object_frame_map = {}
for scene in tqdm(os.listdir(FRAME_DIR)):
    scene_id = scene.split('scene')[1]

    if scene_id not in data:
        continue

    fnames = sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))
    scene_object_frame_map[scene_id] = fnames

filenames_output_file = REF_DATASET / 'scannet_object_id_frame_map_filenames.pth'
torch.save(scene_object_frame_map, filenames_output_file)
print(f"Saved to {filenames_output_file}")