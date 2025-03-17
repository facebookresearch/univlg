# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import ipdb
import numpy as np
import torch
import yaml
from natsort import natsorted as sorted
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
from accelerate import PartialState
from accelerate.utils import gather, gather_object
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

distributed_state = PartialState()
model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
device = torch.device(f'cuda:{distributed_state.process_index}')
model.to(device)
model.eval()
model.requires_grad_(False)
torch.enable_grad(False)
device = 'cuda:0'
model.preprocess = None
preprocess = model.get_preprocess().transform
assert len(preprocess.transforms) == 5
preprocess.transforms = [preprocess.transforms[0], preprocess.transforms[1], preprocess.transforms[4]]

count = 0
scene_object_frame_map = {}
directories = os.listdir(FRAME_DIR)
with distributed_state.split_between_processes(directories) as _directories:
    print(f"Rank {distributed_state.process_index} has {len(_directories)} scenes")
    for scene in tqdm(_directories):
        scene_id = scene.split('scene')[1]
        if scene_id not in data:
            continue
        try:
            frame_paths = [os.path.join(FRAME_DIR, scene, 'color', frame) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))]
            images = np.stack([np.array(Image.open(frame_path)) for frame_path in frame_paths])
            images = torch.from_numpy(images).cuda() / 255
            images = images.permute(0, 3, 1, 2)
            processed_inputs = preprocess(images)
            with torch.inference_mode():
                embeddings = model.get_image_features(processed_inputs)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            scene_object_frame_map[scene] = embeddings.cpu().to(torch.float32)
        except Exception as e:
            print(f'Error processing {scene}: {e}, skipping')

        print(f'Processed {scene}')

scene_object_frame_maps = gather_object([scene_object_frame_map])
del scene_object_frame_map

scene_object_frame_map_combined = {}
for scene_object_frame_map in scene_object_frame_maps:
    scene_object_frame_map_combined.update(scene_object_frame_map)

save_path = REF_DATASET / 'scannet_object_id_frame_map_clip.pth'
torch.save(scene_object_frame_map_combined, save_path)
print(f"Saved to {save_path}")
