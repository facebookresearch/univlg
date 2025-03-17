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
import pickle
import torch.nn.functional as F
from transformers import AutoModel
from pathlib import Path
import hashlib

import ipdb
st = ipdb.set_trace

from collections import defaultdict
from typing import Optional
import pandas as pd
import os

from detectron2.data import DatasetCatalog, MetadataCatalog


split = ['train', 'validation']

ROOT_DIR = os.environ['ROOT_DIR']

MASK3D_processed = f'{ROOT_DIR}/mask3d_processed/scannet'
FRAME_DIR = f'{ROOT_DIR}/SEMSEG_100k/frames_square_highres'
REFER_IT_3D = f'{ROOT_DIR}/refer_it_3d'

_PREDEFINED_SPLITS_REF = {
    "sr3d_ref_scannet_train_single": ("sr3d_train.csv"),
    "sr3d_ref_scannet_val_single": ("sr3d_test.csv"),
    "sr3d_ref_scannet_train_eval_single": ("sr3d_train_eval.csv"),
    "sr3d_ref_scannet_debug_single": ("sr3d_debug.csv"),

    "nr3d_ref_scannet_train_single": ("nr3d_train_filtered.csv"),
    "nr3d_ref_scannet_val_single": ("nr3d_val_filtered.csv"),
    "nr3d_ref_scannet_train_eval_single": ("nr3d_train_eval_filtered.csv"),
    "nr3d_ref_scannet_debug_single": ("nr3d_debug_filtered.csv"),

    "nr3d_ref_scannet_anchor_train_single": ("ScanEnts3D_Nr3D_train.csv"),
    "nr3d_ref_scannet_anchor_val_single": ("ScanEnts3D_Nr3D_val.csv"),
    "nr3d_ref_scannet_anchor_train_eval_single": ("ScanEnts3D_Nr3D_train_eval.csv"),
    "nr3d_ref_scannet_anchor_debug_single": ("ScanEnts3D_Nr3D_debug.csv"),

    "scanrefer_scannet_anchor_train_single": ("ScanRefer_filtered_train_ScanEnts3D_train.csv"),
    "scanrefer_scannet_anchor_val_single": ("ScanRefer_filtered_val_ScanEnts3D_val.csv"),
    "scanrefer_scannet_anchor_train_eval_single": ("ScanRefer_filtered_train_ScanEnts3D_train_eval.csv"),
    "scanrefer_scannet_anchor_debug_single": ("ScanRefer_filtered_train_ScanEnts3D_debug.csv"),
}

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

def generate_object_id_frame_map():
    count = 0
    scene_object_frame_map = {}
    for scene in tqdm(os.listdir(FRAME_DIR)):
        scene_id = scene.split('scene')[1]

        if scene_id not in data:
            continue

        fnames = sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))
        scene_object_frame_map[scene_id] = fnames

    pickle.dump(scene_object_frame_map, open(f'{REFER_IT_3D}/scannet_object_id_frame_map_filenames.pkl', "wb"))
    
    
def normalize_caption(caption: str):
    return caption.lower().replace(".", "").replace(",", "").replace(" ", "")

def generate_relevant_frames_clip():
    model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for scene in tqdm(os.listdir(FRAME_DIR)):
        scene_id = scene.split('scene')[1]
        if scene_id not in data:
            continue

        try:
            frame_paths = [os.path.join(FRAME_DIR, scene, 'color', frame) for frame in sorted(os.listdir(os.path.join(FRAME_DIR, scene, 'color')))]
            images = np.stack([np.array(Image.open(frame_path)) for frame_path in frame_paths])
            images = torch.from_numpy(images).cuda() / 255
            images = images.permute(0, 3, 1, 2)
            
            processed_inputs = preprocess(images)
            embeddings = model.get_image_features(processed_inputs)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            scene_object_frame_map[scene] = embeddings.cpu().to(torch.float32)
        except Exception as e:
            print(f'Error processing {scene}: {e}, skipping')

        print(f'Processed {scene}')

    torch.save(scene_object_frame_map, f'{REFER_IT_3D}/scannet_object_id_frame_map_clip.pkl')
    
    save_path = f"{REFER_IT_3D}/scannet_object_id_frame_map_clip_text.pkl"
    all_utterances = []
    for key, csv_file in _PREDEFINED_SPLITS_REF.items():
        print(key)
        csv_file = Path(REFER_IT_3D) / csv_file
        if csv_file.exists():
            sr3d_data = pd.read_csv(csv_file)
            if "utterance" in sr3d_data.columns:
                utterances = sr3d_data['utterance'].tolist()
            elif "description" in sr3d_data.columns:
                utterances = sr3d_data['description'].tolist()
            else:
                print(f"utterance or description not found in {csv_file}")
                breakpoint()
            all_utterances.extend(utterances)

    all_utterances = [normalize_caption(x) for x in all_utterances]
    embeddings = model.encode_text(all_utterances)
    hashes = {hashlib.md5(utterance.encode()).hexdigest():i for i, utterance in enumerate(all_utterances)}
    torch.save(dict(embeddings=embeddings, hashes=hashes), save_path)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    generate_object_id_frame_map()
    generate_relevant_frames_clip()