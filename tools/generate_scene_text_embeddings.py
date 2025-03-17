# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from pathlib import Path
import json

import ipdb
import pandas as pd
import torch
from transformers import AutoModel
from accelerate import PartialState
from accelerate.utils import gather, gather_object

st = ipdb.set_trace

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

    'scanqa_ref_scannet_train_single': ('ScanQA_v1.0_train.json'),
    'scanqa_ref_scannet_val_single': ('ScanQA_v1.0_val.json'),
    'scanqa_ref_scannet_test_single': ('ScanQA_v1.0_test_w_obj.json'),
    'scanqa_ref_scannet_debug_single': ('ScanQA_v1.0_debug_train_2.json'),
    'scanqa_ref_scannet_debug_test_single': ('ScanQA_v1.0_debug_test.json'),
    'scanqa_ref_scannet_train_eval_single': ('ScanQA_v1.0_train_eval.json'),

    'sqa3d_ref_scannet_train_single': ('SQA_train.json'),
    'sqa3d_ref_scannet_val_single': ('SQA_val.json'),
    'sqa3d_ref_scannet_test_single': ('SQA_test.json'),
    'sqa3d_ref_scannet_debug_single': ('SQA_debug.json'),
    'sqa3d_ref_scannet_train_eval_single': ('SQA_train_eval.json'),
}

distributed_state = PartialState()
model = AutoModel.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
device = torch.device(f'cuda:{distributed_state.process_index}')
model.to(device)
model.eval()
model.requires_grad_(False)
torch.enable_grad(False)

def normalize_caption(caption: str):
    return caption.lower().replace(".", "").replace(",", "").replace(" ", "")

REF_DATASET = Path(os.getenv('REF_DATASET'))

all_utterances = []
for key, csv_file in _PREDEFINED_SPLITS_REF.items():
    print(key)
    csv_file = Path(REF_DATASET) / csv_file
    if csv_file.exists():
        if csv_file.suffix == '.json':
            with open(csv_file) as f:
                data = json.load(f)
            utterances = [item['question'] for item in data]
        else:
            sr3d_data = pd.read_csv(csv_file)
            if "utterance" in sr3d_data.columns:
                utterances = sr3d_data['utterance'].tolist()
            elif "description" in sr3d_data.columns:
                utterances = sr3d_data['description'].tolist()
            else:
                print(f"utterance or description not found in {csv_file}")
        all_utterances.extend(utterances)
    else:
        breakpoint()

all_utterances = [normalize_caption(x) for x in all_utterances]
print(f"Found {len(all_utterances)} utterances")
with distributed_state.split_between_processes(all_utterances) as _all_utterances:
    embeddings = model.encode_text(_all_utterances)

print(f"Before gathering, each GPU has {len(embeddings)} embeddings")
embeddings = gather_object(embeddings)
print(f"After gathering, we have {len(embeddings)} embeddings")

hashes = {hashlib.md5(utterance.encode()).hexdigest():i for i, utterance in enumerate(all_utterances)}

if distributed_state.is_main_process:
    save_path = REF_DATASET / 'scannet_object_id_frame_map_clip_text.pth'
    torch.save(dict(embeddings=embeddings, hashes=hashes), save_path)
    print(f"saved to {save_path}")