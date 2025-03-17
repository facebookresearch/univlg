# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Optional
import pandas as pd
import os
import ast
import json
import copy
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog
from univlg.global_vars import NAME_MAP20

import ipdb
st = ipdb.set_trace


_PREDEFINED_SPLITS_REF = {
    "sr3d_ref_scannet_train_single": ("sr3d_train.csv"),
    "sr3d_ref_scannet_val_single": ("sr3d_test.csv"),
    "sr3d_ref_scannet_train_eval_single": ("sr3d_train_eval.csv"),
    "sr3d_ref_scannet_debug_single": ("sr3d_debug.csv"),

    "nr3d_ref_scannet_anchor_train_single": ("ScanEnts3D_Nr3D_train.csv"),
    "nr3d_ref_scannet_anchor_val_single": ("ScanEnts3D_Nr3D_val.csv"),
    "nr3d_ref_scannet_anchor_train_eval_single": ("ScanEnts3D_Nr3D_train_eval.csv"),
    "nr3d_ref_scannet_anchor_debug_single": ("ScanEnts3D_Nr3D_debug.csv"),

    "scanrefer_scannet_anchor_train_single": ("ScanRefer_filtered_train_ScanEnts3D_train.csv"),
    "scanrefer_scannet_anchor_val_single": ("ScanRefer_filtered_val_ScanEnts3D_val.csv"),
    "scanrefer_scannet_anchor_train_eval_single": ("ScanRefer_filtered_train_ScanEnts3D_train_eval.csv"),
    "scanrefer_scannet_anchor_debug_single": ("ScanRefer_filtered_train_ScanEnts3D_debug.csv"),
    "scanrefer_scannet_test_single": ("ScanRefer_filtered_test.csv"),

    'scanqa_ref_scannet_train_single': ('ScanQA_v1.0_train.json'),
    'scanqa_ref_scannet_val_single': ('ScanQA_v1.0_val.json'),
    'scanqa_ref_scannet_test_single': ('ScanQA_v1.0_test_w_obj.json'),

    'sqa3d_ref_scannet_train_single': ('SQA_train.json'),
    'sqa3d_ref_scannet_val_single': ('SQA_val.json'),
    'sqa3d_ref_scannet_test_single': ('SQA_test.json'),
    'sqa3d_ref_scannet_debug_single': ('SQA_debug.json'),
    'sqa3d_ref_scannet_train_eval_single': ('SQA_train_eval.json'),
}

# ignore_scene_dict = {"scene0585_01", "scene0181_02", "scene0181_02", "scene0284_00"}

def load_ref(
        csv_file: str,
        return_scene_with_batched_captions: bool = False,
        return_scene_batch_size: int = 2,
        subsample_scenes: Optional[int] = None,
    ):
    if 'csv' in csv_file:
        orig_sr3d_data = pd.read_csv(csv_file).to_dict(orient='records')

        sr3d_data = []
        removed_count = 0
        for i, line in enumerate(orig_sr3d_data):
            if 'anchor_ids' in line:
                if type(line['anchor_ids']) is str:
                    anchor_ids = ast.literal_eval(line['anchor_ids'])
                else:
                    anchor_ids = line['anchor_ids']
                line['anchor_ids'] = anchor_ids

            if 'anchors_types' in line:
                if type(line['anchors_types']) is str:
                    anchors_types = ast.literal_eval(line['anchors_types'])
                else:
                    anchors_types = line['anchors_types']

                line['anchors_types'] = anchors_types

            if 'scan_id' in line and line.get('mentions_target_class', True) is True:
                if "nr3d" in csv_file:
                    if str(line['correct_guess']).lower() == 'true':
                        sr3d_data.append(line)
                    else:
                        removed_count += 1
                else:
                    sr3d_data.append(line)
            else:
                removed_count += 1
        print(f"Removed {removed_count} scenes from {csv_file}")
    elif 'json' in csv_file:
        with open(csv_file, 'r') as f:
            sr3d_data = json.load(f)
            print(f"Loaded {len(sr3d_data)} scenes from {csv_file}. JSON mode.")
        if ('train' in csv_file.lower() or 'debug' in csv_file.lower()) and ('sqa' in csv_file.lower()):
            print("Loading alternative situations!")
            ref_dataset = Path(os.environ['REF_DATASET'])
            sqa_json_path = ref_dataset / 'v1_balanced_questions_train_scannetv2.json'
            if not sqa_json_path.exists():
                raise FileNotFoundError(f"File not found: {sqa_json_path}")
            with open(sqa_json_path, 'r') as f:
                more_data = json.load(f)
            alternative_situations = {}
            for x in more_data['questions']:
                alternative_situations[x['question_id']] = x['alternative_situation']
            new_data = []
            for x in sr3d_data:
                for new_situation in alternative_situations[x['question_id']]:
                    _x = copy.deepcopy(x)
                    _x['situation'] = new_situation
                    new_data.append(_x)
            sr3d_data += new_data

    if 'train' in csv_file or 'debug' in csv_file:
        scannet_split = 'scannet200_context_instance_train_200cls_single_highres_100k'
    elif 'test' in csv_file and 'scanrefer' in csv_file.lower():
        scannet_split = 'scannet200_context_instance_test_200cls_single_highres_100k'
    else:
        scannet_split = 'scannet200_context_instance_val_200cls_single_highres_100k'


    scannet_scenes = DatasetCatalog.get(scannet_split)

    if subsample_scenes is not None:
        import random
        random.seed(0)
        scannet_scenes = random.sample(scannet_scenes, subsample_scenes)

    scene_name_to_list_id = {}
    for i in range(len(scannet_scenes)):
        scene_name_to_list_id[scannet_scenes[i]['image_id']] = i

    if return_scene_with_batched_captions:
        tmp_sr3d_data = defaultdict(list)
        cleared_lists = []
        for sr3d_instance in sr3d_data:
            if sr3d_instance['scan_id'] not in scene_name_to_list_id:
                # assert False
                continue
            if len(tmp_sr3d_data[sr3d_instance['scan_id']]) >= return_scene_batch_size:
                # We fill up the entry in the dict until it reaches our desired batch size
                cleared_lists.append(tmp_sr3d_data[sr3d_instance['scan_id']])
                tmp_sr3d_data[sr3d_instance['scan_id']] = []

            tmp_sr3d_data[sr3d_instance['scan_id']].append(sr3d_instance)
        sr3d_data = list(tmp_sr3d_data.values()) + cleared_lists

    return sr3d_data, scannet_scenes, scene_name_to_list_id


def get_sr3d_meta(name_map):
    dataset_categories = [
        {'id': key, 'name': item, 'supercategory': 'nyu40'} for key, item in name_map.items()
    ]

    thing_ids = [k["id"] for k in dataset_categories]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in dataset_categories]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_ref(root):
    return_scene_batch_size = int(os.getenv('RETURN_SCENE_BATCH_SIZE', 6))
    assert 'scannet200_context_instance_train_200cls_single_highres_100k' in DatasetCatalog
    assert 'scannet200_context_instance_val_200cls_single_highres_100k' in DatasetCatalog
    for key, csv_file in _PREDEFINED_SPLITS_REF.items():
        DatasetCatalog.register(key, lambda csv_file=csv_file: load_ref(os.path.join(root, csv_file)))
        MetadataCatalog.get(key).set(**get_sr3d_meta(NAME_MAP20))

        DatasetCatalog.register(f"{key}_batched", lambda csv_file=csv_file: load_ref(
            os.path.join(root, csv_file), return_scene_with_batched_captions=True, return_scene_batch_size=return_scene_batch_size))

        if 'val' in key:
            name = key.replace('val', 'val_50')
            DatasetCatalog.register(f"{name}_batched", lambda csv_file=csv_file: load_ref(
                os.path.join(root, csv_file),
                return_scene_with_batched_captions=True,
                return_scene_batch_size=return_scene_batch_size,
                subsample_scenes=50
            ))

            name = key.replace('val', 'val_5')
            DatasetCatalog.register(f"{name}_batched", lambda csv_file=csv_file: load_ref(
                os.path.join(root, csv_file),
                return_scene_with_batched_captions=True,
                return_scene_batch_size=return_scene_batch_size,
                subsample_scenes=5
            ))

        if 'test' in key:
            name = key.replace('test', 'test_2')
            DatasetCatalog.register(f"{name}_batched", lambda csv_file=csv_file: load_ref(
                os.path.join(root, csv_file),
                return_scene_with_batched_captions=True,
                return_scene_batch_size=return_scene_batch_size,
                subsample_scenes=5
            ))
