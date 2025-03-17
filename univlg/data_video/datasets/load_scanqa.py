# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import ipdb
from detectron2.data import DatasetCatalog, MetadataCatalog
from univlg.global_vars import NAME_MAP20

st = ipdb.set_trace


def load_scanqa(json_file):
    with open(json_file, "r") as scanqa_json:
        scanqa_data = json.load(scanqa_json)

    scannet_split = (
        "scannet_context_instance_train_20cls_single_100k"
        if "train" in json_file
        else "scannet_context_instance_val_20cls_single_100k"
    )

    scannet_scenes = DatasetCatalog.get(scannet_split)

    scene_name_to_list_id = {}
    for i in range(len(scannet_scenes)):
        scene_name_to_list_id[scannet_scenes[i]["image_id"]] = i

    return scanqa_data, scannet_scenes, scene_name_to_list_id


def get_scanqa_meta(name_map):
    dataset_categories = [
        {"id": key, "name": item, "supercategory": "nyu40"}
        for key, item in name_map.items()
    ]

    thing_ids = [k["id"] for k in dataset_categories]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in dataset_categories]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_scanqa(root):
    assert "scannet_context_instance_train_20cls_single_100k" in DatasetCatalog
    assert "scannet_context_instance_val_20cls_single_100k" in DatasetCatalog

    train_json_file = os.path.join(root, "ScanQA_v1.0_train.json")
    val_json_file = os.path.join(root, "ScanQA_v1.0_val.json")
    test_json_file = os.path.join(root, "ScanQA_v1.0_test_w_obj.json")

    DatasetCatalog.register("scanqa_train", lambda: load_scanqa(train_json_file))
    DatasetCatalog.register("scanqa_val", lambda: load_scanqa(val_json_file))
    DatasetCatalog.register("scanqa_test", lambda: load_scanqa(test_json_file))

    scanqa_train_meta = get_scanqa_meta(NAME_MAP20)
    scanqa_val_meta = get_scanqa_meta(NAME_MAP20)
    scanqa_test_meta = get_scanqa_meta(NAME_MAP20)

    MetadataCatalog.get("scanqa_train").set(**scanqa_train_meta)
    MetadataCatalog.get("scanqa_val").set(**scanqa_val_meta)
    MetadataCatalog.get("scanqa_test").set(**scanqa_test_meta)
