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


def load_sqa3d(questions_json_file, annotations_json_file):
    with open(questions_json_file, "r") as questions_json:
        questions_data = json.load(questions_json)["questions"]
    with open(annotations_json_file, "r") as annotations_json:
        annotations_data = json.load(annotations_json)["annotations"]

    sqa3d_data = []
    assert len(questions_data) == len(annotations_data)
    for i in range(len(questions_data)):
        assert questions_data[i]["question_id"] == annotations_data[i]["question_id"]
        sqa3d_data.append({**questions_data[i], **annotations_data[i]})

    scannet_split = (
        "scannet_context_instance_train_20cls_single_100k"
        if "train" in questions_json_file
        else "scannet_context_instance_val_20cls_single_100k"
    )

    scannet_scenes = DatasetCatalog.get(scannet_split)

    scene_name_to_list_id = {}
    for i in range(len(scannet_scenes)):
        scene_name_to_list_id[scannet_scenes[i]["image_id"]] = i

    return sqa3d_data, scannet_scenes, scene_name_to_list_id


def get_sqa3d_meta(name_map):
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


def register_sqa3d(root):
    assert "scannet_context_instance_train_20cls_single_100k" in DatasetCatalog
    assert "scannet_context_instance_val_20cls_single_100k" in DatasetCatalog

    train_questions_json_file = os.path.join(
        root, "balanced/v1_balanced_questions_train_scannetv2.json"
    )
    train_annotations_json_file = os.path.join(
        root, "balanced/v1_balanced_sqa_annotations_train_scannetv2.json"
    )
    val_questions_json_file = os.path.join(
        root, "balanced/v1_balanced_questions_val_scannetv2.json"
    )
    val_annotations_json_file = os.path.join(
        root, "balanced/v1_balanced_sqa_annotations_val_scannetv2.json"
    )
    test_questions_json_file = os.path.join(
        root, "balanced/v1_balanced_questions_test_scannetv2.json"
    )
    test_annotations_json_file = os.path.join(
        root, "balanced/v1_balanced_sqa_annotations_test_scannetv2.json"
    )

    DatasetCatalog.register(
        "sqa3d_train",
        lambda: load_sqa3d(train_questions_json_file, train_annotations_json_file),
    )
    DatasetCatalog.register(
        "sqa3d_val",
        lambda: load_sqa3d(val_questions_json_file, val_annotations_json_file),
    )
    DatasetCatalog.register(
        "sqa3d_test",
        lambda: load_sqa3d(test_questions_json_file, test_annotations_json_file),
    )

    sqa3d_train_meta = get_sqa3d_meta(NAME_MAP20)
    sqa3d_val_meta = get_sqa3d_meta(NAME_MAP20)
    sqa3d_test_meta = get_sqa3d_meta(NAME_MAP20)

    MetadataCatalog.get("sqa3d_train").set(**sqa3d_train_meta)
    MetadataCatalog.get("sqa3d_val").set(**sqa3d_val_meta)
    MetadataCatalog.get("sqa3d_test").set(**sqa3d_test_meta)
