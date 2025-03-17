# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
import json
from pathlib import Path
import ipdb
from typing import List, NamedTuple, Tuple

from detectron2.data import DatasetCatalog
# from .ref_coco_utils import get_root_and_nouns, consolidate_spans
_PREDEFINED_SPLITS_REF = {
    "sr3d_ref_scannet_train_single": ("sr3d_train.csv"),
    "sr3d_ref_scannet_val_single": ("sr3d_test.csv"),
}


st = ipdb.set_trace


def get_data(dataset_path, dataset_name, split, subsample=None):
    with open(dataset_path / dataset_name, "rb") as f:
        data = pickle.load(f)

    # print(dataset_name, dataset_path / dataset_name, data[0])

    with open(dataset_path / "coco" / "annotations" / "instances_train2017.json", "r") as f:
        coco_annotations = json.load(f)

    coco_anns = coco_annotations["annotations"]
    annid2cocoann = {item["id"]: item for item in coco_anns}

    data = [item for item in data if item["split"] == split]

    if subsample is not None:
        random.seed(42)
        data = random.sample(data, subsample)

    all_data = []

    for item in data:
        assert item["split"] == split
        for s in item["sentences"]:
            caption = s["sent"]
            # ann_id = s["ann_id"]
            sent_id = s["sent_id"]
            image_id = item["image_id"]
            original_id = item["ann_id"]
            target_bbox = annid2cocoann[item["ann_id"]]["bbox"] # [x, y, w, h]

            target_segmentation = annid2cocoann[item["ann_id"]]["segmentation"]

            all_data.append(
                {
                    "image_id": image_id,
                    "dataset_name": dataset_name,
                    "original_id": original_id,
                    "caption": caption,
                    "sent_id": sent_id,
                    "coco_path": dataset_path / "coco",
                    "target_bbox": target_bbox,
                    "target_segmentation": target_segmentation,
                }
            )

    return all_data

def register_coco_ref(root):
    dataset_path = Path(root)
    for dataset_name in ["refcoco/refs(unc).p", "refcoco+/refs(unc).p", "refcocog/refs(umd).p"]:
        _name = dataset_name.split("/")[0]
        print(f"Registering {dataset_name}, {_name}")
        DatasetCatalog.register(f"{_name}_debug", lambda dataset_name=dataset_name: get_data(dataset_path, dataset_name, "train", subsample=10))
        DatasetCatalog.register(f"{_name}_train_eval", lambda dataset_name=dataset_name: get_data(dataset_path, dataset_name, "train", subsample=100))
        DatasetCatalog.register(f"{_name}_train", lambda dataset_name=dataset_name: get_data(dataset_path, dataset_name, "train"))
        DatasetCatalog.register(f"{_name}_val", lambda dataset_name=dataset_name: get_data(dataset_path, dataset_name, "val"))