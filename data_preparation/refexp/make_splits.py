# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pandas as pd
import os

SPLIT_MAIN_DIR = "splits/scannet_splits"
SPLITS = {
    "train": f"{SPLIT_MAIN_DIR}/scannetv2_train.txt",
    "val": f"{SPLIT_MAIN_DIR}/scannetv2_val.txt",
    "debug": f"{SPLIT_MAIN_DIR}/two_scene.txt",
    "train_eval": f"{SPLIT_MAIN_DIR}/ten_scene.txt",
}

ROOT_DIR = os.environ["REF_DATASET"]

def filter_json(annos):
    annos = [
        {
            'scan_id': anno['scene_id'],
            'tokens': anno['token'],
            'target_id': anno['object_id'],
            'dataset': 'scanrefer',
            **anno,
        }
        for i, anno in enumerate(annos)
        if anno.get('scene_id', "") != ""
    ]
    return annos


def load_and_filter_json(ann_file):
    with open(ann_file, "r") as f:
        ann = json.load(f)
        ann = filter_json(ann)
    ann = pd.DataFrame(ann)
    return ann


def make_splits_scanrefer(scanrefer_ann_file_train, scanrefer_ann_file_val):
    
    ann = load_and_filter_json(scanrefer_ann_file_train)

    # save as csv
    ann["dataset"] = "scanrefer"
    ann_name = scanrefer_ann_file_train.split("/")[-1].split(".")[0]
    ann.to_csv(f"{ROOT_DIR}/{ann_name}_train.csv", index=False)

    splits = ["debug", "train_eval"]
    for split_name in splits:
        split = SPLITS[split_name]
        with open(split, "r") as f:
            split = f.read().splitlines()

        ann_split = ann[ann["scene_id"].isin(split)]
        ann_name = scanrefer_ann_file_train.split("/")[-1].split(".")[0]

        # save as csv
        ann_split = pd.DataFrame(ann_split)
        ann_split["dataset"] = "scanrefer"
        ann_split.to_csv(f"{ROOT_DIR}/{ann_name}_{split_name}.csv", index=False)

    ann = load_and_filter_json(scanrefer_ann_file_val)
    
    ann["dataset"] = "scanrefer"
    ann_name = scanrefer_ann_file_val.split("/")[-1].split(".")[0]
    ann.to_csv(f"{ROOT_DIR}/{ann_name}_val.csv", index=False)
    

def make_splits_referit3d(ann_file, split_name):
    split = SPLITS[split_name]
    with open(split, "r") as f:
        split = f.read().splitlines()

    # check if file is json or csv
    if ann_file.endswith(".csv"):
        ann = pd.read_csv(ann_file).to_dict(orient="records")
    else:
        with open(ann_file, "r") as f:
            ann = json.load(f)

    ann_split = [row for row in ann if row["scan_id"] in split]

    ann_name = ann_file.split("/")[-1].split(".")[0]

    # save as csv
    ann_split = pd.DataFrame(ann_split)
    ann_split.to_csv(f"{ROOT_DIR}/{ann_name}_{split_name}.csv", index=False)
    
    
if __name__ == "__main__":
    # scanrefer
    scanrefer_ann_file_train = f"{ROOT_DIR}/ScanRefer_filtered_train_ScanEnts3D.json"
    scanrefer_ann_file_val = f"{ROOT_DIR}/ScanRefer_filtered_val_ScanEnts3D.json"
    make_splits_scanrefer(scanrefer_ann_file_train, scanrefer_ann_file_val)
    print("Done with ScanRefer!")
    
    # for referit3d dataset
    datasets = ["sr3d", "ScanEnts3D_Nr3D"]
    splits = ["debug", "train", "val", "train_eval"]
    for dataset in datasets:
        for split in splits:
            ann_file = f"{ROOT_DIR}/{dataset}.csv"
            make_splits_referit3d(ann_file, split)
            
    print("Done with Referit3D!")
    
    