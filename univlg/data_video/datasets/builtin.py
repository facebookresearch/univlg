# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import ipdb

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin import register_all_coco

from .load_scanqa import register_scanqa
from .load_sqa3d import register_sqa3d
from .load_sr3d import register_ref
from .load_coco_ref import register_coco_ref
from .scannet_context import (
    _get_dataset_instances_meta,
    _get_scannet_instances20_meta,
    _get_scannet_instances_meta,
    register_scannet_context_instances,
    register_scannet_context_instances_single,
)

st = ipdb.set_trace


_PREDEFINED_SPLITS_CONTEXT20_SCANNET_SINGLE_100K = {
    "scannet_context_instance_train_20cls_single_100k": (
        "frames_square",
        "scannet20_train.coco.json",
    ),
    "scannet_context_instance_val_20cls_single_100k": (
        "frames_square",
        # "scannet20_two_scene.coco.json"
        # "scannet20_val50.coco.json"
        "scannet20_val.coco.json",
    ),
    "scannet_context_instance_trainval_20cls_single_100k": (
        "frames_square",
        "scannet20_trainval.coco.json",
    ),
    "scannet_context_instance_train_eval_20cls_single_100k": (
        "frames_square",
        "scannet20_ten_scene.coco.json",
    ),
    "scannet_context_instance_test_20cls_single_100k": (
        "frames_square",
        "scannet20_test.coco.json",
    ),
    "scannet_context_instance_debug_20cls_single_100k": (
        "frames_square",
        "scannet20_two_scene.coco.json",
    ),
    "scannet_context_instance_train_20cls_single_25k": (
        "scannet_frames_25k",
        "scannet_train.coco.json",
    ),
    "scannet_context_instance_val_20cls_single_25k": (
        "scannet_frames_25k",
        "scannet_val.coco.json",
    ),
    "scannet_context_instance_train_eval_20cls_single_25k": (
        "scannet_frames_25k",
        "scannet_ten_scene.coco.json",
    ),
    "scannet_context_instance_train_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_train.coco.json",
    ),
    "scannet_context_instance_val_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_val.coco.json",
    ),
    "scannet_context_instance_train_eval_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_ten_scene.coco.json",
    ),
    "scannet_context_instance_debug_20cls_single_highres_100k": (
        "frames_square_highres",
        "scannet_highres_two_scene.coco.json",
    ),
}

_PREDEFINED_SPLITS_CONTEXT20_SCANNET200_SINGLE_100K = {
    "scannet200_context_instance_train_200cls_single_100k": (
        "frames_square",
        "scannet200_train.coco.json",
    ),
    "scannet200_context_instance_val_200cls_single_100k": (
        "frames_square",
        "scannet200_val.coco.json",
    ),
    "scannet200_context_instance_train_eval_200cls_single_100k": (
        "frames_square",
        "scannet200_ten_scene.coco.json",
    ),
    "scannet200_context_instance_debug_200cls_single_100k": (
        "frames_square",
        "scannet200_two_scene.coco.json",
    ),
    "scannet200_context_instance_train_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_train.coco.json",
    ),
    "scannet200_context_instance_val_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_val.coco.json",
    ),
    "scannet200_context_instance_trainval_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_trainval.coco.json",
    ),
    "scannet200_context_instance_test_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_test.coco.json",
    ),
    "scannet200_context_instance_train_eval_200cls_single_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_ten_scene.coco.json",
    ),
    "scannet200_context_instance_debug_200cls_single_highres_100k": {
        "frames_square_highres",
        "scannet200_highres_two_scene.coco.json",
    },
}

_PREDEFINED_SPLITS_CONTEXT20_SCANNET200_100K = {
    "scannet200_context_instance_train_200cls_100k": (
        "frames_square",
        "scannet200_train.coco.json",
    ),
    "scannet200_context_instance_val_200cls_100k": (
        "frames_square",
        "scannet200_val.coco.json",
    ),
    "scannet200_context_instance_train_eval_200cls_100k": (
        "frames_square",
        "scannet200_ten_scene.coco.json",
    ),
    "scannet200_context_instance_debug_200cls_100k": (
        "frames_square",
        "scannet200_two_scene.coco.json",
    ),
    "scannet200_context_instance_train_200cls_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_train.coco.json",
    ),
    "scannet200_context_instance_val_200cls_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_val.coco.json",
    ),
    "scannet200_context_instance_trainval_200cls_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_trainval.coco.json",
    ),
    "scannet200_context_instance_test_200cls_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_test.coco.json",
    ),
    "scannet200_context_instance_train_eval_200cls_highres_100k": (
        "frames_square_highres",
        "scannet200_highres_ten_scene.coco.json",
    ),
    "scannet200_context_instance_debug_200cls_highres_100k": {
        "frames_square_highres",
        "scannet200_highres_two_scene.coco.json",
    },
}

_PREDEFINED_SPLITS_AI2THOR = {
    "ai2thor_train_single": (
        "ai2thor_frames",
        # "ai2thor_two_scene.coco.json"
        "ai2thor_train.coco.json",
    ),
    "ai2thor_val_single": (
        "ai2thor_frames",
        # "ai2thor_two_scene.coco.json"
        "ai2thor_val.coco.json",
    ),
    "ai2thor_train_eval_single": (
        "ai2thor_frames",
        # "ai2thor_two_scene.coco.json"
        "ai2thor_ten_scene.coco.json",
    ),
    "ai2thor_debug_single": (
        "ai2thor_frames",
        "ai2thor_two_scene.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
    "ai2thor_highres_train_single": (
        "ai2thor_frames_512",
        "ai2thor_train_highres.coco.json"
        # "ai2thor_train.coco.json"
    ),
    "ai2thor_highres_val_single": (
        "ai2thor_frames_512",
        "ai2thor_val_highres.coco.json"
        #  "ai2thor_val.coco.json"
    ),
    "ai2thor_highres_val50_single": (
        "ai2thor_frames_512",
        "ai2thor_val50_highres.coco.json"
        #  "ai2thor_val.coco.json"
    ),
    "ai2thor_highres_train_eval_single": (
        "ai2thor_frames_512",
        "ai2thor_ten_scene_highres.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
    "ai2thor_highres_debug_single": (
        "ai2thor_frames_512",
        "ai2thor_two_scene_highres.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
}

_PREDEFINED_SPLITS_S3DIS = {
    "s3dis_train_single": (
        "s3dis_frames_fixed",
        # "s3dis_two_scene.coco.json"
        "s3dis_area_5_train.coco.json",
    ),
    "s3dis_val_single": (
        "s3dis_frames_fixed",
        # "s3dis_two_scene.coco.json"
        "s3dis_area_5_val.coco.json",
    ),
    "s3dis_train_eval_single": (
        "s3dis_frames_fixed",
        # "s3dis_two_scene.coco.json"
        "s3dis_area_5_ten_scene.coco.json",
    ),
    "s3dis_debug_single": (
        "s3dis_frames_fixed",
        "s3dis_area_5_two_scene.coco.json"
        # "s3dis_ten_scene.coco.json"
    ),
}

_PREDEFINED_SPLITS_MATTERPORT = {
    "matterport_train_single": (
        "matterport_frames",
        # "matterport_two_scene.coco.json"
        "m3d_train.coco.json",
    ),
    "matterport_val_single": (
        "matterport_frames",
        # "matterport_two_scene.coco.json"
        "m3d_val.coco.json",
    ),
    "matterport_train_eval_single": (
        "matterport_frames",
        # "matterport_two_scene.coco.json"
        "m3d_ten_scene.coco.json",
    ),
    "matterport_debug_single": (
        "matterport_frames",
        "m3d_two_scene.coco.json"
        # "matterport_ten_scene.coco.json"
    ),
}


_PREDEFINED_SPLITS_AI2THOR_JOINT = {
    "ai2thor_train": (
        "ai2thor_frames",
        # "ai2thor_two_scene.coco.json"
        "ai2thor_train.coco.json",
    ),
    "ai2thor_val": (
        "ai2thor_frames",
        # "ai2thor_two_scene.coco.json"
        "ai2thor_val.coco.json",
    ),
    "ai2thor_train_eval": (
        "ai2thor_frames",
        # "ai2thor_two_scene.coco.json"
        "ai2thor_ten_scene.coco.json",
    ),
    "ai2thor_debug": (
        "ai2thor_frames",
        "ai2thor_two_scene.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
    "ai2thor_highres_train": (
        "ai2thor_frames_512",
        "ai2thor_train_highres.coco.json"
        # "ai2thor_train.coco.json"
    ),
    "ai2thor_highres_val": (
        "ai2thor_frames_512",
        "ai2thor_val_highres.coco.json"
        #  "ai2thor_val.coco.json"
    ),
    "ai2thor_highres_val50": (
        "ai2thor_frames_512",
        "ai2thor_val50_highres.coco.json"
        #  "ai2thor_val.coco.json"
    ),
    "ai2thor_highres_3d_val50": (
        "ai2thor_frames_512",
        "ai2thor_val50_highres.coco.json"
        #  "ai2thor_val.coco.json"
    ),
    "ai2thor_highres_train_eval": (
        "ai2thor_frames_512",
        "ai2thor_ten_scene_highres.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
    "ai2thor_highres_3d_train_eval": (
        "ai2thor_frames_512",
        "ai2thor_ten_scene_highres.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
    "ai2thor_highres_debug": (
        "ai2thor_frames_512",
        "ai2thor_two_scene_highres.coco.json"
        # "ai2thor_ten_scene.coco.json"
    ),
}


def register_all_scannet_context20_100K(root):
    for key, (
        image_root,
        json_file,
    ) in _PREDEFINED_SPLITS_CONTEXT20_SCANNET_100K.items():
        register_scannet_context_instances(
            key,
            _get_scannet_instances20_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_dataset(root, dataset_name="ai2thor"):
    if dataset_name == "ai2thor":
        split_dict = _PREDEFINED_SPLITS_AI2THOR_JOINT
    elif dataset_name == "alfred":
        split_dict = _PREDEFINED_SPLITS_ALFRED_JOINT
    elif dataset_name == "replica":
        split_dict = _PREDEFINED_SPLITS_REPLICA
    elif dataset_name == "scannet200":
        split_dict = _PREDEFINED_SPLITS_CONTEXT20_SCANNET200_100K
    else:
        raise NotImplementedError("dataset_name {} not supported".format(dataset_name))
    for key, (image_root, json_file) in split_dict.items():
        register_scannet_context_instances(
            key,
            _get_dataset_instances_meta(dataset=dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_scannet_context20_scannet_single(root):
    for key, (
        image_root,
        json_file,
    ) in _PREDEFINED_SPLITS_CONTEXT20_SCANNET_SINGLE_100K.items():
        register_scannet_context_instances_single(
            key,
            _get_scannet_instances20_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_dataset_single(root, dataset_name="ai2thor"):
    if dataset_name == "ai2thor":
        split_dict = _PREDEFINED_SPLITS_AI2THOR
    elif dataset_name == "alfred":
        split_dict = _PREDEFINED_SPLITS_ALFRED
    elif dataset_name == "replica":
        split_dict = _PREDEFINED_SPLITS_REPLICA
    elif dataset_name == "s3dis":
        split_dict = _PREDEFINED_SPLITS_S3DIS
    elif dataset_name == "matterport":
        split_dict = _PREDEFINED_SPLITS_MATTERPORT
    elif dataset_name == "scannet200":
        split_dict = _PREDEFINED_SPLITS_CONTEXT20_SCANNET200_SINGLE_100K
    else:
        raise NotImplementedError("dataset_name {} not supported".format(dataset_name))
    for key, (image_root, json_file) in split_dict.items():
        register_scannet_context_instances_single(
            key,
            _get_dataset_instances_meta(dataset=dataset_name),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__.endswith(".builtin"):
    # Detectron1 registers some datasets on its own, we remove them here
    DatasetCatalog.clear()
    MetadataCatalog.clear()

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_scannet_context20_scannet_single(_root)

    # scannet200 register
    register_all_dataset_single(_root, dataset_name="scannet200")
    register_all_dataset(_root, dataset_name="scannet200")

    # matterport register
    register_all_dataset_single(_root, dataset_name="matterport")

    # Referential grounding 3D Datasets
    _root_sr3d = os.getenv(
        "REF_DATASET", "/path/to/language_grounding/refer_it_3d/"
    )
    register_ref(_root_sr3d)
    
    _root_2d = os.getenv("DETECTRON2_DATASETS_2D", "datasets")
    # Register COCO datasets
    register_all_coco(_root_2d)

    _root_coco_ref = os.getenv("DETECTRON2_DATASETS_2D")
    register_coco_ref(_root_coco_ref)
