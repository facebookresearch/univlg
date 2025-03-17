# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from copy import copy
import ipdb
from univlg.data_video.sentence_utils import (
    create_positive_map,
)
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from transformers import AutoTokenizer, RobertaTokenizerFast

from .dataset_mapper_coco import COCOInstanceNewBaselineDatasetMapper, convert_coco_poly_to_mask
import numpy as np
from univlg.data_video.datasets.ref_coco_utils import get_root_and_nouns, consolidate_spans
from pycocotools import mask as coco_mask
from copy import deepcopy

st = ipdb.set_trace

def polygon_to_mask(segmentation, height, width):
    """
    Convert a COCO polygon segmentation to a binary mask.

    Args:
    segmentation (list): List of polygon coordinates in COCO format.
    height (int): Height of the image.
    width (int): Width of the image.

    Returns:
    np.array: Binary mask of shape (height, width).
    """
    rles = coco_mask.frPyObjects(segmentation, height, width)
    rle = coco_mask.merge(rles)
    return coco_mask.decode(rle).astype(np.uint8)


class RefCocoDatasetMapper:
    def __init__(
        self,
        cfg,
        is_train,
        dataset_name,
        decoder_3d,
    ):
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.decoder_3d = decoder_3d

        self.coco_mapper = COCOInstanceNewBaselineDatasetMapper(
            self.cfg,
            is_train=self.is_train,
            dataset_name="coco_2017_train" if self.is_train else "coco_2017_val",
            decoder_3d=self.decoder_3d,
        )

        if self.cfg.TEXT_ENCODER_TYPE == "clip":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        elif self.cfg.TEXT_ENCODER_TYPE == "jina":
            self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
        else:
            t_type = "roberta-base"
            self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type)

    def process_lang_data(self, lang_dict, scene_dict):
        text_caption = lang_dict["caption"]
        _, _, root_spans, neg_spans = get_root_and_nouns(text_caption)
        tokens_positive = consolidate_spans(root_spans, text_caption)

        if len(tokens_positive) == 0:
            tokens_positive = [(0, len(text_caption.split(' ')[0]))]

        max_len = (
            self.cfg.MODEL.MAX_SEQ_LEN
            if not self.cfg.TEXT_ENCODER_TYPE == "clip"
            else 77
        )
        tokenized = self.tokenizer(
            text_caption, return_tensors="pt", max_length=max_len, truncation=True
        )

        try:
            positive_map = create_positive_map(
                tokenized, [tokens_positive], max_query_len=max_len
            )
        except Exception:
            st()
            positive_map = create_positive_map(
                tokenized, [tokens_positive], max_query_len=max_len
            )

        # apply transforms to the target mask and bbox
        image_shape = scene_dict['image_shape']
        transforms = scene_dict['transforms']
        obj = {}
        # obj['bbox'] = []
        obj['bbox'] = copy(lang_dict['target_bbox'])  # [x, y, w, h]
        obj['bbox_mode'] = BoxMode.XYWH_ABS
        obj['segmentation'] = lang_dict['target_segmentation']
        obj['category_id'] = 0

        orig_instance_shape = (scene_dict['height'], scene_dict['width'])
        anno_orig_no_augment = utils.transform_instance_annotations(deepcopy(obj), [], orig_instance_shape)
        orig_instances = utils.annotations_to_instances([anno_orig_no_augment], orig_instance_shape)
        h, w = orig_instances.image_size
        orig_gt_masks = orig_instances.gt_masks
        orig_gt_masks = convert_coco_poly_to_mask(orig_gt_masks.polygons, h, w)
        orig_instances.gt_masks = orig_gt_masks

        anno = utils.transform_instance_annotations(deepcopy(obj), transforms, image_shape)
        instances = utils.annotations_to_instances([anno], image_shape)
        h, w = instances.image_size

        gt_masks = instances.gt_masks
        try:
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
        except Exception:
            assert len(gt_masks.polygons[0]) == 0
            gt_masks = torch.zeros((1, h, w), dtype=torch.uint8)
            gt_masks[0, 0] = 1
            
        instances.gt_masks = gt_masks

        instance_ids = [
            (obj_id + 1) * 1000 + i for i, obj_id in enumerate(instances.gt_classes)
        ]
        instances.instance_ids = torch.tensor(instance_ids, dtype=torch.int64)

        new_lang_dict = {
            "text_caption": text_caption,
            "tokens_positive": tokens_positive,
            "positive_map": positive_map,
            "positive_map_od": None,
            'instances_all': [instances],
            'target_bbox': lang_dict['target_bbox'],
            'target_segmentation': [orig_instances],
        }
        new_lang_dict = {**scene_dict, **new_lang_dict}
        return new_lang_dict

    def get_scene_data(self, input_dataset_dict):
        file_name = input_dataset_dict['coco_path'] / "train2017" / f"{input_dataset_dict['image_id']:012d}.jpg"
        coco_dataset_dict = {'file_name': str(file_name)}
        dataset_dict = self.coco_mapper(coco_dataset_dict)
        return dataset_dict

    def __call__(self, dataset_dict):
        scene_data = self.get_scene_data(dataset_dict)
        aggregated_data = self.process_lang_data(dataset_dict, scene_data)
        aggregated_data['dataset_name'] = self.dataset_name
        return aggregated_data
