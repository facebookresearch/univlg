# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import json
import copy
from copy import deepcopy
import logging
import random
from pathlib import Path
from imageio import imread
from PIL import Image

import ipdb
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
from detectron2.data.catalog import MetadataCatalog
from pycocotools import mask as coco_mask
from fvcore.transforms.transform import NoOpTransform

from .data_utils import augment_depth_intrinsics, get_multiview_xyz
from univlg.modeling.backproject.backproject import augment_and_interpolate_depth

st = ipdb.set_trace

__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen_3D(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]

    This is a special transform function which makes the 2D scale the same as 3D scale
    so 50% of the time we train on 3D scale and use this function
    50% of the time we train on 2D scale and use the build_transform_gen function
    """
    # assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE
    # print(image_size, min_scale, max_scale)

    augmentation = []

    if is_train:
        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )

        augmentation.extend(
            [
                T.ResizeScale(
                    min_scale=min_scale,
                    max_scale=max_scale,
                    target_height=image_size,
                    target_width=image_size,
                ),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ]
        )
    else:
        pass

    return augmentation


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE_2D
    # min_scale = cfg.INPUT.MIN_SCALE
    # max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if is_train:
        if cfg.INPUT.RANDOM_FLIP != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
        augmentation.extend(
            [
                # T.ResizeScale(
                #     min_scale=min_scale,
                #     max_scale=max_scale,
                #     target_height=image_size,
                #     target_width=image_size,
                # ),
                NoOpTransform(),
                T.FixedSizeCrop(crop_size=(image_size, image_size)),
            ]
        )
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST_2D
        max_size = cfg.INPUT.MAX_SIZE_TEST_2D
        sample_style = "choice"
        augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    return augmentation


def load_depth_and_camera_parameters(
    dataset_name, dataset_dict, transforms, is_train,
    augment_3d, size_divisibility, interpolation_method,
    mask_valid, mean_center, do_rot_scale, scales
    ):
    if 'sam' in dataset_name:
        folder = str(Path(dataset_dict['file_name']).parent.parent)
        file_name = dataset_dict['file_name']
        batch_name = file_name.split('/')[-2]
        folder_3d = str(Path(folder).parent) + '_3d' + '/' + batch_name
    else:
        folder_3d = str(Path(dataset_dict['file_name']).parent.parent) + "_3d"

    file_name = str(Path(dataset_dict['file_name']).name).split(".")[0]
    depth_file_name = folder_3d + "/depth_monocular/" + file_name + ".png"
    pose_file_name = folder_3d + "/pose_monocular/" + file_name + ".txt"
    intrinsic_file_name = folder_3d + "/intrinsic_monocular/" + file_name + ".txt"

    depth = imread(depth_file_name).astype(np.float32)
    depth = depth / 1000.0

    intrinsic_ = np.loadtxt(intrinsic_file_name)
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = intrinsic_
    pose = np.loadtxt(pose_file_name)

    depth, intrinsic = augment_depth_intrinsics(
        depth, intrinsic, transforms, is_train)
    dataset_dict["depths"] = [torch.as_tensor(depth)]
    dataset_dict["intrinsics"] = [torch.as_tensor(intrinsic).float()]
    dataset_dict["poses"] = [torch.as_tensor(pose).float()]

    num_views, height, width = 1, depth.shape[0], depth.shape[1]

    multi_scale_xyz, scannet_pc, original_xyz = get_multiview_xyz(
        shape=(num_views, height, width),
        size_divisibility=size_divisibility,
        depths=dataset_dict["depths"],
        poses=dataset_dict["poses"],
        intrinsics=dataset_dict["intrinsics"],
        is_train=is_train,
        augment_3d=is_train and augment_3d,
        interpolation_method=interpolation_method,
        mask_valid=mask_valid,
        mean_center=mean_center,
        do_rot_scale=do_rot_scale,
        scannet_pc=None,
        scales=scales,
    )
    return multi_scale_xyz, scannet_pc, original_xyz


def load_moge_depth(
    dataset_name, dataset_dict, transforms, is_train,
    augment_3d, size_divisibility, interpolation_method,
    mean_center, do_rot_scale, scales
    ):
    if 'sam' in dataset_name:
        folder = str(Path(dataset_dict['file_name']).parent.parent)
        file_name = dataset_dict['file_name']
        batch_name = file_name.split('/')[-2]
        folder_3d = str(Path(folder).parent) + '_3d' + '/' + batch_name
    else:
        file_name = dataset_dict['file_name']
        folder_3d = str(Path(dataset_dict['file_name']).parent.parent) + "_3d_moge"
        
    points_folder = folder_3d + "/points_new"
    masks_folder = folder_3d + "/masks"
    file_name = file_name.split('/')[-1].split('.')[0]
    points = np.load(f"{points_folder}/{file_name}.npy")
    points = points.astype(np.float32) / 1000.0
    masks = np.load(f"{masks_folder}/{file_name}_mask.npy")
    
    # augment points and masks
    points_transforms = deepcopy(transforms)
    if is_train:
        points_transforms[-1].pad_value = -10.0
        points_transforms[0].interp = Image.NEAREST
    else:
        points_transforms[0].interp = Image.NEAREST
    
    points = T.apply_transform_gens(
        points_transforms, T.AugInput(points)
        )[0].image
    
    masks = transforms.apply_segmentation(masks.astype(np.float32)).astype(bool)
    
    points[masks == 0] = -10.0
    
    if scales is None:
        scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    else:
        assert len(scales) == 4
        scales = {f"res{i+2}": s for i, s in enumerate(scales)}
    
    h, w = points.shape[:2]
    pad_h = int(np.ceil(h / size_divisibility) * size_divisibility - h)
    pad_w = int(np.ceil(w / size_divisibility) * size_divisibility - w)
    H_padded = h + pad_h
    W_padded = w + pad_w
    features = {
        k: torch.zeros(1, 1, H_padded // s, W_padded // s) for k, s in scales.items()
    }
    augment = is_train and augment_3d
        
    multi_scale_xyz, scannet_pc, original_xyz = augment_and_interpolate_depth(
        torch.from_numpy(points[None, None]),
        list(features.values()),
        augment=augment,
        method=interpolation_method,
        scannet_pc=None,
        padding=(pad_h, pad_w),
        mean_center=mean_center,
        do_rot_scale=do_rot_scale,
        align_matrix=None,
        vil3d=False,
    )
    multi_scale_xyz = multi_scale_xyz[::-1]
    return multi_scale_xyz, scannet_pc, original_xyz[0]

# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        tfm_gens_3D,
        cfg,
        image_format,
        dataset_name,
        decoder_3d=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.tfm_gens_3D = tfm_gens_3D
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.decoder_3d = decoder_3d
    
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )
        self.class_names = {
            k: v for k, v in enumerate(MetadataCatalog.get(dataset_name).thing_classes)
        }

        self.num_classes = len(self.class_names)
        # if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.DETIC and not self.cfg.OPEN_VOCAB_SOFTMAX:
        #     # add "__background__" class
        #     self.class_names[self.num_classes] = "__background__"

        self.img_format = image_format
        self.is_train = is_train

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        tfm_gens_3D = build_transform_gen_3D(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "tfm_gens_3D": tfm_gens_3D,
            "image_format": cfg.INPUT.FORMAT,
            "cfg": cfg,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        padding_mask = np.ones(image.shape[:2])

        if self.cfg.AUGMENT_WITH_3D_SCALE and self.is_train:
            if random.random() > 0.5:
                tfm_gens = self.tfm_gens_3D
            else:
                tfm_gens = self.tfm_gens
            image, transforms = T.apply_transform_gens(tfm_gens, image)
        else:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["padding_mask"] = torch.as_tensor(
            np.ascontiguousarray(padding_mask)
        )

        if self.decoder_3d:
            if self.cfg.USE_MOGE_DEPTH:
                multi_scale_xyz, scannet_pc, original_xyz = load_moge_depth(
                    self.dataset_name, dataset_dict, transforms, self.is_train,
                    self.cfg.INPUT.AUGMENT_3D, self.cfg.INPUT.SIZE_DIVISIBILITY,
                    self.cfg.MODEL.INTERPOLATION_METHOD,
                    self.cfg.MEAN_CENTER, self.cfg.DO_ROT_SCALE, self.cfg.MULTIVIEW_XYZ_SCALES
                )
            else:
                multi_scale_xyz, scannet_pc, original_xyz = load_depth_and_camera_parameters(
                    self.dataset_name, dataset_dict, transforms, self.is_train,
                    self.cfg.INPUT.AUGMENT_3D, self.cfg.INPUT.SIZE_DIVISIBILITY,
                    self.cfg.MODEL.INTERPOLATION_METHOD, self.cfg.MASK_VALID,
                    self.cfg.MEAN_CENTER, self.cfg.DO_ROT_SCALE, self.cfg.MULTIVIEW_XYZ_SCALES
                )
            
            dataset_dict["multi_scale_xyz"] = multi_scale_xyz
            dataset_dict["scannet_coords"] = scannet_pc
            dataset_dict["original_xyz"] = original_xyz

        mask_format = 'polygon'
        if 'sam' in self.dataset_name or 'paco' in self.dataset_name:
            mask_format = 'bitmask'
        if "sam" in self.dataset_name:
            assert "annotations" not in dataset_dict
            # load annotations
            with open(f"{dataset_dict['annotation_file']}", "r") as f:
                dataset_dict["annotations"] = json.load(f)
            if not self.is_train:
                dataset_dict['annotations_copy'] = copy.deepcopy(dataset_dict['annotations'])
            dataset_dict["annotations"] = dataset_dict["annotations"]["annotations"]

            for anno in dataset_dict['annotations']:
                # orig_h, orig_w = dataset_dict["height"], dataset_dict["width"]
                # orig_mask = segm_to_mask(anno['segmentation'], orig_h, orig_w)
                # anno["bbox"] = _get_xywh_bounding_box(orig_mask)
                # anno['segmentation'] = mask_utils.encode(np.asfortranarray(orig_mask))
                anno["bbox_mode"] = BoxMode.XYWH_ABS
                anno["category_id"] = 0

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            try:
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=mask_format)
                instances = utils.filter_empty_instances(instances)
            except Exception:
                print("No instances found in {}".format(dataset_dict["file_name"]))
            # Generate masks from polygon
            h, w = instances.image_size
            if hasattr(instances, "gt_masks"):
                if mask_format == 'polygon':
                    gt_masks = instances.gt_masks
                    gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                    instances.gt_masks = gt_masks
                elif mask_format == 'bitmask':
                    gt_masks = instances.gt_masks
                    if len(gt_masks) == 0:
                        gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)
                    else:
                        gt_masks = torch.stack([torch.tensor(gt_mask, dtype=torch.uint8) for gt_mask in gt_masks])
                    instances.gt_masks = gt_masks

            instance_ids = [
                (obj_id + 1) * 1000 + i for i, obj_id in enumerate(instances.gt_classes)
            ]
            instances.instance_ids = torch.tensor(instance_ids, dtype=torch.int64)
            dataset_dict["instances"] = instances
            dataset_dict["instances_all"] = [instances]
            dataset_dict["all_classes"] = self.class_names

        dataset_dict["original_all_classes"] = copy.copy(self.class_names)
        dataset_dict["actual_decoder_3d"] = False
        dataset_dict["valid_class_ids"] = np.arange(len(self.class_names))
        dataset_dict["decoder_3d"] = self.decoder_3d
        dataset_dict["file_names"] = [dataset_dict["file_name"]]
        dataset_dict["valids"] = None
        dataset_dict["do_camera_drop"] = False
        dataset_dict["camera_drop_prob"] = 0.0
        dataset_dict["camera_drop_min_frames_keep"] = 1
        dataset_dict["always_keep_first_frame"] = True
        dataset_dict["max_frames"] = 1
        dataset_dict["use_ghost"] = False
        dataset_dict["images"] = [dataset_dict["image"]]
        dataset_dict["multiplier"] = 1000
        dataset_dict["dataset_name"] = self.dataset_name
        dataset_dict["num_classes"] = self.num_classes
        dataset_dict['pseudo_2d_aug'] = self.cfg.PSEUDO_2D_AUG
        dataset_dict['full_scene_dataset'] = False
        dataset_dict['transforms'] = transforms
        dataset_dict['image_shape'] = image_shape
        dataset_dict['is_train'] = self.is_train

        return dataset_dict
