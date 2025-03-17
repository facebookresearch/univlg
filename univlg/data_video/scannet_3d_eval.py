# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging

import detectron2.utils.comm as comm
import ipdb
import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from univlg.global_vars import (
    AI2THOR_CLASS_ID_MULTIPLIER,
    ALFRED_CLASS_ID_MULTIPLIER,
)
from univlg.utils.misc import all_gather, is_dist_avail_and_initialized

from ..utils.util_3d import convert_3d_to_2d_dict_format
from ..utils.util_video_to_3d import convert_video_instances_to_3d
from .segmentation_benchmark.evaluate_semantic_instance import Scannet_Evaluator

st = ipdb.set_trace


class Scannet3DEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        eval_sparse=False,
        distributed=True,
        output_dir=None,
        cfg=None,
    ):
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self.eval_sparse = eval_sparse
        self.cfg = cfg
        self.dataset_name = dataset_name

        if "ai2thor" in dataset_name:
            self.multiplier = AI2THOR_CLASS_ID_MULTIPLIER
        elif "alfred" in dataset_name:
            self.multiplier = ALFRED_CLASS_ID_MULTIPLIER
        else:
            self.multiplier = 1000

        self.scannet_evaluator = Scannet_Evaluator(
            dataset_name, evaluate_subset=cfg.EVALUATE_SUBSET
        )

        self._cpu_device = torch.device("cpu")

    def reset(self):
        self.has_gt_full = []
        self.has_pred_full = []
        self.y_true_full = []
        self.y_score_full = []
        self.hard_false_negatives_full = []

    def parse_ghost(self, input):
        labels = input["scannet_gt_classes"]
        masks = input["scannet_gt_masks"]

        point_semantic_instance_labels = torch.zeros(masks.shape[1], dtype=torch.int64)
        for i in range(masks.shape[0]):
            point_semantic_instance_labels[masks[i].to(torch.bool)] = labels[
                i
            ] * self.multiplier + (i + 1)

        gt = {"masks": masks, "labels": point_semantic_instance_labels}

        return gt

    def parse_normal(self, input):
        h, w = input["instances_all"][0].image_size
        num_frames = len(input["instances_all"])
        target_dict = convert_video_instances_to_3d(
            input["instances_all"],
            num_frames,
            h,
            w,
            self._cpu_device,
            convert_point_semantic_instance=True,
            multiplier=self.multiplier,
            evaluate_subset=self.cfg.EVALUATE_SUBSET,
        )

        # This is especially for Ai2Thor where we evaluate at half resolution instead of full
        # to save memory
        if self.cfg.HIGH_RES_INPUT:
            target_dict["masks"] = target_dict["masks"].float()
            target_dict["masks"] = torch.nn.functional.interpolate(
                target_dict["masks"],
                scale_factor=0.5,
                mode="nearest",
            )
            target_dict[
                "point_semantic_instance_label"
            ] = torch.nn.functional.interpolate(
                target_dict["point_semantic_instance_label"][None].float(),
                scale_factor=0.5,
                mode="nearest",
            )[
                0
            ].long()

        # some depth values might be missing or invalid for eg. glass objects
        # we exclude those points from evaluation if this flag is active
        if self.eval_sparse:
            valids = torch.stack(input["valids"]).reshape(num_frames, h, w)

            if self.cfg.HIGH_RES_INPUT:
                valids = (
                    torch.nn.functional.interpolate(
                        valids.unsqueeze(0).float(),
                        scale_factor=0.5,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .to(torch.bool)
                )

            gt_masks_per_video = target_dict["masks"][:, valids]
            gt_classes_per_video = target_dict["labels"]
            point_semantic_instance_label = target_dict[
                "point_semantic_instance_label"
            ][valids]
        else:
            gt_masks_per_video = target_dict["masks"]
            gt_classes_per_video = target_dict["labels"]
            point_semantic_instance_label = target_dict["point_semantic_instance_label"]

        gt_masks_per_video_3d = gt_masks_per_video.cpu()
        gt_classes_per_video_3d = gt_classes_per_video.cpu()
        point_semantic_instance_label_3d = point_semantic_instance_label.flatten(
            0
        ).cpu()

        gt = {
            "masks": gt_masks_per_video_3d,
            "labels": point_semantic_instance_label_3d,
            "class_labels": gt_classes_per_video_3d,
        }
        return gt

    def parse_gt(self, inputs):
        """
        Args:
        inputs['all_instances']: all k-frames instances in detectron2 format
        Returns:
        [{
            "masks": list of gt instances
            "labels": labels for each mask
        }]
        Takes k image ground-truths, unflatten all images to 3D,
        and prepares for 3D evaluation format.
        """
        if self.cfg.USE_GHOST_POINTS:
            return self.parse_ghost(inputs)
        else:
            return self.parse_normal(inputs)

    def parse_preds(self, output):
        """
        Args:
        outputs['instances']: all k-frames instances in detectron2 format
        Returns:
        [{
            "pred_classes": predicted classes
            "pred_masks": predicted masks
            "pred_scores": scores
        }]
        Takes 3D predictions from model outputs and prepares for 3D evaluation
        format.
        """
        pred = output["instances_3d"]
        for key in pred:
            if isinstance(pred[key], torch.Tensor):
                pred[key] = pred[key].cpu().numpy()
        return pred

    def process(self, inputs, outputs):
        """
        Args:
            inputs (list[dict]): A list of length Bs, each item corresponding to one batch element.
                Depending on the configuration, each dictionary may contain:

                - If USE_GHOST_POINTS is True:
                    * "scannet_gt_classes" (torch.Tensor): shape [M], where M is the number of ground-truth objects.
                    * "scannet_gt_masks" (torch.Tensor): shape [M, P], where P is the total number of 3D points.

                - Otherwise:
                    * "instances_all" (list[detectron2.structures.Instances]): length = number of frames,
                        each Instances object contains per-frame ground-truths. The first Instances object
                        provides the image size (H, W).
                    * "valids" (list[torch.Tensor], optional): length = number of frames, each of shape [H, W],
                        indicating valid depth regions (only used when eval_sparse is True).

            outputs (list[dict]): A list of length Bs, each item corresponding to model output for one batch element.
                Each dictionary contains:
                * "instances_3d" (dict): A dictionary of 3D predictions, where:
                    - "pred_classes" (torch.Tensor): shape [N], indices of predicted classes.
                    - "pred_masks" (torch.Tensor): shape [P, N], binary indicators per point (P) and per instance (N).
                    - "pred_scores" (torch.Tensor): shape [N], confidence scores.
                    - "scannet_gt_masks" (torch.Tensor, optional): shape [M, P], provided if ghost-format ground truth is stored here.
                    - "scannet_gt_classes" (torch.Tensor, optional): shape [M], provided if ghost-format ground truth is stored here.
        """
        if outputs is None:
            return
        
        predictions = []
        ground_truths = []
        for input, output in zip(inputs, outputs):
            # output stores the processed gt in case of ghost point experiment
            if self.cfg.USE_GHOST_POINTS:
                gt_parse_dict = output["instances_3d"]
            else:
                gt_parse_dict = input

            gt = self.parse_gt(gt_parse_dict)
            pred = self.parse_preds(output)
            predictions.append(pred)
            ground_truths.append(gt)

        preds_dict = {}
        gts_dict = {}

        for idx, (p, g) in enumerate(zip(predictions, ground_truths)):
            preds_dict[idx] = p
            gts_dict[idx] = g
        (
            has_gt_full,
            has_pred_full,
            y_true_full,
            y_score_full,
            hard_false_negatives_full,
        ) = self.scannet_evaluator.evaluate(preds_dict, gts_dict)

        self.has_gt_full.append(has_gt_full)
        self.has_pred_full.append(has_pred_full)
        self.y_true_full.append(y_true_full)
        self.y_score_full.append(y_score_full)
        self.hard_false_negatives_full.append(hard_false_negatives_full)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if is_dist_avail_and_initialized():
            has_gt_full = all_gather(self.has_gt_full)
            has_gt_full = list(itertools.chain(*has_gt_full))

            has_pred_full = all_gather(self.has_pred_full)
            has_pred_full = list(itertools.chain(*has_pred_full))

            y_score_full = all_gather(self.y_score_full)
            y_score_full = list(itertools.chain(*y_score_full))

            y_true_full = all_gather(self.y_true_full)
            y_true_full = list(itertools.chain(*y_true_full))

            hard_false_negatives_full = all_gather(self.hard_false_negatives_full)
            hard_false_negatives_full = list(
                itertools.chain(*hard_false_negatives_full)
            )

            # clear the memory
            self.has_gt_full = []
            self.has_pred_full = []
            self.y_true_full = []
            self.y_score_full = []
            self.hard_false_negatives_full = []

            if not comm.is_main_process():
                return {}
        else:
            has_gt_full = self.has_gt_full
            has_pred_full = self.has_pred_full
            y_score_full = self.y_score_full
            y_true_full = self.y_true_full
            hard_false_negatives_full = self.hard_false_negatives_full

            # clear the memory
            self.has_gt_full = []
            self.has_pred_full = []
            self.y_true_full = []
            self.y_score_full = []
            self.hard_false_negatives_full = []

        results = self._eval(
            has_gt_full,
            has_pred_full,
            y_score_full,
            y_true_full,
            hard_false_negatives_full,
        )
        self.scannet_evaluator.print_results(results, self._logger)
        self._results = convert_3d_to_2d_dict_format(results)

        return copy.deepcopy(self._results)

    def _eval(
        self,
        has_gt_full,
        has_pred_full,
        y_score_full,
        y_true_full,
        hard_false_negatives_full,
    ):
        self._logger.info("Preparing results for Scannet3D format ...")
        has_gt = np.stack(has_gt_full).any(0)
        has_pred = np.stack(has_pred_full).any(0)
        di, oi, li = y_score_full[0].shape
        y_score = np.empty((di, oi, li), object)
        y_true = np.empty((di, oi, li), object)

        for di_ in range(di):
            for oi_ in range(oi):
                for li_ in range(li):
                    all_y_scores = []
                    for i in range(len(y_score_full)):
                        all_y_scores.append(y_score_full[i][di_, oi_, li_])
                    y_score[di_, oi_, li_] = np.concatenate(all_y_scores)

                    all_y_trues = []
                    for i in range(len(y_true_full)):
                        all_y_trues.append(y_true_full[i][di_, oi_, li_])
                    y_true[di_, oi_, li_] = np.concatenate(all_y_trues)

        hard_false_negatives = np.stack(hard_false_negatives_full, -1).sum(-1)
        ap = self.scannet_evaluator.compute_ap(
            has_gt, has_pred, y_score, y_true, hard_false_negatives
        )
        mAPs = self.scannet_evaluator.compute_averages(ap)
        return mAPs
