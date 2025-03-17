# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import itertools
import logging
import numpy as np
import torch
from prettytable import PrettyTable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pytorch3d.ops import knn_points
import pyviz3d.visualizer as viz
from pathlib import Path
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from collections import Counter
import json

from univlg.utils.bbox_utils import _set_axis_align_bbox, _get_ious, get_3d_box_scanrefer
from univlg.utils.misc import all_gather, is_dist_avail_and_initialized

import ipdb
st = ipdb.set_trace

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    x_c = 0.5 * (x0 + x1)
    y_c = 0.5 * (y0 + y1)
    z_c = 0.5 * (z0 + z1)
    w = x1 - x0
    h = y1 - y0
    d = z1 - z0
    return torch.stack([x_c, y_c, z_c, w, h, d], dim=-1)

def visualize_pc_masks_and_bbox(
    pc, color, gt_pcs,
    pred_pcs, gt_bbox, pred_bbox,
    data_dir=None, sample_name=None, inputs=None,
    gt_anchor_pcs=None, gt_anchor_bboxs=None, sr3d_data=None,
    anchor_pcs=None, anchor_bboxs=None
):
    """
    Input
        pc: N X 3
        color: N X 3 (0-255)
        gt_masks: M X N
        pred_masks: M_pred X N
        gt_bbox: M X 6
        pred_bbox: M_pred X 6 (min-max box) 
    """
    point_size = 25
    v = viz.Visualizer()
    v.add_points("RGB", pc,
                 colors=color,
                 alpha=0.8,
                 visible=True,
                 point_size=25)

     # add gt masks
    masks_colors = [np.tile(np.array([0, 255, 0])[None], (pc.shape[0], 1)) for pc in gt_pcs]
    v.add_points(
        "Instances (GT)", gt_pcs[0],
        colors=masks_colors[0],
        alpha=0.8,
        visible=False,
        point_size=point_size
    )

    # add pred masks
    dists = knn_points(torch.from_numpy(pc[None]).cuda(), torch.from_numpy(pc[None]).cuda(), K=8)[0][0, :, 1:].mean(1)
    threshold = dists.mean() + 2 * dists.std()
    pc = pc[(dists < threshold).cpu().numpy()]
    masks_colors = [np.tile(np.array([255, 0, 0])[None], (pc.shape[0], 1)) for pc in pred_pcs]
    v.add_points(
        "Instances (PRED)", pred_pcs[0],
        colors=masks_colors[0],
        alpha=0.8,
        visible=False,
        point_size=point_size)

    # add gt boxes
    gt_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(gt_bbox)).numpy()
    v.add_bounding_box(
        'Boxes (GT)',
        position=gt_bbox[..., :3][0],
        size=gt_bbox[..., 3:][0],
        color=np.array([0, 255, 0]),
        alpha=0.8,
        edge_width=0.03
    )

    # add pred boxes
    pred_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(pred_bbox)).numpy()
    v.add_bounding_box(
        'Boxes (Pred)',
        position=pred_bbox[..., :3][0],
        size=pred_bbox[..., 3:][0],
        color=np.array([255, 0, 0]),
        alpha=0.8,
        visible=True,
        edge_width=0.03)

    if gt_anchor_pcs is not None:
        anchor_colors = get_color(len(gt_anchor_pcs))
        for i in range(len(gt_anchor_pcs)):
            _anchor_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(gt_anchor_bboxs[i])).numpy()
            _anchor_name = sr3d_data['anchors_names'][i]
            v.add_points(
                f"A PC {i},{_anchor_name[:5]}", gt_anchor_pcs[i],
                colors=np.array(anchor_colors[i])[None].repeat(gt_anchor_pcs[i].shape[0], axis=0),
                alpha=0.8, visible=False, point_size=point_size)
            v.add_bounding_box(
                f'A BB {i},{_anchor_name[:5]}',
                position=_anchor_bbox[..., :3][0],
                size=_anchor_bbox[..., 3:][0],
                color=np.array([0, 255, 0]),
                alpha=0.8,
                edge_width=0.03,
                visible=False
            )

        v.add_labels(
            'Labels',
            [sr3d_data['text_caption'], sr3d_data['target_name'], sr3d_data['anchors_names']],
            [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            [np.array([255.0, 0.0, 0.0]), np.array([0.0, 255.0, 0.0]), np.array([0.0, 0.0, 255.0])],
            visible=True
        )

    if anchor_pcs is not None:
        anchor_colors = get_color(len(anchor_pcs))
        for i in range(0, min(len(gt_anchor_pcs), len(anchor_pcs))):
            _anchor_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(anchor_bboxs[i])).numpy()
            v.add_points(
                f"PA PC {i}", anchor_pcs[i],
                colors=np.array(anchor_colors[i])[None].repeat(anchor_pcs[i].shape[0], axis=0),
                alpha=0.8, visible=False, point_size=point_size)
            v.add_bounding_box(
                f'PA BB {i}',
                position=_anchor_bbox[..., :3][0],
                size=_anchor_bbox[..., 3:][0],
                color=np.array(anchor_colors[i]),
                alpha=0.8,
                edge_width=0.03,
                visible=False
            )

    if gt_anchor_pcs is not None and anchor_pcs is not None:
        print(f"Found {len(gt_anchor_pcs)} GT Anchors and {len(anchor_pcs)} Pred Anchors")

    if data_dir is None:
        data_dir = os.environ['OUTPUT_DIR_PREFIX'] + '/debug/bdetr2_visualizations'

    data_dir = Path(f"{data_dir}/{inputs[0]['dataset_name']}/{sample_name.replace(' ', '_')[:100]}")
    if not data_dir.exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved to {data_dir}")
    v.save(str(data_dir))

def get_color(max_value: int, colormap='spring'):
    colormap = plt.get_cmap('spring')  # Pink is 0, Yellow is 1
    colors = [mcolors.to_rgb(colormap(i / max_value)) for i in range(max_value)]  # Generate colors
    return (np.array(colors) * 255).astype(int).tolist()

def convert_bbox_to_corners_with_colors(bboxes):
    """
    Convert bounding boxes to a format with 8 corners and deterministically generate a color for each box.

    Args:
    - bboxes (np.array): An array of shape [N, 6] containing bounding boxes.

    Returns:
    - np.array: An array of dictionaries, each with 'corners' and 'color' keys.
    """
    converted = []
    colors = get_color(len(bboxes))
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        corners = np.array([
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
        ])
        converted.append({"corners": corners.tolist(), "color": colors[idx]})
    return np.array(converted)

def crop_pcd_to_combined_bbox(pcd: torch.Tensor, bboxes: torch.Tensor, extra_extent: float) -> torch.Tensor:
    """
    Crop point cloud data (PCD) to match the combined extent of all bounding boxes.

    Parameters:
    - pcd: torch.Tensor of shape [M, 6] representing the point cloud data with [xyz rgb].
    - bboxes: torch.Tensor of shape [N, 6] representing the bounding boxes with [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
    - torch.Tensor: Cropped PCD matching the combined extent of all bounding boxes.
    """
    # Calculate the min/max extent of all bounding boxes
    combined_min = torch.min(bboxes[:, :3], dim=0).values - extra_extent
    combined_max = torch.max(bboxes[:, 3:], dim=0).values + extra_extent

    # Check if points are within the combined bbox extents
    in_combined_bbox = (pcd[:, 0] >= combined_min[0]) & (pcd[:, 0] <= combined_max[0]) & \
                        (pcd[:, 1] >= combined_min[1]) & (pcd[:, 1] <= combined_max[1]) & \
                        (pcd[:, 2] >= combined_min[2]) & (pcd[:, 2] <= combined_max[2])

    return pcd[in_combined_bbox]

def sample_k_rows(tensor, K):
    indices = torch.randperm(tensor.size(0))[:K]
    return tensor[indices]

class ReferrentialGroundingEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        thresholds,
        topks,
        cfg=None
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.thresholds = thresholds
        self.topks = topks
        self.cfg = cfg
        self._cpu_device = torch.device("cpu")
        self.num_viz = 0

    def reset(self):
        self.detection_results = []
        self.mask_detection_results = []
        self.num_viz = 0

    def process(self, inputs, outputs):
        if type(outputs[0]) != dict:
            outputs = outputs[1]
        assert len(inputs) == 1
        for j in range(len(outputs)):
            inputs_ = copy.copy(inputs[0])
            inputs_['sr3d_data'] = [inputs_['sr3d_data'][j]]
            if self.cfg.USE_GT_MASKS:
                self.process_single_gt([inputs_], [outputs[j]])
            else:
                self.process_single([inputs_], [outputs[j]])
                
    def process_single_gt(self, inputs, outputs):
        """
        Process a single input-output pair for 3D referential grounding evaluation.

        This function handles one sample (batch size = 1) from a 3D scene and its corresponding model
        predictions. It extracts predicted instance scores and masks, computes axis-aligned bounding boxes
        (AABBs) for the top-K predictions, and evaluates these predictions against the ground truth target
        (specified in the input). Depending on configuration flags, it may also perform subsampling of the
        point cloud or visualize the results.

        Parameters
        ----------
        inputs : list[dict]
            A list containing one dictionary with the following required keys:
            - "scannet_coords": torch.Tensor of shape (P, 3)
                    3D coordinates of the scene's point cloud (e.g., P ≈ 2.5e5 points).
            - "scannet_color": torch.Tensor of shape (P, 3)
                    Color information for each point (RGB values in 0-255).
            - "scannet_labels": torch.Tensor of shape (P, 2)
                    Per-point labels; the second column (index 1) is used to match the ground truth target.
            - "image_id": str
                    A unique identifier for the scene.
            - "file_name": str
                    File path string used to extract the scene name for visualization.
            - 'dataset_name': str
                    Name of the dataset (used for visualization path).
            - "sr3d_data": list of dict
                    A list with one dictionary that must include:
                    - "target_id": int
                        The ground truth object identifier to detect.
                    - "annotation_id": int or None
                        The annotation id for the target.
                    - "anchor_ids": list of int
                        A list of additional anchor object identifiers (used if visualization is enabled).
                    - "text_caption": str
                        A caption or description for the scene/object.
                    - 'anchors_names': list[str]
                        Names of anchor objects (used for extra viz).
                    - 'target_name': str
                        Name of the target object (used for viz).
                    (Other entries in sr3d_data may be present but are not used in this function.)

        outputs : list[dict]
            A list containing one dictionary with a key "instances_3d", which is a dict that must include:
            - "pred_scores": torch.Tensor of shape (N, C)
                    Prediction confidence scores for N candidate instances. The first column (index 0) is used
                    as the main score (e.g., shape (100, 3)).
            - "pred_masks": torch.Tensor of shape (N, M)
                    Raw prediction masks (pre-thresholding) for each instance over M points (e.g., M ≈ 2.5e5).
            - "scannet_idxs": torch.Tensor or None (optional)
                    Indices used for subsampling the point cloud when FORCE_SUBSAMPLE is enabled.
            - 'reduced_scene_id_to_original_id': torch.Tensor | None
            - (Other keys like "scannet_gt_masks", etc. may be present but are not used.)

        Processing Details
        ------------------
        - The function extracts the predicted scores and applies a sigmoid to the predicted masks,
        thresholding them at 0.5 to yield binary masks.
        - It selects the top-K predictions (with K equal to max(self.topks)) based on the confidence scores.
        - For each of the top predictions, it uses the binary mask to index into "scannet_coords" (a tensor of shape (P, 3))
        to extract the corresponding 3D points, and computes an axis-aligned bounding box via _set_axis_align_bbox.
        If no points are selected, a box with -∞ values is used.
        - If the FORCE_SUBSAMPLE flag is enabled, the function subsamples "scannet_coords", "scannet_labels", and
        "scannet_color" using indices from "scannet_idxs".
        - The ground truth points for the target object (specified by "target_id" in sr3d_data) are
        extracted from "scannet_coords" using "scannet_labels". An axis-aligned bounding box for the ground truth is
        computed, and Intersection-over-Union (IoU) is calculated between each predicted box and the ground truth box.
        - Similarly, mask IoUs are computed between the predicted binary masks and a full ground truth mask derived
        from "scannet_labels".
        - The detection and mask detection results (each a boolean array indicating success at various thresholds)
        are appended to the evaluator's internal lists for later aggregation.

        Returns
        -------
        None
        """

        assert len(inputs) == len(outputs) == 1
        
        # Root word is the first object
        scores = outputs[0]['instances_3d']['pred_scores'][:, 0]
        
        # Average confidence of each prediction mask.
        top_k_weighted_scores = scores
        max_k = max(self.topks)
        
        top_id = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        pred_ids = (outputs[0]['instances_3d']['pred_masks'])[top_id, :].argmax(-1)
        reduced_scene_id_to_original_id = outputs[0]['instances_3d']['reduced_scene_id_to_original_id']
        pred_ids = reduced_scene_id_to_original_id[pred_ids]
        target_id = inputs[0]['sr3d_data'][0]['target_id']
        detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
        correct = (pred_ids == target_id).cpu().numpy()
        for i in range(len(self.topks)):
            detected[i, :] = np.any(correct[:self.topks[i], None], axis=0)
        self.detection_results.append(detected)
        self.mask_detection_results.append(detected)

    def process_single(self, inputs, outputs):
        assert len(inputs) == len(outputs) == 1
        
        # Root word is the first object
        scores = outputs[0]['instances_3d']['pred_scores'][:, 0]
        pred_masks = F.sigmoid(outputs[0]['instances_3d']['pred_masks'])
        masks = pred_masks > 0.5
        
        # Average confidence of each prediction mask.
        top_k_weighted_scores = scores
        top_k_pred_masks = masks.flatten(1)
        max_k = max(self.topks)
        top_id = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        top_masks = top_k_pred_masks[top_id, :].to(self._cpu_device).numpy()

        if self.cfg.FORCE_SUBSAMPLE:
            dists_close = outputs[0]['instances_3d']['scannet_idxs'].cpu()
            inputs[0]['scannet_coords'] = inputs[0]['scannet_coords'][dists_close]
            inputs[0]['scannet_labels'] = inputs[0]['scannet_labels'][dists_close]
            inputs[0]['scannet_color'] = inputs[0]['scannet_color'][dists_close]

        top_bboxs = []
        pred_pcs = []
        for i in range(max_k):
            cur_pc = inputs[0]['scannet_coords'].to(self._cpu_device).numpy()[top_masks[i]]
            pred_pcs.append(cur_pc)
            if cur_pc.shape[0] == 0:
                top_bboxs.append(np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]))
            else:
                top_bboxs.append(_set_axis_align_bbox(cur_pc))
        top_bboxs = np.array(top_bboxs)

        if self.cfg.TEST_DATASET_INFERENCE:
            assert len(inputs[0]['sr3d_data'])
            if pred_pcs[0].shape[0] > 0:
                max_ = np.max(pred_pcs[0], axis=0)
                min_ = np.min(pred_pcs[0], axis=0)
            else:
                max_ = np.array([0.0, 0.0, 0.0])
                min_ = np.array([0.0, 0.0, 0.0])

            center = (max_ + min_) / 2.0
            box_size = max_ - min_
            scanrefer_box = get_3d_box_scanrefer(box_size, 0, center)
            self.detection_results.append({
                "scene_id": inputs[0]['image_id'],
                "object_id": inputs[0]['sr3d_data'][0]['target_id'],
                "ann_id": inputs[0]['sr3d_data'][0]['annotation_id'],
                "bbox": scanrefer_box.tolist(),
            })
            return

        target_id = inputs[0]['sr3d_data'][0]['target_id']

        try:
            gt_indices = torch.nonzero(inputs[0]['scannet_labels'][:, 1].cpu() == target_id, as_tuple=True)[0]
            gt_pc = inputs[0]['scannet_coords'][gt_indices, :].cpu().numpy()
            gt_bbox = np.expand_dims(_set_axis_align_bbox(gt_pc), axis=0)

            ious = _get_ious(top_bboxs, gt_bbox)
            above_threshold = ious > self.thresholds
            detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
            for i in range(len(self.topks)):
                detected[i, :] = np.any(above_threshold[:self.topks[i], :], axis=0)
                
            full_gt_mask = (inputs[0]['scannet_labels'][:, 1].cpu().numpy() == target_id)
            mask_ious = []
            for i in range(max_k):
                pred_mask = top_masks[i].astype(bool)
                intersection = np.logical_and(pred_mask, full_gt_mask).sum()
                union = np.logical_or(pred_mask, full_gt_mask).sum()
                mask_iou = intersection / union if union > 0 else 0.0
                mask_ious.append(mask_iou)
            mask_ious = np.array(mask_ious)[:, None]
            mask_detected = mask_ious > self.thresholds
            mask_result = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
            for i in range(len(self.topks)):
                mask_result[i, :] = np.any(mask_detected[:self.topks[i], :], axis=0)

        except Exception as e:
            if self.cfg.VIZ_EXTRA_REF or self.cfg.TEST_DATASET_INFERENCE:
                print(f"Skipping visualization for {e}. GT Target not found")
                detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
                mask_result = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
                gt_bbox = np.array([0, 0, 0, 0, 0, 0])
                gt_pc = np.array([[0, 0, 0]])
            else:
                raise e

        if self.cfg.VISUALIZE_REF:
            gt_anchor_bboxs = None
            gt_anchor_pcs = None
            anchor_pcs = None
            anchor_bboxs = None
            if self.cfg.VIZ_EXTRA_REF:
                anchor_ids = inputs[0]['sr3d_data'][0]['anchor_ids']
                gt_anchor_pcs = []
                gt_anchor_bboxs = []
                for anchor_id in anchor_ids:
                    gt_anchor_pc = inputs[0]['scannet_coords'][inputs[0]['scannet_labels'][:, 1] == anchor_id, :].cpu().numpy()
                    if len(gt_anchor_pc) > 0:
                        gt_anchor_pcs.append(gt_anchor_pc)
                        gt_anchor_bboxs.append(np.expand_dims(_set_axis_align_bbox(gt_anchor_pc), axis=0))
                    else:
                        print(f"Anchor {anchor_id} has no points")

                anchor_pcs = []
                anchor_bboxs = []
                for n in range(1, outputs[0]['instances_3d']['pred_scores'].shape[1]):
                    anchor_scores = outputs[0]['instances_3d']['pred_scores'][:, n]
                    anchor_masks = F.sigmoid(outputs[0]['instances_3d']['pred_masks']) > 0.5
                    top_anchor_id = torch.argmax(anchor_scores).to(anchor_masks.device)
                    anchor_pc = inputs[0]['scannet_coords'][anchor_masks[top_anchor_id].to(inputs[0]['scannet_coords'].device)].cpu().numpy()
                    if anchor_pc.shape[0] > 0:
                        anchor_pcs.append(anchor_pc)
                        anchor_bboxs.append(np.expand_dims(_set_axis_align_bbox(anchor_pc), axis=0))

            scene_name = inputs[0]['file_name'].split('/')[-3] + " " + inputs[0]['sr3d_data'][0]['text_caption']
            visualize_pc_masks_and_bbox(
                pc=inputs[0]['scannet_coords'].numpy(),
                color=inputs[0]['scannet_color'].numpy(),
                gt_pcs=[gt_pc],
                pred_pcs=pred_pcs,
                gt_bbox=gt_bbox,
                pred_bbox=top_bboxs[0][None],
                data_dir=self.cfg.VISUALIZE_LOG_DIR,
                sample_name=scene_name,
                inputs=inputs,
                gt_anchor_pcs=gt_anchor_pcs,
                gt_anchor_bboxs=gt_anchor_bboxs,
                sr3d_data=inputs[0]['sr3d_data'][0],
                anchor_pcs=anchor_pcs,
                anchor_bboxs=anchor_bboxs
            )

        self.detection_results.append(detected)
        self.mask_detection_results.append(mask_result)

    def evaluate(self):
        if is_dist_avail_and_initialized():
            detection_results = all_gather(self.detection_results)
            detection_results = list(itertools.chain(*detection_results))
            mask_detection_results = all_gather(self.mask_detection_results)
            mask_detection_results = list(itertools.chain(*mask_detection_results))
            if not comm.is_main_process():
                return {}
        else:
            detection_results = self.detection_results
            mask_detection_results = self.mask_detection_results

        if self.cfg.TEST_DATASET_INFERENCE:
            try:
                Path(self.cfg.TEST_RESULT_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
                print(f'exporting test results to {self.cfg.TEST_RESULT_EXPORT_PATH}/{self.dataset_name}_test_results.json')
                with open(f'{self.cfg.TEST_RESULT_EXPORT_PATH}/{self.dataset_name}_test_results.json', 'w') as json_file:
                    json.dump(detection_results, json_file, indent=4)
            except Exception as e:
                print(f"Error exporting test results: {e}")
                st()
            return None

        self.detection_results = []
        self.mask_detection_results = []
        detection_results = np.array(detection_results).astype(np.float32).mean(axis=0)
        mask_detection_results = np.array(mask_detection_results).astype(np.float32).mean(axis=0)
            
        table = PrettyTable()
        table.field_names = ['Dataset', ''] + ['thresold = ' + str(thres) for thres in self.thresholds]
        for i in range(len(self.topks)):
            row = [self.dataset_name, f'top {self.topks[i]} scores (BOX)'] + ['{:.3f}'.format(detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        for i in range(len(self.topks)):
            row = [self.dataset_name, f'top {self.topks[i]} scores (MASK)'] + ['{:.3f}'.format(mask_detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        print(table)

        res = {}
        
        for i in range(len(self.topks)):
            for j in range(len(self.thresholds)):
                res[f"top_{self.topks[i]}_threshold_{self.thresholds[j]}"] = detection_results[i, j]

        for i in range(len(self.topks)):
            for j in range(len(self.thresholds)):
                res[f"mask_top_{self.topks[i]}_threshold_{self.thresholds[j]}"] = mask_detection_results[i, j]
        return res
