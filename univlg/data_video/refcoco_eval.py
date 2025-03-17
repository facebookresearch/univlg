# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from detectron2.evaluation import DatasetEvaluator
import ipdb
import itertools
from prettytable import PrettyTable
import numpy as np
import torch
from torch.nn import functional as F
import detectron2.utils.comm as comm
import wandb
from univlg.utils.misc import all_gather, is_dist_avail_and_initialized
from univlg.utils.bbox_utils import box_iou, box_cxcywh_to_xyxy
from detectron2.data import detection_utils as utils

st = ipdb.set_trace

class RefCOCOEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        thresholds,
        topks,
        cfg=None,
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.thresholds = thresholds
        self.topks = topks
        self.cfg = cfg
        self._cpu_device = torch.device("cpu")
        self.num_viz = 0
        self.use_mask_iou = cfg.COCO_REF_EVAL_USE_MASK_IOU

    def reset(self):
        self.detection_results = []
        self.num_viz = 0
        self.intersections = []
        self.unions = []

    def process(self, inputs, outputs):
        assert len(inputs) == 1
        self.process_single(inputs[0], outputs[0])

    def process_single(self, inputs, outputs):
        # Get prediction scores and masks
        scores = outputs['instances'].scores
        pred_masks = outputs['instances'].pred_masks.sigmoid() > 0.5  # [Q, H, W]

        top_k_weighted_scores = scores
        top_k_pred_masks = pred_masks
        max_k = max(self.topks)

        top_id = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        top_masks = top_k_pred_masks[top_id]

        if self.use_mask_iou:
            gt_mask = inputs['target_segmentation'][0].gt_masks[0].to(top_masks.device)

            ious = []
            intersections = []
            unions = []
            for pred_mask in top_masks:
                intersection = (pred_mask & gt_mask).float().sum()
                union = (pred_mask | gt_mask).float().sum()
                iou = (intersection / union).item() if union > 0 else 0.0
                ious.append(iou)
                intersections.append(intersection.cpu())
                unions.append(union.cpu())

            ious = np.array(ious)
            intersections = np.array(intersections)
            unions = np.array(unions)
            above_threshold = ious[:, np.newaxis] > self.thresholds  # [max_k, num_thresholds]
            self.intersections.append(intersections[0])
            self.unions.append(unions[0])
        else:
            top_bboxs = []
            for i in range(max_k):
                non_zero_indices = torch.nonzero(top_masks[i], as_tuple=True)
                if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
                    top_bboxs.append(torch.tensor([0.0, 0.0, 0.0, 0.0]))
                else:
                    ymin, xmin = non_zero_indices[0].min(dim=0)[0], non_zero_indices[1].min(dim=0)[0]
                    ymax, xmax = non_zero_indices[0].max(dim=0)[0], non_zero_indices[1].max(dim=0)[0]
                    top_bboxs.append(torch.tensor([xmin, ymin, xmax, ymax]))

            top_bboxs = torch.stack(top_bboxs)
            gt_bbox = torch.tensor(inputs['target_bbox']).reshape(-1, 4)
            gt_bbox[:, 2:] += gt_bbox[:, :2]
            ious = box_iou(top_bboxs, gt_bbox)[0].numpy()

            above_threshold = ious > self.thresholds

        detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)

        for i in range(len(self.topks)):
            detected[i, :] = np.any(above_threshold[:self.topks[i], :], axis=0)

        if self.cfg.VISUALIZE_REF:
            self.viz(inputs, outputs, top_masks)
            
        self.detection_results.append(detected)

    def evaluate(self):
        if is_dist_avail_and_initialized():
            print(f'Rank {comm.get_rank()} is gathering results.')
            print(self.detection_results[0].dtype)
            detection_results = all_gather(self.detection_results)
            detection_results = list(itertools.chain(*detection_results))
            
            if self.use_mask_iou:
                intersections = all_gather(self.intersections)
                intersections = list(itertools.chain(*intersections))
                unions = all_gather(self.unions)
                unions = list(itertools.chain(*unions))
            if not comm.is_main_process():
                return {}
            print(f'Gathered {len(detection_results)} grounding results for evaluation for dataset {self.dataset_name}')
        else:
            detection_results = self.detection_results
            intersections = self.intersections
            unions = self.unions

        self.detection_results = []
        self.intersections = []
        self.unions = []
        detection_results = np.array(detection_results).astype(np.float32).mean(axis=0)

        table = PrettyTable()
        table.field_names = [''] + ['thresold = ' + str(thres) for thres in self.thresholds]
        for i in range(len(self.topks)):
            row = ['top {} scores'.format(self.topks[i])] + ['{:.3f}'.format(detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        print(table)
        
        data = {f"top_{self.topks[i]}_threshold_{self.thresholds[j]}" : detection_results[i, j] for i in range(len(self.topks)) for j in range(len(self.thresholds))}
        
        if self.use_mask_iou:
            ciou = np.array(intersections).astype(np.float64).sum() / np.array(unions).astype(np.float64).sum()
            print(f"CIOU: {ciou}")
            data['ciou'] =  ciou
        return data
    
        
    def viz(self, inputs, outputs, top_masks):
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import os

        plt.clf()
        
        top_masks = top_masks.cpu()

        image_path = inputs['file_name']
        image = np.array(Image.open(image_path))

        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Ground Truth visualization
        ax1.imshow(image)
        ax1.set_title("Ground Truth")

        # Draw GT bbox
        gt_bbox = torch.tensor(inputs['target_bbox']).reshape(-1, 4)
        gt_bbox[:, 2:] += gt_bbox[:, :2]
        gt_bbox = gt_bbox.squeeze().cpu().numpy()

        x_min, y_min, x_max, y_max = gt_bbox
        width = x_max - x_min
        height = y_max - y_min
        gt_rect = patches.Rectangle((x_min, y_min), width, height,
                                    linewidth=2, edgecolor='r', facecolor='none')

        ax1.add_patch(gt_rect)

        if self.use_mask_iou:
            gt_mask = inputs['target_segmentation'][0].gt_masks[0].to(top_masks.device)
            ax1.imshow(gt_mask, alpha=0.5, cmap='jet')

        ax2.imshow(image)
        ax2.set_title("Predictions")

        # Define colors and alphas for top 3 predictions
        # colors = ['red', 'green', 'blue']
        colors = ['green']
        alphas = [0.5]
        # alphas = [0.7, 0.5, 0.3]

        # Draw top predicted masks and bboxes
        for i in range(min(1, len(top_masks))):  # Visualize top 3 predictions
            mask = top_masks[i].cpu().numpy()
            bbox = None
            if self.use_mask_iou:
                # Compute bbox from mask
                non_zero_indices = np.nonzero(mask)
                if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                else:
                    ymin, xmin = non_zero_indices[0].min(), non_zero_indices[1].min()
                    ymax, xmax = non_zero_indices[0].max(), non_zero_indices[1].max()
                    bbox = [xmin, ymin, xmax, ymax]

            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = (*plt.cm.colors.to_rgb(colors[i]), alphas[i])

            # Overlay the mask on the image
            ax2.imshow(colored_mask)

            if bbox is not None:
                # Draw predicted bbox with different color
                pred_rect = patches.Rectangle((bbox[0], bbox[1]),
                                              bbox[2] - bbox[0],
                                              bbox[3] - bbox[1],
                                              linewidth=2, edgecolor=colors[i], facecolor='none')
                ax2.add_patch(pred_rect)

        # Print the % area of each mask over all pixels
        for i in range(min(1, len(top_masks))):
            mask = top_masks[i].cpu().numpy()
            print(f"Mask {i+1} area: {np.sum(mask) / np.prod(mask.shape)}")

        # Save the figure
        os.makedirs('visualization', exist_ok=True)
        plt.savefig(f'visualization/refcoco_viz_{self.num_viz}.png')
        plt.close(fig)
        # Load the saved image
        saved_image_path = f'visualization/refcoco_viz_{self.num_viz}.png'
        saved_image = Image.open(saved_image_path)

        if wandb.run is not None:
            wandb.log({
                f"RefCOCO_Visualization": wandb.Image(saved_image, caption=inputs['text_caption']),
                f"Mask_1_area": np.sum(top_masks[0].cpu().numpy()) / np.prod(top_masks[0].shape),
                f"Mask_2_area": np.sum(top_masks[1].cpu().numpy()) / np.prod(top_masks[1].shape) if len(top_masks) > 1 else 0,
                f"Mask_3_area": np.sum(top_masks[2].cpu().numpy()) / np.prod(top_masks[2].shape) if len(top_masks) > 2 else 0,
            })

        self.num_viz += 1