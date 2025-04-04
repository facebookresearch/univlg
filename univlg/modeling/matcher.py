# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import ipdb
import torch
import torch.nn.functional as F
from detectron2.projects.point_rend.point_features import point_sample
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from univlg.utils.bbox_utils import box_xyzxyz_to_cxcyczwhd, giou

st = ipdb.set_trace


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class VideoHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_mask: float = 1,
        cost_dice: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_mask_det: float = 1,
        cost_dice_det: float = 1,
        cost_class_det: float = 1,
        num_points: int = 0,
        supervise_sparse: bool = False,
        cfg=None,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        self.cost_mask_det = cost_mask_det
        self.cost_dice_det = cost_dice_det
        self.cost_class_det = cost_class_det

        self.cfg = cfg

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "all costs cant be 0"

        self.num_points = num_points
        self.supervise_sparse = supervise_sparse

    def compute_box_cost(self, out_bbox, tgt_bbox):
        """
        Compute the bounding box loss
        out_bbox and target_bbox are both in the format [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        out_bbox_ = box_xyzxyz_to_cxcyczwhd(out_bbox)
        tgt_bbox_ = box_xyzxyz_to_cxcyczwhd(tgt_bbox)
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
        try:
            cost_giou = -giou(out_bbox, tgt_bbox)
        except:
            print("some error in giou, handling for now...")
            cost_giou = 0.0

        if torch.isnan(cost_giou).any() or torch.isinf(cost_giou).any():
            print("nan in giou")
            cost_giou = torch.tensor(0.0, requires_grad=True).to(cost_giou)

        if torch.isnan(cost_bbox).any() or torch.isinf(cost_bbox).any():
            print("nan in bbox")
            cost_bbox = torch.tensor(0.0, requires_grad=True).to(cost_bbox)
        return cost_bbox, cost_giou

    @torch.no_grad()
    def memory_efficient_forward_3d(self, outputs, targets, actual_decoder_3d=False):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_masks"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].sigmoid()
            positive_map = targets[b]["positive_map"]
            positive_map = positive_map[:, : out_prob.shape[1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))
    
            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]

            # gt masks are already padded when preparing target
            
            if self.cfg.USE_GT_MASKS:
                tgt_mask = targets[b]["all_relevant_ids"].to(out_mask)
            elif self.cfg.USE_SEGMENTS:
                tgt_mask = targets[b]["segment_mask"].to(out_mask)
            elif self.cfg.INPUT.VOXELIZE:
                tgt_mask = targets[b]["voxel_masks"].to(
                    out_mask
                )  # [num_gts, T, H_pred, W_pred]
            else:
                tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, : tgt_mask.shape[1]]
            if tgt_mask.shape[1] > out_mask.shape[1]:
                # assert (tgt_mask[:, out_mask.shape[1]:].sum() == 0).item()
                print("WARNING: Target masks have more relevant ids than predicted masks. Truncating.")
                tgt_mask = tgt_mask[:, : out_mask.shape[1]]

            if out_mask.shape[1] > self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS:
                # randomly subsample points
                idx = torch.randperm(out_mask.shape[1], device=out_mask.device)[
                    : self.cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS
                ]
                out_mask = out_mask[:, idx]
                tgt_mask = tgt_mask[:, idx]

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                if tgt_mask.shape[1] == 0:
                    cost_mask = 0
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            if self.cfg.USE_BOX_LOSS and actual_decoder_3d and not ('ref' not in targets[b]['dataset_name'] and self.cfg.NO_BOX_COST_IN_MATCHING_FOR_DET):
                cost_bbox, cost_giou = self.compute_box_cost(
                    outputs['pred_boxes'][b], targets[b]['boxes']
                )
            else:
                cost_bbox = 0
                cost_giou = 0

            if self.cfg.USE_DIFF_DET_REF_MATCHING and 'ref' not in targets[b]['dataset_name']:
                mask_weight, class_weight, dice_weight = self.cost_mask_det, self.cost_class_det, self.cost_dice_det
            else:
                mask_weight, class_weight, dice_weight = self.cost_mask, self.cost_class, self.cost_dice
            
            try:
                # Final cost matrix
                C = (
                    mask_weight * cost_mask
                    + class_weight * cost_class
                    + dice_weight * cost_dice
                    + self.cost_bbox * cost_bbox
                    + self.cost_giou * cost_giou
                )
            except Exception:
                st()
            C = C.reshape(num_queries, -1).cpu()
            try:
                indices.append(linear_sum_assignment(C))
            except Exception:
                # check for nanas
                if torch.isnan(C).any():
                    print("nan in cost matrix")
                if torch.isnan(cost_mask).any():
                    print("nan in cost mask")
                if torch.isnan(cost_class).any():
                    print("nan in cost class")
                if torch.isnan(cost_dice).any():
                    print("nan in cost dice")
                if torch.isnan(out_mask).any():
                    print("nan in out mask")
                print(tgt_mask.sum(-1))
                st()
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        
    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].sigmoid()
            positive_map = targets[b]["positive_map"]
            positive_map = positive_map[:, : out_prob.shape[1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))

            out_mask = outputs["pred_masks"][b]  # [num_queries, T, H_pred, W_pred]

            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gts, T, H_pred, W_pred]

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).flatten(1)

            if self.supervise_sparse:
                valid = targets[b]["valids"].to(tgt_mask)[None]
                valid_mask = (
                    point_sample(
                        valid,
                        point_coords.repeat(valid.shape[0], 1, 1),
                        align_corners=False,
                    )
                    .flatten(1)
                    .bool()
                )
                tgt_mask[~valid_mask.repeat(tgt_mask.shape[0], 1)] = 0
                out_mask[~valid_mask.repeat(out_mask.shape[0], 1)] = out_mask.min()

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]   

    @torch.no_grad()
    def forward(self, outputs, targets, decoder_3d=False, actual_decoder_3d=False):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if decoder_3d:
            return self.memory_efficient_forward_3d(outputs, targets, actual_decoder_3d)
        else:
            return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
