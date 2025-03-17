# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import box_area

import ipdb
st = ipdb.set_trace

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def get_3d_box_scanrefer(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def _set_axis_align_bbox(pc):
    # Compute object bounding box.
    # pc has shape (number_of_points, 3)

    max_ = np.max(pc, axis=0)
    min_ = np.min(pc, axis=0)

    cx, cy, cz = (max_ + min_) / 2.0
    lx, ly, lz = max_ - min_

    xmin = cx - lx / 2.0
    xmax = cx + lx / 2.0
    ymin = cy - ly / 2.0
    ymax = cy + ly / 2.0
    zmin = cz - lz / 2.0
    zmax = cz + lz / 2.0

    return np.array([xmin, ymin, zmin, xmax, ymax, zmax])


def _get_bbox_volume(bbox):
    # bbox format [xmin, ymin, zmin, xmax, ymax, zmax]

    return (bbox[:, 3] - bbox[:, 0]) * (bbox[:, 4] - bbox[:, 1]) * (bbox[:, 5] - bbox[:, 2])


def _get_bbox_intersection_volume(bbox_a, bbox_b):
    # bbox format [xmin, ymin, zmin, xmax, ymax, zmax]
    # both arguments have shape (num_boxes, 6)

    xA = np.maximum(bbox_a[:, 0][:, np.newaxis], bbox_b[:, 0][np.newaxis, :])
    yA = np.maximum(bbox_a[:, 1][:, np.newaxis], bbox_b[:, 1][np.newaxis, :])
    zA = np.maximum(bbox_a[:, 2][:, np.newaxis], bbox_b[:, 2][np.newaxis, :])
    xB = np.minimum(bbox_a[:, 3][:, np.newaxis], bbox_b[:, 3][np.newaxis, :])
    yB = np.minimum(bbox_a[:, 4][:, np.newaxis], bbox_b[:, 4][np.newaxis, :])
    zB = np.minimum(bbox_a[:, 5][:, np.newaxis], bbox_b[:, 5][np.newaxis, :])

    return np.clip(xB - xA, 0, None) * np.clip(yB - yA, 0, None) * np.clip(zB - zA, 0, None)


def _get_ious(bbox_a, bbox_b):
    # bbox format [xmin, ymin, zmin, xmax, ymax, zmax]
    # both arguments have shape (num_boxes, 6)

    intersection_vol = _get_bbox_intersection_volume(bbox_a, bbox_b)
    a_vol = _get_bbox_volume(bbox_a)
    b_vol = _get_bbox_volume(bbox_b)
    union_vol = np.expand_dims(a_vol + b_vol, axis=1) - intersection_vol
    return intersection_vol / union_vol


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )

def _iou(box_a, box_b):
    if box_a.shape[-1] == 6:
        iou, union = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(box_a),
            box_cxcyczwhd_to_xyzxyz(box_b)
        )
    else:
        iou, union = box_iou(
            box_cxcywh_to_xyxy(box_a),
            box_cxcywh_to_xyxy(box_b)
        )
    return iou, union


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume

def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    x_c = 0.5 * (x0 + x1)
    y_c = 0.5 * (y0 + y1)
    z_c = 0.5 * (z0 + z1)
    w = x1 - x0
    h = y1 - y0
    d = z1 - z0
    return torch.stack([x_c, y_c, z_c, w, h, d], dim=-1)

def box_cxcywh_to_xyxy(x, protect=False):
    x_c, y_c, w, h = x.unbind(-1)
    if torch.isnan(x).any():
        st()
    assert not torch.isnan(x).any(), x
    if protect:
        w = torch.clamp(w, min=1e-3)
        h = torch.clamp(h, min=1e-3)
        assert (w > 0).all()
        assert (h > 0).all()
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou2d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), boxes1
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def giou(boxes1, boxes2):
    if boxes1.shape[-1] == 6:
        cost_giou = generalized_box_iou3d(
            boxes1,
            boxes2
        )
    else:
        cost_giou = generalized_box_iou2d(
            boxes1,
            boxes2
        )
    return cost_giou

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

