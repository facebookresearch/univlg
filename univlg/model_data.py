# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from .utils.util_video_to_3d import convert_video_instances_to_3d
from detectron2.structures import ImageList
from univlg.data_video.sentence_utils import (
    convert_od_to_grounding_simple,
    sample_classes,
)
from univlg.modeling.backproject.backproject import (
    interpolate_feats_3d,
    multiscsale_voxelize,
    voxel_map_to_source,
    voxelization,
)
from pytorch3d.ops import knn_points
import numpy as np
import torch
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_min

from univlg.global_vars import (
    LEARNING_MAP,
    LEARNING_MAP20,
    MATTERPORT_ALL_CLASSES_TO_21,
    MATTERPORT_LEARNING_MAP,
    S3DIS_NAME_MAP,
    SCANNET200_LEARNING_MAP,
)

from einops import repeat

def expand_tensor(arr, num_frames, orig_bs):
    if isinstance(arr, torch.Tensor):
        if arr.shape[0] == 1:
            return repeat(arr, '() ... -> (orig_bs) ...', orig_bs=orig_bs).contiguous()
        elif arr.shape[0] == num_frames:
            return repeat(arr, 'num_frames ... -> (orig_bs num_frames) ...', orig_bs=orig_bs).contiguous()
        else:
            return arr
    elif isinstance(arr, list):
        return [expand_tensor(a, num_frames, orig_bs) for a in arr]
    elif isinstance(arr, dict):
        return {k: expand_tensor(v, num_frames, orig_bs) for k, v in arr.items()}

def slice_tensor(arr, num_frames, orig_bs):
    if isinstance(arr, torch.Tensor):
        if arr.shape[0] == orig_bs:
            return arr[[0]]
        elif arr.shape[0] == orig_bs * num_frames:
            return arr.view(orig_bs, num_frames, *arr.shape[1:])[0]
        else:
            return arr
    elif isinstance(arr, list):
        return [slice_tensor(a, num_frames, orig_bs) for a in arr]
    elif isinstance(arr, dict):
        return {k: slice_tensor(v, num_frames, orig_bs) for k, v in arr.items()}

def load_scannet_data(
    cfg,
    batched_inputs,
    multiview_data,
    scannet_pc_list=None,
    do_knn=False,
    images=None,
    shape=None,
    is_training=None,
    tokenizer=None,
    device=None,
):
    assert device is not None
    assert tokenizer is not None
    assert is_training is not None
    scennet_pc_processed = []
    scannet_labels_processed = []
    scannet_segments_processed = []
    scannet_idxs = []
    scannet_p2vs = []
    scannet_gt_instances = []

    for i, batched_input in enumerate(batched_inputs):
        if "scannet200" in batched_input["dataset_name"]:
            learning_map = SCANNET200_LEARNING_MAP
        elif "matterport" in batched_input["dataset_name"]:
            learning_map = MATTERPORT_LEARNING_MAP
        elif "scannet" in batched_input["dataset_name"]:
            learning_map = (
                LEARNING_MAP20
                if cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == 20
                else LEARNING_MAP
            )
        elif "s3dis" in batched_input["dataset_name"]:
            learning_map = {k: k for k in S3DIS_NAME_MAP.keys()}
        elif "ai2thor" in batched_input["dataset_name"] or 'arkitscenes' in batched_input["dataset_name"] or 'sam' in batched_input["dataset_name"] or 'coco' in batched_input["dataset_name"] or 'cortex_ref' in batched_input["dataset_name"] or "replica" in batched_input["dataset_name"]:
            # we don't need learning map for these datasets
            pass
        else:
            raise NotImplementedError
        
        if (
            ("ai2thor" in batched_input["dataset_name"]
            or "alfred" in batched_input["dataset_name"] or 'arkitscenes' in batched_input["dataset_name"] 
            or 'sam' in batched_input["dataset_name"] or 'coco' in batched_input["dataset_name"] 
            or 'replica' in batched_input["dataset_name"])
            and 'cortex_ref' not in batched_input["dataset_name"]
        ):
            scennet_pc_processed.append(
                multiview_data["multi_scale_xyz"][-1][i].reshape(-1, 3)
            )
            bs, v, H, W = shape
            h, w = multiview_data["multi_scale_xyz"][-1].shape[2:4]
            images_ = images.tensor.reshape(bs, v, 3, H, W)[i]
            scannet_segments_processed.append(
                torch.arange(len(scennet_pc_processed[-1]), device=device)
            )

            scannet_p2v = voxelization(scennet_pc_processed[-1][None], 0.02)[0]
            scannet_p2vs.append(scannet_p2v)

            target_dict = prepare_targets(
                cfg, [batched_input], images_, valids=None, dataset_names=[batched_input["dataset_name"]],
                is_training=is_training, tokenizer=tokenizer, device=device
            )[0]

            scannet_masks = target_dict["masks"]
            scannet_masks = (
                F.interpolate(
                    scannet_masks.float(), size=(h, w), mode="nearest"
                )
                .to(torch.bool)
                .flatten(1)
            )
            scannet_classes = target_dict["labels"]

            if len(scannet_masks) == 0:
                print("no valid masks, recovering...")
                scannet_masks = torch.zeros(
                    (1, scennet_pc_processed[-1].shape[0]), device=device
                )
                scannet_classes = torch.ones(
                    (1), device=device, dtype=torch.int64
                ) * (len(batched_input["all_classes"]) - 1)

            if cfg.MODEL.OPEN_VOCAB:
                positive_map, positive_map_od, tokens_positive, text_caption = (
                    target_dict["positive_map"],
                    target_dict["positive_map_od"],
                    target_dict["tokens_positive"],
                    target_dict["text_caption"],
                )
                if len(positive_map) == 0:
                    positive_map = torch.zeros(
                        (1, cfg.MODEL.MAX_SEQ_LEN), device=device
                    )
                    positive_map[:, 0] = 1.0
                    positive_map_od = (
                        torch.ones((cfg.MODEL.MAX_SEQ_LEN)) * -1
                    )
                    tokens_positive = [[]]
                    text_caption = ""
            
            batched_input["scannet_coords"] = scennet_pc_processed[-1]
            
        else:
            if scannet_pc_list is not None:
                scannet_pc = scannet_pc_list[i].float()
            else:
                scannet_pc = batched_input["scannet_coords"].to(device).float()
            scannet_labels = batched_input["scannet_labels"].to(device)
            scannet_segments = batched_input["scannet_segments"].to(device)

            if do_knn:
                assert (
                    is_training or cfg.FORCE_SUBSAMPLE
                ), "knn is only used during training - at test time we use full pc"
                our_pc = (
                    multiview_data["multi_scale_xyz"][3][i]
                    .to(device)
                    .flatten(0, 2)
                )
                dists_idxs = knn_points(scannet_pc[None], our_pc[None])
                dists_close = dists_idxs.dists.squeeze() < cfg.KNN_THRESH
                
                scennet_pc_processed.append(scannet_pc[dists_close])
                scannet_labels_processed.append(scannet_labels[dists_close])
                segments = scannet_segments[dists_close]
                scannet_segments_processed.append(segments)
                scannet_idxs.append(dists_close)
            else:
                scennet_pc_processed.append(scannet_pc)
                scannet_labels_processed.append(scannet_labels)
                scannet_segments_processed.append(
                    batched_input["scannet_segments"].to(device)
                )

            # voxelization
            if len(scennet_pc_processed[-1]) == 0:
                scennet_pc_processed[-1] = torch.zeros(
                    1, 3, device=device, dtype=torch.float32
                )
                scannet_p2vs.append(
                    torch.zeros(1, device=device, dtype=torch.int64)
                )
            else:
                scannet_p2v = voxelization(scennet_pc_processed[-1][None], 0.02)[0]
                scannet_p2vs.append(scannet_p2v)

            # extract masks and labels
            target_dict = {}

            if "test" not in batched_input["dataset_name"]:
                unique_instances = torch.unique(scannet_labels_processed[-1][:, 1])
                if len(unique_instances) > 0 and unique_instances[0] == -1:
                    unique_instances = unique_instances[1:]

                if "ref" in batched_input["dataset_name"]:
                    assert len(batched_input['sr3d_data']) == 1
                    target_id = batched_input['sr3d_data'][0]['target_id']
                    anchor_ids = batched_input['sr3d_data'][0]['anchor_ids']
                    relevant_ids = [target_id, *anchor_ids]
                    if not all([relevant_id in unique_instances for relevant_id in relevant_ids]):
                        # sometimes some anchors are very small, just ignore them
                        # the evaluation does not happen on anchor ids, so it's fine
                        keep_mask = torch.tensor([relevant_id in unique_instances for relevant_id in relevant_ids])
                        relevant_ids = [relevant_id for relevant_id in relevant_ids if relevant_id in unique_instances]
                        batched_input['sr3d_data'][0]['positive_map'] = batched_input['sr3d_data'][0]['positive_map'][keep_mask]
                        batched_input['sr3d_data'][0]['tokens_positive'] = [batched_input['sr3d_data'][0]['tokens_positive'][i] for i in range(len(batched_input['sr3d_data'][0]['tokens_positive'])) if keep_mask[i]]
                    unique_instances = torch.tensor(relevant_ids)

                num_unique_instances = len(unique_instances)
                scannet_masks = []
                scannet_classes = []
                if num_unique_instances == 0:
                    num_unique_instances = 1

                # for the referit3d setting that allows GT mask proposals as input
                if cfg.USE_GT_MASKS:
                    all_scannet_masks = scannet_labels_processed[-1][:, 1]
                    if 'ref' in batched_input['dataset_name']:
                        all_relevant_ids = torch.tensor(relevant_ids)

                all_det_relevant_ids = []
                for k in range(num_unique_instances):
                    if len(unique_instances) == 0:
                        num_unique_instances = 0
                        break

                    if len(scannet_labels_processed) == 0:
                        print(f"scannet_labels_processed is empty, {scannet_labels_processed}, {unique_instances}, {scannet_labels_processed}")
                    scannet_mask = (
                        scannet_labels_processed[-1][:, 1] == unique_instances[k]
                    )
                    if len(scannet_labels_processed[-1]) == 0 or len(scannet_labels_processed[-1][:, 0]) == 0 or len(scannet_labels_processed[-1][:, 0][scannet_mask]) == 0:
                        print(f"scannet_mask is empty, {scannet_mask}, {scannet_labels_processed[-1]}, {unique_instances[k]}, {scannet_labels_processed[-1][:, 1]}")

                    class_label = scannet_labels_processed[-1][:, 0][scannet_mask][0].item()

                    if 'ref' not in batched_input['dataset_name']:
                        if class_label not in learning_map:
                            continue
                        class_label = learning_map[class_label]
                        if class_label == 0:
                            continue

                        if cfg.MATTERPORT_ALL_CLASSES_TO_21:
                            class_label = MATTERPORT_ALL_CLASSES_TO_21[class_label]

                    all_det_relevant_ids.append(class_label)

                    scannet_masks.append(scannet_mask)
                    scannet_classes.append(class_label - 1)

                if cfg.USE_GT_MASKS and 'ref' not in batched_input['dataset_name']:
                    if len(all_det_relevant_ids) == 0:
                        all_det_relevant_ids = [0]
                    all_relevant_ids = torch.tensor(all_det_relevant_ids)

                if len(scannet_masks) == 0:
                    print("no valid masks, recovering...")
                    scannet_masks = torch.zeros(
                        (1, scennet_pc_processed[-1].shape[0]), device=device
                    )
                    scannet_classes = (
                        torch.ones((1), device=device, dtype=torch.int64)
                        * batched_input['num_classes']
                        - 1
                    )
                    if cfg.MODEL.OPEN_VOCAB:
                        positive_map = torch.zeros(
                            (1, cfg.MODEL.MAX_SEQ_LEN), device=device
                        )
                        positive_map[:, 0] = 1.0
                        positive_map_od = (
                            torch.ones((cfg.MODEL.MAX_SEQ_LEN)) * -1
                        )
                        tokens_positive = [[]]
                        text_caption = ""

                else:
                    scannet_masks = torch.stack(scannet_masks, dim=0)
                    scannet_classes = torch.tensor(
                        scannet_classes, device=device, dtype=torch.int64
                    )
                    if cfg.MODEL.OPEN_VOCAB:
                        if "ref" not in batched_input['dataset_name']:
                            if cfg.RANDOM_SELECT_CLASSES and is_training:
                                assert not cfg.DISABLE_SHUFFLE, "shuffle is important for unbiased detection prompts"
                                scannet_masks, scannet_classes, all_classes = sample_classes(
                                    scannet_masks.clone(), scannet_classes.clone(),
                                    copy.deepcopy(batched_input['all_classes']), cfg.MAX_CLASSES, 
                                    cfg.RETURN_ORIGINAL_PROB
                                )
                            else:
                                all_classes = batched_input['all_classes']

                            (
                                positive_map,
                                positive_map_od,
                                tokens_positive,
                                text_caption,
                            ) = convert_od_to_grounding_simple(
                                scannet_classes.tolist(),
                                all_classes,
                                disable_shuffle=cfg.DISABLE_SHUFFLE
                                or (not is_training),
                                separation_tokens=". ",
                                tokenizer=tokenizer,
                                max_query_len=cfg.MODEL.MAX_SEQ_LEN,
                                add_detection_prompt=cfg.ADD_DETECTION_PRETEXT
                            )
                            positive_map = positive_map.to(device)

                        else:
                            positive_map, positive_map_od, tokens_positive, text_caption = batched_input['sr3d_data'][0]['positive_map'], \
                                batched_input['sr3d_data'][0]['positive_map_od'], batched_input['sr3d_data'][0]['tokens_positive'], batched_input['sr3d_data'][0]['text_caption']
                            positive_map = positive_map.to(device)
            else:
                positive_map, positive_map_od, tokens_positive, text_caption = batched_input['sr3d_data'][0]['positive_map'], \
                    batched_input['sr3d_data'][0]['positive_map_od'], batched_input['sr3d_data'][0]['tokens_positive'], batched_input['sr3d_data'][0]['text_caption']
                positive_map = positive_map.to(device)        

        if cfg.USE_GT_MASKS:
            all_scannet_masks_no_downsample = all_scannet_masks.clone()
            scannet_p2v = scannet_p2vs[-1]
            all_scannet_masks = scatter_min(
                all_scannet_masks, scannet_p2vs[-1], dim=0
            )[0]
            unique_all_scannet_masks, inverse_indices = torch.unique(
                all_scannet_masks, return_inverse=True
            )

            new_all_scannet_masks = torch.arange(
                unique_all_scannet_masks.shape[0], device=device
            )[inverse_indices]
            all_scannet_masks = new_all_scannet_masks

            reduced_scene_id_to_original_id = unique_all_scannet_masks
            all_relevant_ids = torch.tensor(
                [(unique_all_scannet_masks == relevant_id).nonzero().item() if relevant_id in unique_all_scannet_masks else 0 for relevant_id in all_relevant_ids],
            )

        # voxelized segments
        if cfg.USE_SEGMENTS:
            if len(scannet_segments_processed[-1]) == 0:
                scannet_segments = torch.zeros(1, device=device)
            else:
                scannet_segments = scannet_segments_processed[-1]
            scannet_p2v = scannet_p2vs[-1]

            
            fake_segments = 'scannetpp' in batched_input['dataset_name'] or \
                    'coco' in batched_input['dataset_name'] or \
                    'cortex_ref' in batched_input['dataset_name'] or \
                    'sam' in batched_input['dataset_name'] or \
                    'arkitscenes' in batched_input['dataset_name'] or \
                    'ai2thor' in batched_input['dataset_name'] or \
                    's3dis' in batched_input['dataset_name'] or \
                    'paco' in batched_input['dataset_name'] or \
                    'replica' in batched_input['dataset_name']

            if not fake_segments:
                scannet_segments = scatter_min(
                    scannet_segments, scannet_p2vs[-1], dim=0
                )[0]
                unique_segments, inverse_indices = torch.unique(
                    scannet_segments, return_inverse=True
                )
                new_segments = torch.arange(
                    unique_segments.shape[0], device=device
                )[inverse_indices]
                segments = new_segments
            else:
                segments = torch.arange(scannet_p2vs[-1].max() + 1, device=device)
            
            scannet_segments_processed[-1] = segments

            if "test" not in batched_input["dataset_name"]:
                segment_mask = scatter_mean(
                    scannet_masks.float(), scannet_p2v[None], dim=1
                )
                segment_mask = (
                    scatter_mean(segment_mask, segments[None], dim=1) > 0.5
                )
                voxel_masks = segment_mask

        else:
            if "test" not in batched_input["dataset_name"]:
                voxel_masks = (
                    scatter_mean(
                        scannet_masks.float(), scannet_p2vs[-1][None], dim=1
                    )
                    > 0.5
                )

        # for some datasets like ai2thor and s3dis, the segments are fake
        # and we would like to do subsampling there
        if cfg.USE_SEGMENTS:
            fake_segments = segments.max() == len(segments) - 1 and (
                segments.max() != 0
            )
            # print(batched_input['dataset_name'], fake_segments)
            if cfg.HIGH_RES_SUBSAMPLE and is_training and fake_segments:
                idx = torch.randperm(
                    voxel_masks.shape[1], device=voxel_masks.device
                )[: cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS]
                voxel_masks = voxel_masks[:, idx]
                pc = scatter_mean(scennet_pc_processed[-1], scannet_p2vs[-1], dim=0)
                
                pc = pc[idx]
                # scannet_masks = scannet_masks[:, idx]
                scannet_masks = voxel_masks
                scennet_pc_processed[-1] = pc
                scannet_p2vs[-1] = torch.arange(len(idx)).to(scannet_p2vs[-1])
                segments = torch.arange(len(idx)).to(scannet_segments_processed[-1])
                scannet_segments_processed[-1] = segments
                empty_masks = voxel_masks.sum(1) == 0
                voxel_masks = voxel_masks[~empty_masks]
                scannet_classes = scannet_classes[~empty_masks]
                scannet_masks = scannet_masks[~empty_masks]

                if cfg.MODEL.OPEN_VOCAB:
                    positive_map = positive_map[~empty_masks]
                    
        if "test" not in batched_input["dataset_name"]:
            target_dict = {
                "masks": scannet_masks.float(),
                "labels": scannet_classes,
                "p2v": scannet_p2vs[-1],
                "max_valid_points": scannet_masks.shape[1],
                "segments": segments if cfg.USE_SEGMENTS else None,
                "segment_mask": voxel_masks if cfg.USE_SEGMENTS else None,
                "voxel_masks": voxel_masks,
                "positive_map": positive_map
                if cfg.MODEL.OPEN_VOCAB
                else None,  # cfg.MODEL.MAX_SEQ_LEN
                "tokens_positive": tokens_positive
                if cfg.MODEL.OPEN_VOCAB
                else None,  # list
                "positive_map_od": positive_map_od
                if cfg.MODEL.OPEN_VOCAB
                else None,  # N X cfg.MODEL.MAX_SEQ_LEN
                "text_caption": text_caption if cfg.MODEL.OPEN_VOCAB else None,
                "all_scannet_masks": all_scannet_masks if cfg.USE_GT_MASKS else None,
                "all_relevant_ids": all_relevant_ids if cfg.USE_GT_MASKS else None,
                "all_scannet_masks_no_downsample": all_scannet_masks_no_downsample if cfg.USE_GT_MASKS else None,
                "reduced_scene_id_to_original_id": reduced_scene_id_to_original_id if cfg.USE_GT_MASKS else None,
                'dataset_name': batched_input['dataset_name'],
            }
        else:
            target_dict = {
                "p2v": scannet_p2vs[-1],
                "max_valid_points": scannet_p2vs[-1].shape[0],
                "segments": segments if cfg.USE_SEGMENTS else None,
                "positive_map": positive_map
                if cfg.MODEL.OPEN_VOCAB and not cfg.DETIC
                else None,  # cfg.MODEL.MAX_SEQ_LEN
                "tokens_positive": tokens_positive
                if cfg.MODEL.OPEN_VOCAB and not cfg.DETIC
                else None,  # list
                "positive_map_od": positive_map_od
                if cfg.MODEL.OPEN_VOCAB and not cfg.DETIC
                else None,  # N X cfg.MODEL.MAX_SEQ_LEN
                "text_caption": text_caption if cfg.MODEL.OPEN_VOCAB else None,
                "all_scannet_masks": all_scannet_masks if cfg.USE_GT_MASKS else None,
                "all_relevant_ids": all_relevant_ids if cfg.USE_GT_MASKS else None,
                "all_scannet_masks_no_downsample": all_scannet_masks_no_downsample if cfg.USE_GT_MASKS else None,
                "reduced_scene_id_to_original_id": reduced_scene_id_to_original_id if cfg.USE_GT_MASKS else None,
            }
        scannet_gt_instances.append(target_dict)

    # get max points
    valid_points = [x.shape[0] for x in scennet_pc_processed]
    max_points = max(valid_points)

    # batch the points and labels
    scannet_pc_batched = torch.zeros(
        len(scennet_pc_processed), max_points, 3, device=device
    )
    scannet_p2v_batched = torch.zeros(
        len(scannet_p2vs), max_points, dtype=torch.int64, device=device
    )

    valid_segment_points = [x.shape[0] for x in scannet_segments_processed]
    max_segment_points = max(valid_segment_points)
    voxel_sizes = np.array([x.max().item() for x in scannet_p2vs])
    max_voxel_size = voxel_sizes == voxel_sizes.max()
    max_voxel_valid_points = np.array(valid_points)[np.nonzero(max_voxel_size)[0]]

    if (max_voxel_valid_points < max_points).any():
        max_segment_points += 1  # extra segment due to voxel padding
    scannet_segments_batched = torch.zeros(
        len(scannet_segments_processed),
        max_segment_points,
        dtype=torch.int64,
        device=device,
    )

    for j, (pc, p2v, segments) in enumerate(
        zip(
            scennet_pc_processed,
            scannet_p2vs,
            scannet_segments_processed,
        )
    ):
        scannet_pc_batched[j, : pc.shape[0]] = pc
        scannet_pc_batched[j, pc.shape[0] :] = -10

        scannet_segments_batched[j, : segments.shape[0]] = segments
        if segments.shape[0] != 0:
            scannet_segments_batched[j, segments.shape[0] :] = segments.max() + 1

        scannet_p2v_batched[j, : p2v.shape[0]] = p2v
        if p2v.shape[0] != 0:
            scannet_p2v_batched[j, p2v.shape[0] :] = p2v.max() + 1

    scannet_all_masks_batched = None
    if cfg.USE_GT_MASKS:
        scannet_all_masks_batched = torch.zeros(
            len(scannet_gt_instances),
            max_segment_points,
            dtype=torch.long,
            device=device,
        )

        true_all_relevant_ids = []
        # Order matters for these 2 loops
        for j, gt_instances in enumerate(scannet_gt_instances):
            true_all_relevant_ids.append(gt_instances["all_relevant_ids"].clone())
            _ids_in_masks = set(gt_instances["all_scannet_masks"].long().unique().tolist())
            _all_ids = set(gt_instances["all_relevant_ids"].long().unique().tolist())
            _all_ids = torch.tensor([x for x in _all_ids], device=device, dtype=torch.long)
            not_found_ids = torch.tensor([x.item() for x in gt_instances["all_relevant_ids"] if x.item() not in _ids_in_masks], device=device, dtype=torch.long)
            if not_found_ids.shape[0] > 0:
                gt_instances["all_scannet_masks"][-_all_ids.shape[0]:] = _all_ids
                print(f"Setting last {not_found_ids.shape[0]} ids in scannet_all_masks to the wrong ids., {_all_ids}")

            _ids_in_masks = set(gt_instances["all_scannet_masks"].long().unique().tolist())
            not_found_ids = torch.tensor([x.item() for x in gt_instances["all_relevant_ids"] if x.item() not in _ids_in_masks], device=device, dtype=torch.long)
            if not_found_ids.shape[0] > 0:
                print(f"WARNING!!!! not found {not_found_ids}")
            if len(gt_instances["all_scannet_masks"]) == 0:
                _num = 1
            else:
                _num = (gt_instances["all_scannet_masks"].max() + 1).item()
            if gt_instances["all_relevant_ids"].shape[0] > 0:
                assert gt_instances["all_relevant_ids"].max() < _num, f"all relevant ids (max: {gt_instances['all_relevant_ids'].max()}) must be less than the number of classes: {_num}"
                gt_instances["all_relevant_ids"] = F.one_hot(gt_instances["all_relevant_ids"], num_classes=_num)
            else:
                gt_instances["all_relevant_ids"] = torch.zeros((0, _num), device=device, dtype=bool)

        for j, gt_instances in enumerate(scannet_gt_instances):
            scannet_all_masks_batched[j, : gt_instances["all_scannet_masks"].shape[0]] = gt_instances["all_scannet_masks"]
            if gt_instances["all_scannet_masks"].shape[0] != 0:
                scannet_all_masks_batched[j, gt_instances["all_scannet_masks"].shape[0] :] = gt_instances["all_scannet_masks"].max() + 1

        true_all_relevant_ids = torch.cat(true_all_relevant_ids, dim=0)

    return (
        scannet_pc_batched,
        scannet_p2v_batched,
        scannet_gt_instances,
        scannet_idxs,
        scannet_segments_batched,
        scannet_all_masks_batched,
    )


def prepare_targets(cfg, targets, images, valids=None, cheating=False, dataset_names=None, is_training=None, tokenizer=None, device=None):
    assert is_training is not None
    assert tokenizer is not None
    assert device is not None
    if isinstance(images, ImageList):
        h_pad, w_pad = images.tensor.shape[-2:]
    else:
        h_pad, w_pad = images.shape[-2:]
    gt_instances = []
    for targets_per_video in targets:
        num_frames = len(targets_per_video["images"])

    sample = "instances_all" if not cheating else "instances_all_full"
    for i, targets_per_video in enumerate(targets):
        target_dict = convert_video_instances_to_3d(
            targets_per_video[sample],
            num_frames,
            h_pad,
            w_pad,
            device,
            multiplier=targets_per_video["multiplier"],
        )

        if cfg.MODEL.OPEN_VOCAB:
            if 'ref' in dataset_names[i]:
                target_dict["positive_map"] = targets_per_video['positive_map'].to(device)
                target_dict["tokens_positive"] = targets_per_video['tokens_positive']
                target_dict["positive_map_od"] = targets_per_video['positive_map_od']
                target_dict["text_caption"] = targets_per_video['text_caption']
            else:
                if cfg.RANDOM_SELECT_CLASSES and is_training:
                    assert not cfg.DISABLE_SHUFFLE, "shuffle is important for unbiased detection prompts"
                    masks, classes, all_classes = sample_classes(
                        target_dict['masks'].clone(), target_dict['labels'].clone(),
                        copy.deepcopy(targets[i]["all_classes"]), cfg.MAX_CLASSES, 
                        cfg.RETURN_ORIGINAL_PROB
                    )
                    target_dict['masks'] = masks
                    target_dict['labels'] = classes
                else:
                    all_classes = targets[i]["all_classes"]
                (
                    positive_map,
                    positive_map_od,
                    tokens_positive,
                    text_caption,
                ) = convert_od_to_grounding_simple(
                    target_dict["labels"].tolist(),
                    all_classes,
                    disable_shuffle=cfg.DISABLE_SHUFFLE
                    or (not is_training),
                    separation_tokens=". ",
                    tokenizer=tokenizer,
                    max_query_len=cfg.MODEL.MAX_SEQ_LEN,
                    add_detection_prompt=cfg.ADD_DETECTION_PRETEXT
                )
                positive_map = positive_map.to(device)
                target_dict["positive_map"] = positive_map
                target_dict["tokens_positive"] = tokens_positive
                target_dict["positive_map_od"] = positive_map_od
                target_dict["text_caption"] = text_caption
        
        target_dict['dataset_name'] = dataset_names[i] if dataset_names is not None else None

        if valids is not None:
            target_dict.update({"valids": valids[i]})

        gt_instances.append(target_dict)

    return gt_instances