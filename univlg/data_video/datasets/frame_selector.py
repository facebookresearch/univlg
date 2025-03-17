# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import cycle
import torch
import pickle
import numpy as np
import os
from typing import Optional
import ipdb
from pathlib import Path
st = ipdb.set_trace

def select_m_items(d, m, require_n_target_frames: int = 0, at_most_n_relevant_frames: Optional[int] = None):
    key_iterators = [(idx, key, iter(value)) for idx, (key, value) in enumerate(d)]
    cyclic_key_iterators = cycle(key_iterators)
    selected = set()
    counts = [0] * len(key_iterators)

    if require_n_target_frames > 0:
        for i in range(require_n_target_frames):
            if i >= len(d[0][1]):
                if i == 0:
                    print(f"Required {require_n_target_frames} frames, but only {len(d[0][1])} frames available")
                break
            selected.add(d[0][1][i])
            counts[0] += 1

    while len(selected) < m:
        try:
            idx, current_key, current_iterator = next(cyclic_key_iterators)
            item = next(current_iterator)
            while item in selected:
                item = next(current_iterator)

            selected.add(item)
            counts[idx] += 1

            if at_most_n_relevant_frames is not None and counts[idx] >= at_most_n_relevant_frames:
                # print(f"Selected {at_most_n_relevant_frames} frames for object {current_key}, skipping further frames")
                break

        except StopIteration:
            key_iterators = [(idx, k, it) for idx, k, it in key_iterators if k in d]
            cyclic_key_iterators = cycle(key_iterators)
            if not key_iterators:
                break

    return list(selected)


scene_object_counts = None
PRECOMPUTED_SCANNET_PATH = Path(os.getenv('PRECOMPUTED_SCANNET_PATH'))

def get_scene_object_pixel_counts(scan_id: str):
    global scene_object_counts
    if scene_object_counts is None:
        scene_object_counts = torch.load(PRECOMPUTED_SCANNET_PATH / 'scannet_object_id_frame_map.pth', weights_only=False)
    return scene_object_counts[scan_id]

def furthest_point_sample(xyz, npoint, initial_centroids=None):
    device = xyz.device
    N, D = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10

    if initial_centroids is not None:
        num_initial = len(initial_centroids)
        centroids[:num_initial] = torch.tensor(initial_centroids, dtype=torch.long, device=device)
        for i in range(num_initial):
            centroid = xyz[centroids[i], :].view(1, D)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
        start_idx = num_initial
    else:
        start_idx = 0
        farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)

    for i in range(start_idx, npoint):
        if initial_centroids is None or i > 0:
            farthest = torch.max(distance, dim=0)[1]
        else:
            farthest = centroids[i]
        centroids[i] = farthest
        centroid = xyz[farthest, :].view(1, D)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]

    return centroids
    
from collections import defaultdict
import hashlib
from pathlib import Path

text_data, image_data, scene_object_frame_map_filenames = None, None, None
scene_pixel_dict_with_dataloader_idxs = defaultdict(dict)
scene_full_scene_file_idxs_to_dataloader_idxs = defaultdict(dict)
scene_full_scene_file_to_dataloder_ordered_list = defaultdict(list)

def get_data(get_clip_data: bool = True):
    global text_data, image_data, scene_object_frame_map_filenames, scene_pixel_dict_with_dataloader_idxs, scene_full_scene_file_to_dataloder_ordered_list
    
    if text_data is None and get_clip_data:
        text_data = torch.load(PRECOMPUTED_SCANNET_PATH / 'scannet_object_id_frame_map_clip_text.pth', weights_only=False)
    if image_data is None and get_clip_data:
        image_data = torch.load(PRECOMPUTED_SCANNET_PATH / 'scannet_object_id_frame_map_clip.pth', weights_only=False)
    if scene_object_frame_map_filenames is None:
        scene_object_frame_map_filenames = torch.load(PRECOMPUTED_SCANNET_PATH / 'scannet_object_id_frame_map_filenames.pth', weights_only=False)

def get_dataloader_filename_mapping(dataset_dict, scan_id):
    global text_data, image_data, scene_object_frame_map_filenames, scene_pixel_dict_with_dataloader_idxs, scene_full_scene_file_to_dataloder_ordered_list

    pixel_dict = get_scene_object_pixel_counts(scan_id=scan_id)
    dataloader_filenames = [Path(x).name for x in dataset_dict['file_names']]
    scene_filenames = scene_object_frame_map_filenames[scan_id.removeprefix('scene')]

    if scan_id not in scene_full_scene_file_to_dataloder_ordered_list:
        full_scene_file_idxs = []
        scene_full_scene_file_to_dataloder_ordered_list[scan_id] = []
        for dataloader_idx, dataloader_filename in enumerate(dataloader_filenames):
            full_scene_file_idx = scene_filenames.index(dataloader_filename)
            assert full_scene_file_idx != -1, f"Full scene file index not found for {dataloader_filename} in scene {scan_id}"
            full_scene_file_idxs.append(full_scene_file_idx)
            scene_full_scene_file_idxs_to_dataloader_idxs[scan_id][full_scene_file_idx] = dataloader_idx
            scene_full_scene_file_to_dataloder_ordered_list[scan_id].append(full_scene_file_idx)

    if scan_id not in scene_pixel_dict_with_dataloader_idxs:
        for object_id, (indices, counts) in pixel_dict.items():
            new_indices = []
            new_counts = []
            for scene_file_idx, count in zip(indices.tolist(), counts.tolist()): # obj_idx: tensor (scene_index, counts), n, 2
                if scene_file_idx in scene_full_scene_file_idxs_to_dataloader_idxs[scan_id]:
                    new_indices.append(scene_full_scene_file_idxs_to_dataloader_idxs[scan_id][scene_file_idx])
                    new_counts.append(count)
            new_indices = torch.tensor(new_indices, dtype=indices.dtype, device=indices.device)
            new_counts = torch.tensor(new_counts, dtype=counts.dtype, device=counts.device)
            assert len(new_indices) == len(new_counts)
            scene_pixel_dict_with_dataloader_idxs[scan_id][object_id] = (new_indices, new_counts)

def normalize_caption(caption: str):
    return caption.lower().replace(".", "").replace(",", "").replace(" ", "")

def get_relevant_clip_frames(
    scan_id: str,
    relevant_ids: list[int],
    num_total_scene_frames: int,
    num_frames: int,
    shuffle: bool = False,
    fraction_relevant_frames: float = 0.5,
    require_n_target_frames: int = 0,
    at_most_n_relevant_frames: Optional[int] = None,
    force_full_random_relevant_frames: bool = False,
    num_important_relevant_ids: Optional[int] = None,
    dataset_dict=None,
    clip_only: bool = False,
    **kwargs
):
    global text_data, image_data, scene_object_frame_map_filenames, scene_pixel_dict_with_dataloader_idxs, scene_full_scene_file_to_dataloder_ordered_list
    try:
        get_data()
        get_dataloader_filename_mapping(dataset_dict, scan_id)
    except Exception as e:
        print(f"Failed to get dataloader filename mapping for {scan_id} with error {e}")
        raise e

    pixel_dict = scene_pixel_dict_with_dataloader_idxs[scan_id]
    object_ids: list[int] = relevant_ids
    object_lists = []
    heavy_sample_relevant_ids = []
    
    for i, object_id in enumerate(object_ids):
        try:
            indices, counts = pixel_dict[object_id]
        except:
            print(f"Object {i}, {object_id} not found in scene {scan_id}, keys: {pixel_dict.keys()}")
            if num_important_relevant_ids is not None and i < num_important_relevant_ids:
                heavy_sample_relevant_ids = True
            continue

        if len(indices) == 0:
            print(f"Object {i}, {object_id} has no frames in scene {scan_id}")
            if num_important_relevant_ids is not None and i < num_important_relevant_ids:
                heavy_sample_relevant_ids = True
            continue
        
        if shuffle:
            if force_full_random_relevant_frames:
                shuffle_idx = torch.randperm(len(counts))
            else:
                shuffle_idx = torch.multinomial(
                    counts.float().pow(0.25), len(counts), replacement=False
                )
            indices, counts = indices[shuffle_idx], counts[shuffle_idx]

        object_lists.append((object_id, indices.tolist()))

    num_to_select = min(num_frames, len(object_lists))
    if num_important_relevant_ids is not None:
        num_to_select = min(num_to_select, num_important_relevant_ids)

    if num_frames <= num_to_select:
        print(f"Selecting {num_to_select} frames for {scan_id} with {len(object_lists)} objects")
    
    try:
        if heavy_sample_relevant_ids or clip_only:
            frame_ids = []
        else:
            frame_ids = select_m_items(object_lists, num_to_select)
    except Exception as e:
        print(f"Failed to select frames for {scan_id} with error {e}")
        frame_ids = []

    try:
        sr3d_data = kwargs.get('sr3d_data', None)
        assert sr3d_data is not None
        _txt = sr3d_data.get("original_utterance", sr3d_data.get("utterance", sr3d_data.get("description", sr3d_data.get("question", None))))
        if _txt is None:
            print(f"Text is None for {scan_id}, {sr3d_data.keys()}")
            _txt = "An indoor scene."
        text_caption = normalize_caption(_txt)
        text_hash = hashlib.md5(text_caption.encode()).hexdigest()
        image_embedding = image_data[scan_id] if scan_id in image_data else None
        if image_embedding is not None:
            image_embedding = image_embedding[torch.tensor(scene_full_scene_file_to_dataloder_ordered_list[scan_id])]
            assert image_embedding.shape[0] == num_total_scene_frames, f"Image embedding shape {image_embedding.shape[0]} does not match number of scene frames {num_total_scene_frames}"
            if text_hash in text_data['hashes']:
                text_embedding_index = text_data['hashes'][text_hash]
                text_embedding = torch.from_numpy(text_data['embeddings'][text_embedding_index])
                
                top_values, top_indices = (text_embedding @ image_embedding.T).sort(descending=True)
                top_values = top_values[:-num_total_scene_frames//2]
                top_indices = top_indices[:-num_total_scene_frames//2]

                frame_ids_set = set(frame_ids)
                valid_mask = (top_values >= 0) & torch.isfinite(top_values) & torch.tensor([idx not in frame_ids_set for idx in top_indices])
                top_values = top_values[valid_mask]
                top_indices = top_indices[valid_mask]

                if heavy_sample_relevant_ids and not clip_only:
                    num_clip_frames = (num_frames - len(frame_ids))
                else:
                    num_clip_frames = int((num_frames - len(frame_ids)) / 3)
                    num_clip_frames += (num_to_select - len(frame_ids))
                    num_clip_frames = min(num_clip_frames, num_frames - len(frame_ids))

                if num_clip_frames > 0:
                    sampled_indices = torch.multinomial(top_values, num_samples=num_clip_frames, replacement=num_clip_frames > top_values.shape[0])
                    clip_frame_ids = top_indices[sampled_indices].tolist()
                else:
                    clip_frame_ids = []

                frame_ids = list(set(frame_ids + clip_frame_ids))
            else:
                print(f"Text hash {text_hash} not found in text data, {text_caption}. Using FPS only, image_embedding: {image_embedding is not None}")

            if not (image_embedding.shape[0] == num_total_scene_frames):
                print(f"Image embedding shape {image_embedding.shape[0]} does not match number of scene frames {num_total_scene_frames}")

            if len(frame_ids) < num_frames:
                frame_ids = furthest_point_sample(image_embedding, num_frames, initial_centroids=frame_ids)

    except Exception as e:
        import traceback
        print(f"Failed to select frames for {scan_id} with error {e}")
        print("Traceback:")
        print(traceback.format_exc())

    assert all([x < num_total_scene_frames for x in frame_ids]), f"Frame ids {frame_ids} have values greater than {num_total_scene_frames}"
    frame_ids = list(set(frame_ids))

    if len(frame_ids) < num_frames:
        unchosen_frame_ids = np.setdiff1d(np.arange(num_total_scene_frames), np.array(frame_ids))
        if len(unchosen_frame_ids) == 0:
            unchosen_frame_ids = np.arange(num_total_scene_frames)
        random_frame_ids = np.random.choice(unchosen_frame_ids, size=num_frames - len(frame_ids), replace=True)
        frame_ids.extend(random_frame_ids)

    if len(frame_ids) > num_frames:
        print(f"Selected {len(frame_ids)} frames for {scan_id}, desired: {num_frames}")
        frame_ids = frame_ids[:num_frames]

    return frame_ids


def get_relevant_frames(
    scan_id: str,
    relevant_ids: list[int],
    num_total_scene_frames: int,
    num_frames: int,
    shuffle: bool = False,
    fraction_relevant_frames: float = 0.5,
    require_n_target_frames: int = 0,
    at_most_n_relevant_frames: Optional[int] = None,
    dataset_dict = None,
    **kwargs
):
    global scene_pixel_dict_with_dataloader_idxs
    try:
        get_data(get_clip_data=False)
        get_dataloader_filename_mapping(dataset_dict, scan_id)
    except Exception as e:
        print(f"Failed to get dataloader filename mapping for {scan_id} with error {e}")
        raise e

    pixel_dict = scene_pixel_dict_with_dataloader_idxs[scan_id]
    object_ids: list[int] = relevant_ids
    object_lists = []

    for object_id in object_ids:
        try:
            indices, counts = pixel_dict[object_id]
        except:
            print(f"Object {object_id} not found in scene {scan_id}, keys: {pixel_dict.keys()}")
            continue

        if shuffle:
            if len(counts) == 0:
                print(f"Object {object_id} has no frames in scene {scan_id}")
                continue
                
            shuffle_idx = torch.multinomial(
                counts.float(), len(counts), replacement=False
            )
            indices, counts = indices[shuffle_idx], counts[shuffle_idx]

        object_lists.append((object_id, indices.tolist()))

    try:
        frame_ids = select_m_items(object_lists, int(num_frames * fraction_relevant_frames), require_n_target_frames=require_n_target_frames, at_most_n_relevant_frames=at_most_n_relevant_frames)
    except Exception as e:
        print(f"Failed to select frames for {scan_id} with error {e}")
        frame_ids = []

    if len(frame_ids) < num_frames:
        unchosen_frame_ids = np.setdiff1d(np.arange(num_total_scene_frames), np.array(frame_ids))
        if len(unchosen_frame_ids) == 0:
            unchosen_frame_ids = np.arange(num_total_scene_frames)
        random_frame_ids = np.random.choice(unchosen_frame_ids, size=num_frames - len(frame_ids), replace=True)
        frame_ids.extend(random_frame_ids)

    return frame_ids
