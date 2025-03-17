# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import re

import ipdb
import torch
from torch_scatter import scatter_mean

st = ipdb.set_trace


def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name


def sanity_check_target_after_processing(target):
    assert len(target.bbox) == len(target.extra_fields["boxes"])


def create_positive_map_for_od_labels(tokenized, label_to_positions, max_query_len):
    """construct a map such that positive_map[i] = j, where j is the object detection label of the token i"""
    """
    {3: [1: 5)}
    256 : -1 3 3 3 3 -1 .. 8 8 ..
    the woman in the garden
    -1 -1 -1 -1 -1
    """
    positive_map = (
        torch.ones(max_query_len, dtype=torch.float) * -1
    )  # -1 means no match
    keys = list(label_to_positions.keys())
    for j, key in enumerate(keys):
        tok_list = label_to_positions[key]
        # one label only mapps to one location
        beg, end = tok_list
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except Exception:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except Exception:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        assert beg_pos is not None and end_pos is not None
        positive_map[beg_pos : end_pos + 1].fill_(key)
    return positive_map


def create_positive_map(tokenized, tokens_positive, max_query_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), max_query_len), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for beg, end in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            assert end_pos <= max_query_len, f"end_pos {end_pos} > max_query_len {max_query_len}"
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    assert (
        positive_map[:, -1].sum() == 0
    ), "the last token should not be used, NoOBJ token"
    _ret = positive_map / (positive_map.sum(-1)[:, None] + 1e-6)

    _condition = torch.allclose(
        _ret.sum(-1), torch.ones_like(_ret.sum(-1))
    )

    # if not _condition:
    #     print(f"Failed condition")
    #     print(f"positive_map: {positive_map}")
    #     print(f"positive_map.sum(-1): {positive_map.sum(-1)}")
    #     print(f"_ret: {_ret}")
    #     print(f"_ret.sum(-1): {_ret.sum(-1)}")
    #     print(f"tokens_positive: {tokens_positive}")
    #     print(f"tokenized: {tokenized}")
    #     assert False

    return _ret


def convert_od_to_grounding_simple_indv(
    labels,
    ind_to_class,
    disable_shuffle=True,
    add_detection_prompt=False,
    separation_tokens=" ",
    caption_prompt=None,
    tokenizer=None,
    max_query_len=256,
):
    label_list = list(sorted(ind_to_class.keys()))  # do not include the background
    if not disable_shuffle:
        random.shuffle(label_list)

    label_to_positions = {label: i for i, label in enumerate(label_list)}

    positive_map = torch.zeros(len(labels), max_query_len, dtype=torch.float)
    positions = torch.tensor([label_to_positions[label] for label in labels])
    positive_map_od = torch.ones(max_query_len, dtype=torch.float) * -1

    if len(labels) != 0:
        positive_map[torch.arange(len(labels)), positions] = 1.0
        positive_map_od[: len(label_list)] = torch.tensor(label_list)
        assert torch.allclose(
            positive_map.sum(-1), torch.ones_like(positive_map.sum(-1))
        ), "some positive maps are empty, possibly due to sequence length larger than max_query_len"
    else:
        positions = torch.tensor([0])
    return positive_map, positive_map_od, label_list


def convert_od_to_grounding_simple(
    labels,
    ind_to_class,
    disable_shuffle=True,
    add_detection_prompt=False,
    separation_tokens=" ",
    caption_prompt=None,
    tokenizer=None,
    max_query_len=256,
):
    """
    Convert object detection data into grounding data format, on the fly.
    ind_to_class: {0: "__background__", 1 : "person" ...}, contiguous id
    """

    def generate_sentence_from_labels(
        positive_label_list, negative_label_list, disable_shuffle=True
    ):
        label_to_positions = {}
        label_list = negative_label_list + positive_label_list
        if not disable_shuffle:
            random.shuffle(label_list)
            assert (
                caption_prompt is None
            ), "Should not specify caption_prompt when shuffle is enabled!!"  # avoid potential bug

        if add_detection_prompt:
            pheso_caption = "object detection : "
        else:
            pheso_caption = ""

        for index, label in enumerate(label_list):
            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]["prefix"]

            start_index = len(pheso_caption)
            if caption_prompt is not None:
                pheso_caption += clean_name(caption_prompt[index]["name"])
            else:
                pheso_caption += clean_name(
                    ind_to_class[label]
                )  # NOTE: slight change...
            end_index = len(pheso_caption)

            if caption_prompt is not None:
                pheso_caption += caption_prompt[index]["suffix"]

            # e.g.: pheso_caption = "cat dog", where cat is label 4, and dog is label 17
            # label_to_positions: {4: (0, 3), 17: (4, 7)}
            label_to_positions[label] = [start_index, end_index]

            if index != len(label_list) - 1:
                pheso_caption += separation_tokens

        return label_to_positions, pheso_caption

    label_list = list(sorted(ind_to_class.keys()))  # do not include the background
    label_to_positions, pheso_caption = generate_sentence_from_labels(
        positive_label_list=label_list,
        negative_label_list=[],
        disable_shuffle=disable_shuffle,
    )
    tokens_positive = [[label_to_positions[label]] for label in labels]

    # -1 is to preserve a token for the NoOBJ token
    tokenized = tokenizer(
        pheso_caption, return_tensors="pt", max_length=max_query_len, truncation=True
    )

    positive_map_od = create_positive_map_for_od_labels(
        tokenized, label_to_positions, max_query_len
    )
    positive_map = create_positive_map(
        tokenized, tokens_positive, max_query_len=max_query_len
    )

    _condition = torch.allclose(
        positive_map.sum(-1), torch.ones_like(positive_map.sum(-1))
    )

    if not _condition:
        # breakpoint()
        print("Temporarily ignoring but need to fix this")

    # assert _condition, f"some positive maps are empty, possibly due to sequence length larger than max_query_len: {max_query_len}, {pheso_caption}, {positive_map.sum(-1)}, {positive_map.shape}, {len(tokenized)}, {len(tokens_positive)}, {len(label_to_positions)}, {type(tokenized)}"

    return positive_map, positive_map_od, tokens_positive, pheso_caption


def convert_grounding_to_od_logits(logits, num_class, positive_map_od, reduce="mean"):
    # assert NotImplementedError, "Need to verify the correctness of this function"
    assert logits.max() <= 1.0 and logits.min() >= 0.0, "logits should be in [0, 1]"
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_class).to(logits.device)
    for label_j in range(num_class):
        locations_label_j = (
            (positive_map_od == label_j).nonzero(as_tuple=True)[0].tolist()
        )
        if len(locations_label_j) == 0:
            continue
        if reduce == "sum":
            scores[:, :, label_j] = logits[
                :, :, torch.LongTensor(locations_label_j)
            ].sum(-1)
        else:
            scores[:, :, label_j] = logits[
                :, :, torch.LongTensor(locations_label_j)
            ].mean(-1)
    # scores[:, :, -1] = logits[:, :, -1]
    return scores


def convert_grounding_to_od_logits_ref(logits, num_class, positive_maps, reduce="mean"):
    """
    Here the logits are raw outputs not the softmax output!
    logits: (batch_size, q, seq_len)
    num_class: N
    positive_maps: (batch_size, seq_len)
    """
    assert logits.max() <= 1.0 and logits.min() >= 0.0, "logits should be in [0, 1]"
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_class).to(logits.device)
    for i in range(len(positive_maps)):
        locations_label_j = (positive_maps[i]).nonzero(as_tuple=True)[0].tolist()
        if len(locations_label_j) == 0:
            continue
        if reduce == "sum":
            scores[:, :, i] = logits[:, :, torch.LongTensor(locations_label_j)].sum(-1)
        else:
            scores[:, :, i] = logits[:, :, torch.LongTensor(locations_label_j)].mean(-1)
    # scores[:, :, -1] = logits[:, :, -1]
    return scores


def convert_grounding_to_od_logits_batched(logits, num_class, positive_map_od):
    """
    Here the logits are raw outputs not the softmax output!
    logits: (batch_size, q, seq_len)
    num_class: N
    positive_map_od: (batch_size, seq_len)
    """
    scores = torch.ones(logits.shape[0], logits.shape[1], num_class).to(logits) * -100.0

    positive_map_od[positive_map_od == -1] = num_class

    scores_ = scatter_mean(
        logits, positive_map_od[:, None, :].expand(-1, logits.shape[1], -1), dim=2
    )
    mask = torch.ones_like(scores_).bool()
    mask.scatter_(2, positive_map_od[:, None, :].expand(-1, logits.shape[1], -1), False)
    scores_[mask] = -100.0

    # remove invalid scores
    scores_ = scores_[..., :-1]

    scores[:, :, : scores_.shape[-1]] = scores_

    # mask = torch.ones_like(scores).bool()
    # mask.scatter_(2, positive_map_od[:, None, :].expand(-1, logits.shape[1], -1), False)
    # scores[mask] = -100.0

    # scores[:, :, -1] = logits[:, :, -1]
    return scores


def get_positive_tokens(caption, token_lists):
    tokens_positive = []
    for token in token_lists:
        start_index = caption.find(token)
        if start_index == -1:
            raise ValueError(f"token {token} not found in caption {caption}, token_lists: {token_lists}")
        end_index = start_index + len(token)
        tokens_positive.append([[start_index, end_index]])
    return tokens_positive


def sample_classes(scannet_masks, scannet_classes, all_classes, max_classes, return_original_prob=None):
    """
        scannet_masks: N X M
        scannet_classes: N
        all_classes: dict of class names
        max_classes: int
    """

    name_to_cls_id = {v: k for k, v in all_classes.items()}

    if max_classes is None:
        max_classes = len(all_classes)

    if len(all_classes) <= max_classes:
        return scannet_masks, scannet_classes, all_classes

    if return_original_prob is not None and random.random() < return_original_prob:
        return scannet_masks, scannet_classes, all_classes

    max_posiitive_classes = max_classes // 2
    total_positive_classes = set(scannet_classes.tolist())
    negative_classes = set(all_classes.keys()) - total_positive_classes

    if len(total_positive_classes) > max_posiitive_classes:
        total_positive_classes = random.sample(total_positive_classes, max_posiitive_classes)
    else:
        total_positive_classes = list(total_positive_classes)

    negative_classes = random.sample(negative_classes, min(max_classes - len(total_positive_classes), len(negative_classes)))
    selected_classes = total_positive_classes + negative_classes
    new_all_classes = {class_name: all_classes[class_name] for class_name in selected_classes}

    new_scannet_classes = torch.isin(scannet_classes, torch.tensor(total_positive_classes).to(scannet_classes.device))
    new_scannet_masks = scannet_masks[new_scannet_classes]
    new_scannet_classes = scannet_classes[new_scannet_classes]

    if '__background__' in name_to_cls_id:
        new_all_classes[name_to_cls_id['__background__']] = '__background__'

    return new_scannet_masks, new_scannet_classes, new_all_classes


if __name__ == "__main__":
    pass
