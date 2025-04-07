# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast

import ipdb
from random import shuffle
from univlg.data_video.sentence_utils import (
    create_positive_map,
    create_positive_map_for_od_labels,
    get_positive_tokens,
)
from transformers import AutoTokenizer, RobertaTokenizerFast

from univlg.data_video.datasets.ref_coco_utils import get_root_and_nouns, consolidate_spans

from .dataset_mapper_scannet import ScannetDatasetMapper
import torch
import hashlib
from pathlib import Path
import numpy as np
import os
st = ipdb.set_trace

def get_chars_from_positive(_tokens_positive, _utterance):
    chars_positive = []
    for token in _tokens_positive:
        if (isinstance(token, list) or isinstance(token, tuple)) and not isinstance(token[0], int):
            chars_positive.extend(get_chars_from_positive(token, _utterance))
        else:
            # If token is a tuple/list of start and end positions
            if isinstance(token, (list, tuple)) and len(token) == 2:
                start, end = token
                chars_positive.append(_utterance[start:end])
            else:
                chars_positive.append(token)
    return chars_positive

class Sr3dDatasetMapper:
    def __init__(
        self, cfg, is_train, dataset_name, scannet_dict, scene_to_id_map, decoder_3d
    ):
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.scannet_dict = scannet_dict
        self.scene_to_id_map = scene_to_id_map
        self.decoder_3d = decoder_3d
        
        self.scannet_name = (
            "scannet_context_instance_train_20cls_single_100k"
            if "train" in dataset_name
            else "scannet_context_instance_val_20cls_single_100k"
        )
        self.scannet_mapper = ScannetDatasetMapper(
            self.cfg,
            is_train=self.is_train,
            dataset_name=self.scannet_name,
            dataset_dict=self.scannet_dict,
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

        self.USE_AUTO_NOUN_DETECTION = getattr(cfg, "USE_AUTO_NOUN_DETECTION", False)
        self.span_preds = None

    def process_lang_data(self, dataset_dict):
        if "dataset" not in dataset_dict:
            breakpoint()

        original_utterance = dataset_dict.get("utterance", dataset_dict.get("description", None))
        if ("scanrefer" in dataset_dict["dataset"] or "nr3d" in dataset_dict["dataset"]) and self.cfg.LOAD_SCANENTS and (self.USE_AUTO_NOUN_DETECTION is False or "scanrefer" in dataset_dict["dataset"] or self.is_train) and not self.cfg.TEST_DATASET_INFERENCE:
            dataset_dict = self.process_scanent_lang_data(dataset_dict)
            if self.USE_AUTO_NOUN_DETECTION and not self.is_train:
                dataset_dict["anchor_ids"] = []
                dataset_dict["anchors_types"] = []
                if 'tokens_positive' in dataset_dict:
                    dataset_dict["tokens_positive"] = [dataset_dict["tokens_positive"][0]]

        elif "anchor_ids" not in dataset_dict:
            dataset_dict["anchor_ids"] = []
            dataset_dict["anchors_types"] = []

        if self.cfg.TEST_DATASET_INFERENCE:
            dataset_dict["utterance"] = original_utterance
            dataset_dict["instance_type"] = dataset_dict["target_id"]

        utterance = dataset_dict["utterance"].lower()
        target_id = dataset_dict["target_id"]

        if type(dataset_dict['anchor_ids']) is str:
            anchor_ids = ast.literal_eval(dataset_dict['anchor_ids'])
        else:
            anchor_ids = dataset_dict['anchor_ids']

        target_name = dataset_dict['instance_type']
        if type(dataset_dict['anchors_types']) is str:
            anchor_names = ast.literal_eval(dataset_dict['anchors_types'])
        else:
            anchor_names = dataset_dict['anchors_types']

        all_object_names = [target_name] + anchor_names

        if "tokens_positive" not in dataset_dict or self.USE_AUTO_NOUN_DETECTION:
            if self.USE_AUTO_NOUN_DETECTION and not self.is_train:
                if self.span_preds is None:
                    span_pred_path = Path(f"{os.environ['PRECOMPUTED_SCANNET_PATH']}/span_pred_text.pth", weights_only=True)
                        
                    if span_pred_path.exists():
                        self.span_preds = torch.load(span_pred_path)
                    else:
                        print(f"Failed to find span_pred_path: {span_pred_path}")

                def normalize_caption(caption: str):
                    return caption.lower().replace(".", "").replace(",", "").replace(" ", "")
                
                text_hash = hashlib.md5(normalize_caption(original_utterance).encode()).hexdigest()
                tokens_positive = None
                if text_hash in self.span_preds["hashes"]:
                    target_str = self.span_preds["target_strs"][self.span_preds["hashes"][text_hash]]
                    try:
                        tokens_positive = get_positive_tokens(utterance, [target_str])
                    except ValueError as e:
                        print(f"Failed to get positive tokens for {target_str}: {e}. tokens_positive: \
                              {tokens_positive}")
                else:
                    print(f"Failed to find hash: {text_hash} in span_preds")

                if tokens_positive is None:
                    _, _, root_spans_spacy, _ = get_root_and_nouns(utterance)
                    tokens_positive = [consolidate_spans(root_spans_spacy, utterance)]
            else:
                tokens_positive = get_positive_tokens(utterance, all_object_names)

            dataset_dict["tokens_positive"] = tokens_positive
        
        tokens_positive = dataset_dict["tokens_positive"]

        max_len = (
            self.cfg.MODEL.MAX_SEQ_LEN
            if not self.cfg.TEXT_ENCODER_TYPE == "clip"
            else 77
        )
        tokenized = self.tokenizer(
            utterance, return_tensors="pt", max_length=max_len, truncation=True
        )

        positive_map = create_positive_map(
            tokenized, tokens_positive, max_query_len=max_len
        )

        return {
            "text_caption": utterance,
            "target_id": target_id,
            "anchor_ids": anchor_ids,
            "target_name": target_name,
            "anchors_names": anchor_names,
            "tokens_positive": tokens_positive,
            "tokenized": tokenized,
            "positive_map": positive_map,
            "positive_map_od": None,
            "annotation_id": dataset_dict['ann_id'] if "ann_id" in dataset_dict else None
        }

    def process_lang_data_scanqa(self, dataset_dict):
        utterance = dataset_dict["question"].lower() + " find relevant objects"

        target_id = dataset_dict["object_ids"][0]
        anchor_ids = dataset_dict["object_ids"][1:]

        all_object_names = ["relevant objects"] * len(dataset_dict["object_ids"])

        tokens_positive = get_positive_tokens(utterance, all_object_names)

        max_len = (
            self.cfg.MODEL.MAX_SEQ_LEN
            if not self.cfg.TEXT_ENCODER_TYPE == "clip"
            else 77
        )
        tokenized = self.tokenizer(
            utterance, return_tensors="pt", max_length=max_len, truncation=True
        )

        positive_map = create_positive_map(
            tokenized, tokens_positive, max_query_len=max_len
        )
        labels_to_positions = {k : v[0] for k, v in enumerate(tokens_positive)}
        positive_map_od = create_positive_map_for_od_labels(
            tokenized, labels_to_positions, max_query_len=max_len
        )

        return {
            "text_caption": utterance,
            "target_id": target_id,
            "anchor_ids": anchor_ids,
            "target_name": "relevant objects",
            "anchors_names": ["relevant objects"] * len(anchor_ids),
            "tokens_positive": tokens_positive,
            "tokenized": tokenized,
            "positive_map": positive_map,
            "positive_map_od": positive_map_od,
            "answers": dataset_dict["answers"]
        }

    def process_lang_data_sqa3d(self, dataset_dict):
        utterance = (dataset_dict["situation"] + " " + dataset_dict["question"]).lower() + " find relevant objects"

        target_id = 0
        anchor_ids = []

        all_object_names = ["relevant objects"]

        tokens_positive = get_positive_tokens(utterance, all_object_names)

        max_len = (
            self.cfg.MODEL.MAX_SEQ_LEN
            if not self.cfg.TEXT_ENCODER_TYPE == "clip"
            else 77
        )
        tokenized = self.tokenizer(
            utterance, return_tensors="pt", max_length=max_len, truncation=True
        )

        positive_map = create_positive_map(
            tokenized, tokens_positive, max_query_len=max_len
        )
        labels_to_positions = {k : v[0] for k, v in enumerate(tokens_positive)}
        positive_map_od = create_positive_map_for_od_labels(
            tokenized, labels_to_positions, max_query_len=max_len
        )

        return {
            "text_caption": utterance,
            "target_id": target_id,
            "anchor_ids": anchor_ids,
            "target_name": "relevant objects",
            "anchors_names": [],
            "tokens_positive": tokens_positive,
            "tokenized": tokenized,
            "positive_map": positive_map,
            "positive_map_od": positive_map_od,
            "answers": dataset_dict["answers"]
        }

    def process_scanent_lang_data(self, dataset_dict):
        target_id, target_name = dataset_dict["target_id"], None
        anchor_ids, anchor_names = [], []
        tokens = ast.literal_eval(dataset_dict["tokens"])
        utterance = " ".join(tokens)
        original_utterance = dataset_dict.get("utterance", dataset_dict.get("description", None))

        tokens_positive = []
        object_ids = []

        if "entities" not in dataset_dict:
            breakpoint()

        for entity in ast.literal_eval(dataset_dict["entities"]):
            assert len(entity) == 2
            for name in entity[1]:
                new_id = int(name.split("_")[0])
                _token_idxs = sorted(entity[0])
                obj_name = " ".join([tokens[i] for i in range(_token_idxs[0], _token_idxs[-1] + 1)])

                if new_id != target_id:
                    if new_id not in object_ids:
                        anchor_ids.append(new_id)
                        anchor_names.append(obj_name)
                else:
                    target_id = new_id
                    target_name = obj_name

                start_index = len(' '.join(tokens[:entity[0][0]]))

                # to account for space
                if entity[0][0] != 0:
                    start_index += 1
                assert start_index != -1

                end_index = len(' '.join(tokens[:entity[0][-1]])) + len(tokens[entity[0][-1]])

                # to account for space
                if entity[0][-1] != 0:
                    end_index += 1

                if new_id not in object_ids:
                    # ensure target_id is always the first object
                    if new_id == target_id:
                        tokens_positive = [[[start_index, end_index]]] + tokens_positive
                        object_ids = [new_id] + object_ids
                    else:
                        tokens_positive.append([[start_index, end_index]])
                        object_ids.append(new_id)
                else:
                    index = object_ids.index(new_id)
                    tokens_positive[index].append([start_index, end_index])

        if target_name is None:
            print("target not found in entities")
            # this is annotation error
            # we basically make the first word of the sentene the target
            object_ids = [target_id] + object_ids
            target_name = tokens[0]
            start_index = utterance.find(tokens[0])
            end_index = start_index + len(tokens[0])
            tokens_positive = [[[start_index, end_index]]] + tokens_positive

        assert len(anchor_ids) + 1 == len(tokens_positive), f"{anchor_ids=},{tokens_positive=}"
        assert object_ids[0] == target_id, f"{object_ids=},{target_id=},{anchor_ids=},{tokens_positive=},{dataset_dict}="
        dataset_dict["utterance"] = utterance
        dataset_dict["target_id"] = target_id
        dataset_dict["anchor_ids"] = anchor_ids
        dataset_dict["instance_type"] = target_name
        dataset_dict["anchors_types"] = anchor_names
        dataset_dict["tokens_positive"] = tokens_positive
        dataset_dict["original_utterance"] = original_utterance if original_utterance is not None else utterance
        return dataset_dict

    def get_scene_data(self, dataset_dict: dict):
        scene_name = dataset_dict['scan_id']
        scene_id = self.scene_to_id_map[scene_name]
        scene_dict = self.scannet_dict[scene_id]

        if self.is_train or self.cfg.FORCE_SUBSAMPLE:
            anchor_ids = dataset_dict.get('anchor_ids', [])
            if not self.cfg.FORCE_SUBSAMPLE:
                assert "anchor_ids" in dataset_dict, "anchor_ids not found in dataset_dict"

            if type(anchor_ids) is str:
                anchor_ids = ast.literal_eval(anchor_ids)

            relevant_ids = [dataset_dict['target_id'], *anchor_ids]
            num_important_relevant_ids = len(relevant_ids)

        else:
            relevant_ids = None
            num_important_relevant_ids = None

        scene_data = self.scannet_mapper(
            scene_dict,
            relevant_ids=relevant_ids,
            num_important_relevant_ids=num_important_relevant_ids,
            sr3d_data=dataset_dict,
            referential_dataset=True,
        )
        scene_data['relevant_ids'] = relevant_ids
        return scene_data

    def __call__(self, dataset_dict):
        if isinstance(dataset_dict, dict):
            assert self.is_train, "Training mode requires a single annotation dict"
            # Training mode where we have a single annotation dict, we still return it as a list of [a single] dict for consistency
            if "scanqa" in self.dataset_name:
                dataset_dict_processed = self.process_lang_data_scanqa(dataset_dict)
                if 'scan_id 'not in dataset_dict:
                    dataset_dict['scan_id'] = dataset_dict['scene_id']
                if 'target_id' not in dataset_dict:
                    dataset_dict['target_id'] = dataset_dict['object_ids'][0]
            elif "sqa3d" in self.dataset_name:
                dataset_dict_processed = self.process_lang_data_sqa3d(dataset_dict)
                if 'scan_id 'not in dataset_dict:
                    dataset_dict['scan_id'] = dataset_dict['scene_id']
                if 'target_id' not in dataset_dict:
                    dataset_dict['target_id'] = dataset_dict_processed['target_id']
            else:
                dataset_dict_processed = self.process_lang_data(dataset_dict)
            dataset_dict['anchor_ids'] = dataset_dict_processed['anchor_ids']
            scene_data = self.get_scene_data(dataset_dict)
            dataset_dict = [dataset_dict_processed]
        else:
            assert not self.is_train, "Eval mode requires a list of annotation dicts"
            # Eval mode where we have all annotation dicts for a single scene. We return a list of dicts.
            # if self.cfg.FORCE_SUBSAMPLE:
            #     assert len(dataset_dict) == 1, "FORCE_SUBSAMPLE is only supported for single annotations"
            scene_data = self.get_scene_data(dataset_dict[0])
            if "scanqa" in self.dataset_name:
                dataset_dict = [self.process_lang_data_scanqa(dd) for dd in dataset_dict]
            elif "sqa3d" in self.dataset_name:
                dataset_dict = [self.process_lang_data_sqa3d(dd) for dd in dataset_dict]
            else:
                dataset_dict = [self.process_lang_data(dd) for dd in dataset_dict]

        aggregated_data = {**scene_data, 'sr3d_data': dataset_dict}
        aggregated_data['dataset_name'] = self.dataset_name
        return aggregated_data
    
    
class ScanqaDatasetMapper:
    def __init__(
        self, cfg, is_train, dataset_name, scannet_dict, scene_to_id_map, decoder_3d
    ):
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.scannet_dict = scannet_dict
        self.scene_to_id_map = scene_to_id_map
        self.decoder_3d = decoder_3d
        self.scannet_name = (
            "scannet_context_instance_train_20cls_single_100k"
            if "train" in dataset_name
            else ("scannet_context_instance_val_20cls_single_100k" if "val" in dataset_name else "scannet_context_instance_test_20cls_single_100k")
        )
        self.scannet_mapper = ScannetDatasetMapper(
            self.cfg,
            is_train=self.is_train,
            dataset_name=self.scannet_name,
            dataset_dict=self.scannet_dict,
            decoder_3d=self.decoder_3d,
        )

    def __call__(self, dataset_dict):
        scene_name = dataset_dict["scene_id"]
        scene_id = self.scene_to_id_map[scene_name]
        scene_dict = self.scannet_dict[scene_id]

        # TODO: Pass relevant_ids keyword argument
        scene_data = self.scannet_mapper(scene_dict)
        dataset_dict_processed = self.process_scanqa_lang_data(dataset_dict)
        aggregated_data = {**scene_data, **dataset_dict_processed}
        return aggregated_data


class Sqa3dDatasetMapper:
    def __init__(
        self, cfg, is_train, dataset_name, scannet_dict, scene_to_id_map, decoder_3d
    ):
        self.cfg = cfg
        self.is_train = is_train
        self.dataset_name = dataset_name
        self.scannet_dict = scannet_dict
        self.scene_to_id_map = scene_to_id_map
        self.decoder_3d = decoder_3d
        self.scannet_name = (
            "scannet_context_instance_train_20cls_single_100k"
            if "train" in dataset_name
            else "scannet_context_instance_val_20cls_single_100k"
        )
        self.scannet_mapper = ScannetDatasetMapper(
            self.cfg,
            is_train=self.is_train,
            dataset_name=self.scannet_name,
            dataset_dict=self.scannet_dict,
            decoder_3d=self.decoder_3d,
        )

    def __call__(self, dataset_dict):
        scene_name = dataset_dict["scene_id"]
        scene_id = self.scene_to_id_map[scene_name]
        scene_dict = self.scannet_dict[scene_id]

        # TODO: Pass relevant_ids keyword argument
        scene_data = self.scannet_mapper(scene_dict)
        aggregated_data = {**scene_data, **dataset_dict}
        return aggregated_data

