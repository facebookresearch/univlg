# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import ipdb
import pandas as pd
import torch
import transformers
from accelerate import PartialState
from accelerate.utils import gather_object
from tqdm import tqdm

st = ipdb.set_trace

distributed_state = PartialState()

_PREDEFINED_SPLITS_REF = {
    "sr3d_ref_scannet_train_single": ("sr3d_train.csv"),
    "sr3d_ref_scannet_val_single": ("sr3d_test.csv"),
    "sr3d_ref_scannet_train_eval_single": ("sr3d_train_eval.csv"),
    "sr3d_ref_scannet_debug_single": ("sr3d_debug.csv"),
    # "nr3d_ref_scannet_train_single": ("nr3d_train_filtered.csv"),
    "nr3d_ref_scannet_val_single": ("nr3d_val_filtered.csv"),
    "nr3d_ref_scannet_train_eval_single": ("nr3d_train_eval_filtered.csv"),
    "nr3d_ref_scannet_debug_single": ("nr3d_debug_filtered.csv"),
    # "nr3d_ref_scannet_anchor_train_single": ("ScanEnts3D_Nr3D_train.csv"),
    "nr3d_ref_scannet_anchor_val_single": ("ScanEnts3D_Nr3D_val.csv"),
    "nr3d_ref_scannet_anchor_train_eval_single": ("ScanEnts3D_Nr3D_train_eval.csv"),
    "nr3d_ref_scannet_anchor_debug_single": ("ScanEnts3D_Nr3D_debug.csv"),
    # "scanrefer_scannet_anchor_train_single": ("ScanRefer_filtered_train_ScanEnts3D_train.csv"),
    "scanrefer_scannet_anchor_val_single": ("ScanRefer_filtered_val_ScanEnts3D_val.csv"),
    "scanrefer_scannet_anchor_train_eval_single": ("ScanRefer_filtered_train_ScanEnts3D_train_eval.csv"),
    "scanrefer_scannet_anchor_debug_single": ("ScanRefer_filtered_train_ScanEnts3D_debug.csv"),
    "scanrefer_scannet_test_single" : ("ScanRefer_filtered_test.csv"),
}


def normalize_caption(caption: str):
    return caption.lower().replace(".", "").replace(",", "").replace(" ", "")


from pathlib import Path

REF_DATASET = Path(os.getenv('REF_DATASET'))
root = REF_DATASET
save_path = REF_DATASET / 'span_pred_text_test.pth'


all_utterances = []
all_dataset_names = []
for key, csv_file in _PREDEFINED_SPLITS_REF.items():
    print(key)
    csv_file = Path(root) / csv_file
    if csv_file.exists():
        sr3d_data = pd.read_csv(csv_file)
        if "utterance" in sr3d_data.columns:
            utterances = sr3d_data["utterance"].tolist()
        elif "description" in sr3d_data.columns:
            utterances = sr3d_data["description"].tolist()
        else:
            print(f"utterance or description not found in {csv_file}")
            breakpoint()

        all_utterances.extend(utterances)
        if "nr3d" in key:
            all_dataset_names.extend(["nr3d"] * len(utterances))
        elif "sr3d" in key:
            all_dataset_names.extend(["sr3d"] * len(utterances))
        elif "scanrefer" in key:
            all_dataset_names.extend(["scanrefer"] * len(utterances))
        else:
            breakpoint()

all_normalized_utterances = [normalize_caption(x) for x in all_utterances]
print(f"Found {len(all_utterances)} utterances")
print("Loading LLaMA")

device = distributed_state.device
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device=device)

print("Pipeline loaded.")
dataset_name = "merge"
output_annos = defaultdict(list)

prompt = "Given a sentence that asks for an object in a scene. Extract the primary subject from each sentence. Do not include any accompanying adjectives. Output the primary subject only. Do not include any other text or punctuation. Do not ask any questions. Only output a single subject, not multiple."

nr3d_messages = [
    {
        "role": "system",
        "content": prompt,
    },
    {"role": "user", "content": "if looking at the shoes from the kitchen the pair on the far left"},
    {"role": "assistant", "content": "pair"},
    {"role": "user", "content": "the left filing cabinet of the two next each other it is also closest to the door"},
    {"role": "assistant", "content": "cabinet"},
    {"role": "user", "content": "under the desk with three monitors the file cabinet is in the middle"},
    {"role": "assistant", "content": "file cabinet"},
    {"role": "user", "content": "looking from above with the couch to your left and the bed to your right the middle shoe is what you want"},
    {"role": "assistant", "content": "shoe"},
    {"role": "user", "content": "the upper bar in the bathtub"},
    {"role": "assistant", "content": "bar"},
    {"role": "user", "content": "this desk is located close to the window in front of it is a maroon office chair and a blue exercise ball"},
    {"role": "assistant", "content": "desk"},
    {"role": "user", "content": "this is the middle monitor on the desk with three monitors"},
    {"role": "assistant", "content": "monitor"},
    {"role": "user", "content": "stand facing the desk it is the books on the left edge of the desk"},
    {"role": "assistant", "content": "books"},
    {"role": "user", "content": "bottom towel bar in the shower"},
    {"role": "assistant", "content": "bar"},
    {"role": "user", "content": "wall mounted cabinet above the toilet"},
    {"role": "assistant", "content": "wall mounted cabinet"},
    {"role": "user", "content": "facing the desk with the two cabinet 's next to each other it is the cabinet on the right"},
    {"role": "assistant", "content": "cabinet"},
    {"role": "user", "content": "looking for a towel they are hanging on the wall inside of bath tub there hanging on a bar in the shower tub"},
    {"role": "assistant", "content": "towel"},
    {"role": "user", "content": "the shoes closest the bed"},
    {"role": "assistant", "content": "shoes"},
    {"role": "user", "content": "the front entrance door into room"},
    {"role": "assistant", "content": "door"},
    {"role": "user", "content": "the backpack farthest from the bathroom and closest to the kitchen sink"},
    {"role": "assistant", "content": "backpack"},
    {"role": "user", "content": "the table with three monitors choose the middle one"},
    {"role": "assistant", "content": "monitors"},
    {"role": "user", "content": "the light grey file cabinet next to the desk"},
    {"role": "assistant", "content": "file cabinet"},
    {"role": "user", "content": "tall flat object that is used to close off the opening between rooms . object is located in the opening of the bathroom ."},
    {"role": "assistant", "content": "object"},
    {"role": "user", "content": "elongated l shaped object used for holding and preparing food . object is located in the kitchen and includes the sink and the rest of the top of the counter ."},
    {"role": "assistant", "content": "object"},
    {"role": "user", "content": "the shower curtain is white and is pushed open to the left side when facing the bathtub . the shower curtain is pushed open on the side of the blue / teal rug and tub faucet ."},
    {"role": "assistant", "content": "shower curtain"},
    {"role": "user", "content": "black bag shaped object used for easily carrying multiple items . has straps connected for easy carrying and is located on the floor in front of the sink ."},
    {"role": "assistant", "content": "object"},
    {"role": "user", "content": "there is a storage bin in the room . it has a green and blue trash can on top of it ."},
    {"role": "assistant", "content": "storage bin"},
    {"role": "user", "content": "the windows above the small piano keyboard"},
    {"role": "assistant", "content": "windows"},
]

sr3d_messages = [
    {
        "role": "system",
        "content": "Given a sentence that asks for an object in a scene. Extract the primary subject from each sentence. Do not include any accompanying adjectives. Output the primary subject only. Do not include any other text or punctuation.",
    },
    {"role": "user", "content": "find the office chair that is near the copier."},
    {"role": "assistant", "content": "office chair"},
    {"role": "user", "content": "select the trash can that is near the printer."},
    {"role": "assistant", "content": "trash can"},
    {"role": "user", "content": "the monitor that is near the door."},
    {"role": "assistant", "content": "monitor"},
    # New
    {"role": "user", "content": "choose the stool that is in the center of the table and the radiator"},
    {"role": "assistant", "content": "stool"},
    {"role": "user", "content": "the backpack that is in the center of the closet and the stove"},
    {"role": "assistant", "content": "backpack"},
    {"role": "user", "content": "find the towel that is in the middle of the closet and the table"},
    {"role": "assistant", "content": "towel"},
    {"role": "user", "content": "select the sink that is near the bed"},
    {"role": "assistant", "content": "sink"},
    {"role": "user", "content": "the backpack that is in the center of the desk and the stove"},
    {"role": "assistant", "content": "backpack"},
    {"role": "user", "content": "choose the bottle that is in the center of the table and the bathroom vanity"},
    {"role": "assistant", "content": "bottle"},
]

scanrefer_messages = [
    {
        "role": "system",
        "content": "Given a sentence that asks for an object in a scene. Extract the primary subject from each sentence. Do not include any accompanying adjectives. Output the primary subject only. Do not include any other text or punctuation.",
    },
    {"role": "user", "content": "underneath the white desk is a drawer unit . the unit supports the desk . it consists of two separate drawers ."},
    {"role": "assistant", "content": "drawer unit"},
    {"role": "user", "content": "this is a orange throw pillow . this pillow is on the couch ."},
    {"role": "assistant", "content": "throw pillow"},
    {"role": "user", "content": "there is a black office chair in the middle of the room . it is in front of the desk with the computer screen on top ."},
    {"role": "assistant", "content": "office chair"},
    {"role": "user", "content": "the computer tower is below the southern side of the rightmost desk in the room . the computer tower is a black rectangle ."},
    {"role": "assistant", "content": "computer tower"},
    {"role": "user", "content": "a silver bin . placed on the left to the toilet ."},
    {"role": "assistant", "content": "bin"},
    {"role": "user", "content": "a dark marble vanity . placed on the bathroom ."},
    {"role": "assistant", "content": "vanity"},
    {"role": "user", "content": "it is a white refrigerator . the white refrigerator is sitting across from the white table in the middle of the room ."},
    {"role": "assistant", "content": "refrigerator"},
    {"role": "user", "content": "this is a green bucket . it is on top of a black storage bin ."},
    {"role": "assistant", "content": "bucket"},
    {"role": "user", "content": "the bathroom vanity is to the right of the toilet . it is above the scale that is on the floor ."},
    {"role": "assistant", "content": "bathroom vanity"},
    {"role": "user", "content": "the object is a toilet . if you were entering the bathroom it would be slightly to your left against the wall in front of you ."},
    {"role": "assistant", "content": "object"},
    {"role": "user", "content": "there is a square nightstand . it is to the side of the bed on the left ."},
    {"role": "assistant", "content": "nightstand"},
    {"role": "user", "content": "a brown wooden shelf . hanging on the wall ."},
    {"role": "assistant", "content": "shelf"},
]

all_targets = []
combined_data = [(x, y) for x, y in zip(all_utterances, all_dataset_names)]

with distributed_state.split_between_processes(combined_data) as _data:
    for caption, dataset_name in tqdm(_data):
        end_1 = (caption.find(".") + len(caption)) % len(caption)
        bk = end_1
        sen_caption = caption[:bk] + "."
        if dataset_name == "nr3d":
            messages = nr3d_messages
        elif dataset_name == "sr3d":
            messages = sr3d_messages
        elif dataset_name == "scanrefer":
            messages = scanrefer_messages
        else:
            breakpoint()

        outputs = pipeline([*messages, {"role": "user", "content": sen_caption}])
        final_output = outputs[0]["generated_text"][-1]["content"]
        final_output = final_output.split('\n')[0]
        final_output = final_output.rstrip()
        final_output = final_output.rstrip('.,!?')
        print(f"INPUT: {sen_caption}, OUTPUT: {final_output}, TARGET: {final_output}")
        all_targets.append(final_output)

print(f"Rank: {distributed_state.process_index}, World Size: {distributed_state.num_processes}, generated {len(all_targets)} targets")
all_targets = gather_object(all_targets)
all_targets = all_targets[: len(all_utterances)]
print(f"Found {len(all_targets)} targets after gathering")

if distributed_state.is_main_process:
    hashes = {hashlib.md5(normalize_caption(utterance).encode()).hexdigest(): i for i, utterance in enumerate(all_normalized_utterances)}
    torch.save(dict(target_strs=all_targets, hashes=hashes), save_path)
    print(f"saved to {save_path}")

distributed_state.wait_for_everyone()
