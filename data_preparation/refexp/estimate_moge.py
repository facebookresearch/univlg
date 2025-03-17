# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import ipdb
import glob
import socket
import sys
import argparse
import numpy as np
from copy import copy
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Subset
import argparse

# See: https://github.com/microsoft/MoGe
sys.path.append(os.path.join(os.getcwd(), 'MoGe'))
from moge.model import MoGeModel

st = ipdb.set_trace


def split_dataset(dataset, n: int, m: int):
    # Ensure m is valid
    if m < 0 or m >= n:
        raise ValueError(f"m must be between 0 and {n-1}, but got {m}.")
    
    # Calculate the size of each subset
    total_len = len(dataset)
    subset_size = total_len // n
    remainder = total_len % n

    # Calculate the start and end index of the m-th subset
    start_idx = m * subset_size + min(m, remainder)
    end_idx = start_idx + subset_size + (1 if m < remainder else 0)

    # Return the m-th subset
    indices = list(range(start_idx, end_idx))
    if isinstance(dataset, torch.utils.data.Dataset):
        return Subset(dataset, indices)
    else:
        return dataset[slice(start_idx, end_idx)]
    
def save(points, invalid_mask, save_folder, file_prefix):
    """
        intrinsic: [4] fx, fy, cx, cy
        extrinsic: [4, 4] camera to world transformation matrix
        depth: [H, W] depth map
    """

    # save points
    points_folder = save_folder + '/points_new'
    os.makedirs(points_folder, exist_ok=True)
    points = (points * 1000.0).astype(np.int16)
    points = np.save(points_folder + '/' + file_prefix + '.npy', points)
    
    # masks
    masks_folder = save_folder + '/masks'
    os.makedirs(masks_folder, exist_ok=True)
    invalid_mask = np.save(masks_folder + '/' + file_prefix + '.npy', invalid_mask)

# Define dataloader for loading images from a folder
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, scannet_like=False, sam_like=False, folder_min=None, folder_max=None):
        if scannet_like:
            self.files = sorted(glob.glob(folder + '/*/color/*.jpg'))
        elif sam_like:
            if folder_min is None or folder_max is None:
                print(f"Folder {folder} is sam_like and no folder_min or folder_max provided")
                self.files = sorted(glob.glob(folder + '/**/*.jpg', recursive=True))
            else:
                files_all = []
                for i in range(folder_min, folder_max + 1):
                    files_all.extend(sorted(glob.glob(folder + f'/batch_{i}/*.jpg')))
                self.files = files_all
        else:
            self.files = sorted(glob.glob(folder + '/*.jpg') + glob.glob(folder + '/*.png') + glob.glob(folder + '/*.jpeg'))

        print(f"Found {len(self.files)} files")
        # debug
        # self.files = [file for file in self.files if file.split('/')[-3] == 'scene_5' or file.split('/')[-3] == 'scene_10']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = cv2.imread(self.files[idx])[:, :, ::-1]
        return {'image': image, 'file': self.files[idx]}


def main(args):
    print(f"Initializing models")
    device = torch.device("cuda")

    # Load the model from huggingface hub (or load from local).
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)  
    
    print(f"Initialized models")

    datasets = []
    for folder in args.folders:
        scannet_like = 'SEMSEG_100k' in folder
        sam_like = 'sam' in folder.lower() or "coco" in folder.lower() or "paco" in folder.lower()
        dataset = ImageFolderDataset(folder, scannet_like, sam_like, args.folder_min, args.folder_max)
        datasets.append(dataset)
        print(f"Dataset {folder} length: {len(dataset)}, scannet_like: {scannet_like}, sam_like: {sam_like}")

    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    print(f"Combined dataset length: {len(combined_dataset)}")

    subset = split_dataset(combined_dataset, args.num_tasks, args.task_id)
    print(f"Processing {len(subset)} images")

    for data in tqdm(subset):
        image, file = data['image'], data['file']

        if 'SEMSEG_100k' in file:
            save_folder = f"{os.path.dirname(os.path.dirname(file))}/{file.split('/')[-3]}"
        elif 'sam' in file.lower():
            batch_name = file.split('/')[-2]
            save_folder = str(Path(os.path.dirname(file)).parent) + '_' + args.suffix + '/' + batch_name
        else:
            save_folder = str(Path(file).parent.parent) + '_' + args.suffix
        
        file_prefix = os.path.splitext(os.path.basename(file))[0]
        
        image = torch.tensor(image / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)    
        output = model.infer(image)
        points = output['points'].cpu()
        invalid_mask = output['mask'].cpu()
        
        save(points, invalid_mask, save_folder, file_prefix)
        
        if args.visualize:
            import wandb
            if not wandb.run:
                wandb.init(project='depth_estimation')
            
            point_cloud = output['points']
            wandb.log({
                "point_cloud": wandb.Object3D(torch.cat([point_cloud.reshape(-1, 3), image.permute(1, 2, 0).reshape(-1, 3) * 255.0], dim=1).cpu().numpy()),
                "rgb": wandb.Image(image),
            })
            continue


if __name__ == '__main__':
    print(f"Running on hostname {socket.gethostname()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', nargs='+', required=True, help='List of folders to process')
    parser.add_argument('--folder_min', type=int, default=None)
    parser.add_argument('--folder_max', type=int, default=None)
    parser.add_argument('--task_id', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))
    parser.add_argument('--num_tasks', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)))
    parser.add_argument('--output_folder', type=str, default=None, help='Force output to this folder')
    parser.add_argument('--visualize', action='store_true', help='Visualize results without saving')
    parser.add_argument('--suffix', type=str, default='3d', help='3d')
    args = parser.parse_args()
    print(f"Running task {args.task_id} of {args.num_tasks} with output folder {args.output_folder}")
    print(f"Processing folders: {args.folders}")
    main(args)
    print('Done!')