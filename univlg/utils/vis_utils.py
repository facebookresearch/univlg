# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pyviz3d.visualizer as vis
import torch
import wandb
from detectron2.structures import Instances
from .detectron2_visualizer import Visualizer
from einops import rearrange
from univlg.global_vars import (
    CLASS_NAME_DICT,
    GENERAL_COLOR_MAP_200,
    SCANNET_COLOR_MAP_20,
    SCANNET_COLOR_MAP_200,
)
from univlg.global_vars import CLASS_NAME_DICT, AI2THOR_CLASS_NAMES
from univlg.modeling.backproject.backproject import voxel_map_to_source
from univlg.data_video.sentence_utils import convert_grounding_to_od_logits
from sklearn.manifold import TSNE
from torch.nn import functional as F


from .feature_vis import embedding_to_3d_color

st = ipdb.set_trace


def visualize_embeddings(embeddings, words, name="embeddings"):
    tsne = TSNE(n_components=2, random_state=0, perplexity=len(words) - 1)
    two_d_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    for i, word in enumerate(words):
        x, y = two_d_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(
            word,
            (x, y),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
    plt.savefig(f"{name}.png")


def visualize_features(features):
    """visualize PCA of features

    Keyword arguments:
    features -- features to visualize (B X V X H X W X C)

    Return: None
    """
    b, v, h, w, c = features.shape
    color = embedding_to_3d_color(
        rearrange(features, "b v h w c -> b c (v h w)"), project_type="pca"
    )
    color = rearrange(color, "b c (v h w) -> b v h w c", v=v, h=h, w=w).numpy()

    if wandb.run is None:
        wandb.init(project="univlg")

    wandb.log(
        {"features": [wandb.Image(color[i, j]) for i in range(b) for j in range(v)]}
    )


def visualize_pca_from_feature_folders(feature_folders):
    """visualize PCA of features

    Keyword arguments:
    feature_folders -- list of feature folders

    Return: None
    """
    if wandb.run is None:
        wandb.init(project="univlg")

    feature_dict = {}
    for feature_folder in feature_folders:
        # iterate over all files in the folder
        for file in sorted(os.listdir(feature_folder)):
            if file.endswith(".npy"):
                feature_path = os.path.join(feature_folder, file)
                feature = torch.from_numpy(np.load(feature_path))
                if feature.shape[1] != 1:
                    print(f"Skipping {feature_path} as it has shape {feature.shape}")
                    continue
                if feature_folder in feature_dict:
                    feature_dict[feature_folder].append(feature)
                else:
                    feature_dict[feature_folder] = [feature]

    # compute principal components from all features
    all_features = torch.cat(
        [torch.cat(feature_dict[feature_folder], 1) for feature_folder in feature_dict],
        dim=0,
    )
    print(all_features.mean(), all_features.sum())

    # st()
    pca = (
        embedding_to_3d_color(
            rearrange(all_features, "b v h w c -> c (b v h w)")[None],
        )[0]
        .permute(1, 0)
        .contiguous()
    )

    output = rearrange(
        pca,
        "(b v h w) c -> b v h w c",
        b=all_features.shape[0],
        v=all_features.shape[1],
        h=all_features.shape[2],
        w=all_features.shape[3],
    )

    output = output.numpy()
    for j in range(output.shape[1]):
        wandb.log(
            {
                "principal_component": [
                    wandb.Image(output[i, j]) for i in range(output.shape[0])
                ]
            }
        )


def visualize_matched_gts_preds(outputs, targets, indices, prefix, use_sofmax=False):
    """
    Inputs:
        outputs: dicts with keys "pred_masks", "pred_classes", "scores"
        targets: list of N dicts with keys "masks", "labels"
        indices: hungarian matched indices tuple
    """
    images = []
    
    coco_metadata = {}
    coco_metadata['thing_classes'] = AI2THOR_CLASS_NAMES
    
    images = torch.stack(
        [target['images'] for target in targets]
    ).flatten(0, 1)[:, None]
    masks = [target['masks'] for target in targets]
    labels = [target['labels'] for target in targets]
    
    # visualize pred
    output_masks = outputs['pred_masks']
    if output_masks.ndim == 3:
        output_masks = voxel_map_to_source(
            output_masks.permute(0, 2, 1), outputs['p2v']
        ).permute(0, 2, 1)
        V, _, H, W = targets[0]['images'].shape
        H_, W_ = H // 4, W // 4
        output_masks = rearrange(output_masks, 'B N (V H W) -> B N V H W', V=V, H=H_, W=W_)
        masks = [rearrange(mask, 'N (V H W) -> N V H W', V=V, H=H_, W=W_) for mask in masks]
        masks = [
            F.interpolate(
                mask.float(), size=(H, W), mode='nearest'
            ) > 0.0 for mask in masks
        ]
        images = torch.stack(
            [target['images'] for target in targets]
        )
        
    if use_sofmax:
        output_scores = outputs['pred_logits'].softmax(-1)
        output_labels = outputs['pred_logits'].softmax(-1)[..., :-1].argmax(-1)
    else:
        output_labels = outputs['pred_logits'].sigmoid()
        num_classes = len(AI2THOR_CLASS_NAMES)
        output_scores = torch.cat(
            [
                convert_grounding_to_od_logits(
                    logits=output_labels[i][None],
                    num_class=num_classes + 1,
                    positive_map_od=targets[i]["positive_map_od"],
                    reduce='mean',
                )
                for i in range(len(output_labels))
            ]
        )
        output_labels = output_scores.argmax(-1)
    
    bs, n, v = output_masks.shape[:3]
    H, W = images.shape[-2:]
    try:
        output_masks = F.interpolate(
            output_masks.flatten(0, 1), size=images.shape[-2:], mode='bilinear', align_corners=False
        ).reshape(bs, n, v, H, W) > 0.0
    except:
        st()
    pred_masks = []
    pred_labels = []
    pred_scores = []
    for i in range(len(indices)):
        pred_mask = torch.zeros_like(masks[i])
        pred_label = torch.zeros_like(labels[i])
        pred_mask[indices[i][1]] = output_masks[i][indices[i][0]]
        pred_label[indices[i][1]] = output_labels[i][indices[i][0]]
        pred_score = torch.zeros_like(labels[i], dtype=outputs['pred_logits'].dtype)
        score_matched = torch.zeros_like(labels[i], dtype=outputs['pred_logits'].dtype)
        pred_score[indices[i][1]] = output_scores[i][indices[i][0]][torch.arange(len(pred_label)), pred_label[indices[i][1]]].to(pred_score.dtype)
        score_matched[indices[i][1]] = output_scores[i][indices[i][0]][torch.arange(len(pred_label)), labels[i][indices[i][1]]].to(score_matched.dtype)
        
        for p_, s_ in zip(pred_score, score_matched):
            if s_ > p_:
                st()
        
        gt_names = [AI2THOR_CLASS_NAMES[label] for label in labels[i]]
        combined_score = [f'{pred_score_*100.0:.2f} / {score_matched_*100.0:.2f} {gt_name}' for pred_score_, score_matched_, gt_name in zip(pred_score, score_matched, gt_names)]
        pred_masks.append(pred_mask)
        pred_labels.append(pred_label)
        pred_scores.append(combined_score)
        
        
    visualize_2d_masks(
        images.cpu(), pred_masks, pred_labels, pred_scores, coco_metadata, field_name=f'matching/{prefix}',
        gt_masks=masks, gt_labels=labels
    )


def convert_mask_to_detectron_vis(im, masks, labels, coco_metadata, scores=None):
    H, W = im.shape[-2:]
    v = Visualizer(im, coco_metadata)
    predictions = Instances((H, W))
    predictions.pred_masks = masks
    predictions.pred_classes = labels
    if scores is not None:
        predictions.scores = scores
    return v.draw_instance_predictions(predictions).get_image()


def visualize_2d_masks(
    images, masks, labels, scores, coco_metadata,
    captions=None, field_name=None,
    gt_masks=None, gt_labels=None):
    """sumary_line
    
    Keyword arguments:
    images: B, V, 3, H, W
    masks: list of length B, each elemet N X V X H X W
    labels: list of length B, each element N
    """
    if wandb.run is None:
        wandb.init(project='univlg')
    
    B, V, _, H, W = images.shape
    for i in range(B):
        # images_ = []
        if len(masks[i]) == 0:
            continue
        for j in range(V):
            im = images[i, j].permute(1, 2, 0).cpu().numpy()
            
            if j > 0:
                if torch.allclose(masks[i][:, j], masks[i][:, j-1]):
                    st()
            
            try:
                vis_im = convert_mask_to_detectron_vis(
                   im, masks[i][:, j].cpu().numpy(), labels[i].cpu().numpy(),
                   coco_metadata, scores[i] if scores is not None else None)
            except:
                st()   
            if gt_masks is not None:
                gt_im = convert_mask_to_detectron_vis(
                    im, gt_masks[i][:, j].cpu().numpy(), gt_labels[i].cpu().numpy(), coco_metadata)
                vis_im = np.concatenate([vis_im, gt_im], axis=1)
            
            # v = Visualizer(im, coco_metadata)
            # predictions = Instances((H, W))
            # predictions.pred_masks = masks[i][:, j].cpu().numpy()
            # predictions.pred_classes = labels[i].cpu().numpy()
            # instance_result = v.draw_instance_predictions(predictions).get_image()
            # images_.append(instance_result)
        
            # image = np.concatenate(images_, axis=1)
            captions_ = captions[i] if captions is not None else f"i = {i}, j = {j}"
            wandb.log({field_name: wandb.Image(vis_im, caption=captions_)})


def convert_instance_to_semantic(masks, labels):
    """sumary_line

    Keyword arguments:
        masks: N, V, H, W
        labels: N
    returns:
        V, H, W
    """
    N, V, H, W = masks.shape
    semantic = torch.zeros(V, H, W, dtype=torch.long, device=masks.device)
    for i in range(N):
        semantic[masks[i] > 0] = labels[i]
    return semantic


def visualize_query_dist(query_dist, labels, subsets=None):
    """
    query_dist: list of N dist, each of length num_queries X num_classes
    labels: num_classes
    subsets: list of subset classes
    """
    if wandb.run is None:
        wandb.init(project="univlg")

    query_dist_full = np.stack(query_dist, axis=0)
    query_dist_full = query_dist_full.mean(0)  # num_queries X num_classes

    data = []
    if subsets is not None:
        for i, subset in enumerate(subsets):
            class_ids = CLASS_NAME_DICT[subset]
            for class_id in class_ids:
                data.append(
                    [f"{i}_{labels[class_id - 1]}", query_dist_full[:, class_id - 1]]
                )
    else:
        for i in range(len(labels)):
            data.append([labels[i], query_dist_full[:, i]])

    data.append(["No Object", query_dist_full[:, -1]])
    for i in range(len(query_dist_full)):
        data_ = [[data[j][0], data[j][1][i]] for j in range(len(data))]
        table = wandb.Table(data=data_, columns=["Class", "Distribution"])
        wandb.log(
            {
                f"all_charts/bar_chart_{i}": wandb.plot.bar(
                    table, "Class", "Distribution", title="Query Distribution"
                ),
            }
        )


def visualize_2d_masks_semantic(
    images, masks, thing_classes, captions=None, field_name="sem_pred", gt_masks=None
):
    # uses native wandb logging instead of detectron2 visualizer
    """sumary_line

    Keyword arguments:
        images: B, 3, H, W
        masks: B, H, W
        coco_metadata: metadata for the dataset
        gt_masks: B, H, W
    """

    if wandb.run is None:
        wandb.init(project="univlg")

    B, H, W = masks.shape
    class_labels = {i + 1: thing_classes[i] for i in range(len(thing_classes))}

    # from mask2former_video.global_vars import AI2THOR_SCANNET_CLASS_IDS, AI2THOR_ONLY_CLASS_IDS
    # class_labels = {1: "scannet", 2: "ai2thor"}
    # new_masks = torch.zeros_like(masks)
    # scannet_mask = (masks[None] == torch.tensor(AI2THOR_SCANNET_CLASS_IDS)[:, None, None, None]).any(0)
    # new_masks[scannet_mask] = 1

    # ai2thor_mask = (masks[None] == torch.tensor(AI2THOR_ONLY_CLASS_IDS)[:, None, None, None]).any(0)
    # new_masks[ai2thor_mask] = 2
    # masks = new_masks

    # gt_new_masks = torch.zeros_like(gt_masks)
    # scannet_mask = (gt_masks[None] == torch.tensor(AI2THOR_SCANNET_CLASS_IDS)[:, None, None, None]).any(0)
    # gt_new_masks[scannet_mask] = 1

    # ai2thor_mask = (gt_masks[None] == torch.tensor(AI2THOR_ONLY_CLASS_IDS)[:, None, None, None]).any(0)
    # gt_new_masks[ai2thor_mask] = 2
    # gt_masks = gt_new_masks

    for i in range(B):
        wandb.log(
            {
                f"{field_name}": wandb.Image(
                    images[i].permute(1, 2, 0).numpy(),
                    masks={
                        "predictions": {
                            "mask_data": masks[i].numpy(),
                            "class_labels": class_labels,
                        },
                        "ground_truth": {
                            "mask_data": gt_masks[i].numpy(),
                            "class_labels": class_labels,
                        },
                    },
                ),
            }
        )


def get_color_pc_from_mask(
    _mask, label, pcd, instance=False, color_map=SCANNET_COLOR_MAP_20
):
    point_select = np.zeros(_mask.shape[1], dtype=bool)
    if _mask is not None:
        for i, __mask in enumerate(_mask):
            __mask = __mask.nonzero()[0]
            point_select[__mask] = True

    masks_pcs = pcd[point_select, :]

    color_masks = np.zeros((_mask.shape[1], 3), dtype=np.float32)
    if _mask is not None:
        for i, __mask in enumerate(_mask):
            __mask = __mask.nonzero()[0]
            if instance:
                color_masks[__mask] = np.array(
                    color_map[(i + 1) % (len(color_map) - 1)]
                )
            else:
                color_masks[__mask] = np.array(
                    color_map[(label[i].item() + 1) % (len(color_map) - 1)]
                )
    masks_colors = color_masks[point_select, :]
    return masks_pcs, masks_colors


def plot_3d_offline(
    pc,
    pc_color,
    masks,
    labels,
    valids=None,
    gt_masks=None,
    gt_labels=None,
    scene_name=None,
    data_dir=None,
    mask_classes=None,
    dataset_name="scannet",
    fps_xyz=None,
):
    """
    Input:
        pc: N, 3
        pc_color: N, 3 (range: [0, 1])
        masks: M, N
        labels: M
        valids: N,
        gt_masks: M_, N
        gt_labels: M_
        scene_name: str
        data_dir: str
        mask_classes: list of classes to exclude from the visualization
        dataset_name: str
        fps_xyz: N, 3
    """
    if "ai2thor" in dataset_name:
        color_map = GENERAL_COLOR_MAP_200
    elif "scannet200" in dataset_name:
        color_map = SCANNET_COLOR_MAP_200
    else:
        color_map = SCANNET_COLOR_MAP_20

    color_map = {i: color for i, color in enumerate(color_map.values())}
    if valids is not None:
        pc = pc[valids]
        pc_color = pc_color[valids]
        if masks is not None:
            masks = masks[:, valids]
        if gt_masks is not None:
            gt_masks = gt_masks[:, valids]

    if mask_classes is not None:
        mask_classes = set(mask_classes)
        if masks is not None:
            masks = masks[labels != mask_classes]
            labels = labels[labels != mask_classes]
        if gt_masks is not None:
            gt_masks = gt_masks[gt_labels != mask_classes]
            gt_labels = gt_labels[gt_labels != mask_classes]

    v = vis.Visualizer()
    point_size = 25
    v.add_points(
        "RGB",
        pc,
        colors=pc_color * 255,
        alpha=0.8,
        visible=False,
        point_size=point_size,
    )

    if gt_masks is not None:
        masks_pcs, masks_colors = get_color_pc_from_mask(
            gt_masks, gt_labels, pc, color_map=color_map
        )
        v.add_points(
            "Semantics (200)",
            masks_pcs,
            colors=masks_colors,
            alpha=0.8,
            visible=False,
            point_size=point_size,
        )

        masks_pcs, masks_colors = get_color_pc_from_mask(
            gt_masks, gt_labels, pc, instance=True, color_map=color_map
        )
        v.add_points(
            "Instances (200)",
            masks_pcs,
            colors=masks_colors,
            alpha=0.8,
            visible=False,
            point_size=point_size,
        )

    masks_pcs, masks_colors = get_color_pc_from_mask(
        masks, labels - 1, pc, color_map=color_map
    )
    v.add_points(
        "Semantics (18)",
        masks_pcs,
        colors=masks_colors,
        visible=False,
        alpha=0.8,
        point_size=point_size,
    )

    masks_pcs, masks_colors = get_color_pc_from_mask(
        masks, labels - 1, pc, instance=True, color_map=color_map
    )
    v.add_points(
        "Instances (18)",
        masks_pcs,
        colors=masks_colors,
        visible=False,
        alpha=0.8,
        point_size=point_size,
    )

    if fps_xyz is not None:
        masks = np.eye(len(labels), dtype=bool)
        masks_pcs, masks_colors = get_color_pc_from_mask(
            masks, labels - 1, fps_xyz, color_map=color_map
        )
        v.add_points(
            "FPS",
            fps_xyz,
            colors=masks_colors,
            visible=False,
            alpha=0.8,
            point_size=200,
        )

    if data_dir is None:
        data_dir = "/path/to/language_grounding/bdetr2_visualizations"

    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)

    v.save(f"{data_dir}/{scene_name}")


def plot_only_3d(xdata, ydata, zdata, color=None, b_min=2, b_max=8, view=(45, 45)):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=200)
    ax.view_init(view[0], view[1])

    ax.set_xlim(b_min, b_max)
    ax.set_ylim(b_min, b_max)
    ax.set_zlim(b_min, b_max)

    ax.scatter3D(xdata, ydata, zdata, c=color, cmap="rgb", s=0.1)


def visualize_knn_sensor_mesh_pc(
    images, pixel_std, pixel_mean,
    scannet_color, our_pc, dists_close,
    scannet_pc, legend="", shape=None
):
    vis_images = images * pixel_std + pixel_mean
    # vis_images = vis_images.reshape(1, -1, 3, 512, 512)[0]
    
    if shape:
        vis_images = F.interpolate(
            vis_images, size=shape, mode="bilinear", align_corners=False
        )    
    else:
        vis_images = F.interpolate(
            vis_images, scale_factor=0.25, mode="bilinear", align_corners=False
        )

    vis_images = vis_images.permute(0, 2, 3, 1)
    our_color = vis_images.reshape(-1, 3).cpu().numpy()
    our_color /= 255.0
    our_color = np.clip(our_color, 0, 1)
    scannet_colors = scannet_color.reshape(-1, 3).cpu().numpy()
    scannet_colors /= 255.0
    scannet_colors = np.clip(scannet_colors, 0, 1)

    if wandb.run is None:
        wandb.init(project="univlg")

    wandb.log({
        f"{legend}our_pc": wandb.Object3D(np.concatenate([our_pc.cpu().numpy(), our_color * 255.0], axis=1)),
        f"{legend}scannet_pc": wandb.Object3D(np.concatenate([scannet_pc.cpu().numpy(), scannet_colors * 255.0], axis=1)),
        f"{legend}scannet_pc_close": wandb.Object3D(np.concatenate([scannet_pc[dists_close].cpu().numpy(), scannet_colors[dists_close.cpu().numpy()] * 255.0], axis=1)),
        f"{legend}scannet_pc_far": wandb.Object3D(np.concatenate([scannet_pc[~dists_close].cpu().numpy(), scannet_colors[~dists_close.cpu().numpy()] * 255.0], axis=1)) if (~dists_close).sum().item() != 0 else None, 
        f"{legend}image": wandb.Image(vis_images[0].cpu().numpy())
    })

    # xdata, ydata, zdata = our_pc.T.cpu().numpy()
    # plot_only_3d(xdata, ydata, zdata, our_color)
    # plt.savefig('our_pc.png')

    # xdata, ydata, zdata = scannet_pc.T.cpu().numpy()
    # plot_only_3d(xdata, ydata, zdata, scannet_colors)
    # plt.savefig('scannet_pc.png')

    # xdata, ydata, zdata = scannet_pc[dists_close].T.cpu().numpy()
    # plot_only_3d(xdata, ydata, zdata, scannet_colors[dists_close.cpu().numpy()])
    # plt.savefig('scannet_pc_close.png')

    # if (~dists_close).sum().item() != 0:
    #     xdata, ydata, zdata = scannet_pc[~dists_close].T.cpu().numpy()
    #     plot_only_3d(xdata, ydata, zdata, scannet_colors[~dists_close.cpu().numpy()])
    #     plt.savefig('scannet_pc_far.png')

    # print percentage of points that are close
    print("Percentage of points that are close: ", dists_close.sum().item() / dists_close.shape[0])


if __name__ == "__main__":
    feature_folders = [
        # "/path/to/language_grounding/feature_vis/2d",
        # "/path/to/language_grounding/feature_vis/no_tri_no_vox"
        # '/path/to/language_grounding/feature_vis/2d_single_view',
        "/path/to/language_grounding/feature_vis/3d_single_view_float64",
        "/path/to/language_grounding/feature_vis/3d_single_view_float64_rerun",
        # 3d_single_view_float64_rerun
    ]
    visualize_pca_from_feature_folders(feature_folders)
