# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from datetime import datetime
from pathlib import Path

import detectron2.utils.comm as comm
import ipdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyviz3d.visualizer as viz
import torch
import torch.distributed
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import _log_api_usage
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel
from torch_scatter import scatter_mean

import wandb
from univlg import add_maskformer2_config, add_maskformer2_video_config
from univlg.data_video.data_utils import get_multiview_xyz
from univlg.data_video.sentence_utils import convert_grounding_to_od_logits_ref
from univlg.modeling.backproject.backproject import multiscsale_voxelize
from univlg.utils.decoupled_utils import breakpoint_on_error
from univlg.utils.misc import nanmax, nanmin

warnings.filterwarnings("ignore")

st = ipdb.set_trace


def create_ddp_model(
    model, *, fp16_compression=False, find_unused_parameters=False, **kwargs
):
    """
    Create a DistributedDataParallel model if there are >1 processes.

    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """  # noqa W605
    if comm.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(
        model, **kwargs, find_unused_parameters=find_unused_parameters
    )
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import \
            default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_color(max_value: int, colormap='spring'):
    colormap = plt.get_cmap('spring')  # Pink is 0, Yellow is 1
    colors = [mcolors.to_rgb(colormap(i / max_value)) for i in range(max_value)]  # Generate colors
    return (np.array(colors) * 255).astype(int).tolist()

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    x_c = 0.5 * (x0 + x1)
    y_c = 0.5 * (y0 + y1)
    z_c = 0.5 * (z0 + z1)
    w = x1 - x0
    h = y1 - y0
    d = z1 - z0
    return torch.stack([x_c, y_c, z_c, w, h, d], dim=-1)

def visualize_pc_masks_and_bbox(
    pc, color, captions=None, pred_bboxs=None, pred_masks=None, mask_pc=None, data_dir=None,
):
    """
    Visualize a point cloud and its predicted bounding box.
    
    Parameters:
      pc: N x 3 numpy array representing the point cloud.
      color: N x 3 numpy array (0-255) with colors corresponding to each point.
      captions: List of strings containing the captions for each point cloud.
      pred_bboxs: M x 6 numpy array representing a bounding box in [xmin, ymin, zmin, xmax, ymax, zmax] format.
      pred_masks: M x N bool numpy array representing a mask for M objects.
      data_dir: (optional) Base directory to save the visualization.
      sample_name: (optional) Name of the sample (used to structure the output directory).
      inputs: (optional) List of dictionaries containing metadata (e.g., 'dataset_name').
    """
    pred_colors = get_color(len(pred_bboxs))
    point_size = 25

    v = viz.Visualizer()
    v.add_points("RGB", pc, colors=color, alpha=0.8, visible=True, point_size=point_size)
    v.add_labels(
        'Labels',
        [captions],
        [np.array([1.0, 0.0, 0.0])],
        [np.array([255.0, 0.0, 0.0])],
        visible=True
    )

    # Convert predicted bounding box to center-size format and add to visualization
    if pred_bboxs is not None:
        pred_bboxs = torch.from_numpy(pred_bboxs) if isinstance(pred_bboxs, np.ndarray) else pred_bboxs
        pred_bboxs = box_xyzxyz_to_cxcyczwhd(pred_bboxs).cpu().numpy()
        for i in range(pred_bboxs.shape[0]):
            v.add_bounding_box(
                f"Boxes (Pred) {i}",
                position=pred_bboxs[..., :3][i],
                size=pred_bboxs[..., 3:][i],
                color=np.array(pred_colors[i]),
                alpha=0.8,
                visible=True,
                edge_width=0.03
            )

    if pred_masks is not None:
        for i in range(pred_masks.shape[0]):
            if mask_pc[pred_masks[i]].shape[0] == 0:
                print(f"Mask {i} is empty")
                continue

            v.add_points(
                f"Masks (Pred) {i}",
                mask_pc[pred_masks[i]],
                colors=np.array(pred_colors[i])[None].repeat(mask_pc[pred_masks[i]].shape[0], axis=0),
            )

    data_dir = Path(data_dir)
    datetime_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")[:-3]
    data_dir = data_dir / datetime_str
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved to {data_dir}")
    v.save(str(data_dir))


def load_3d_data(cfg, batched_inputs, images_shape, device):
    valids = None
    multiview_data = None
    bs, v = images_shape[:2]
    
    multiview_data = {}
    multiview_data["multi_scale_xyz"] = [
        torch.stack(
            [batched_inputs[i]["multi_scale_xyz"][j] for i in range(bs)], dim=0
        ).to(device)
        for j in range(len(batched_inputs[0]["multi_scale_xyz"]))
    ]

    voxel_size = cfg.INPUT.VOXEL_SIZE[::-1]

    if cfg.INPUT.VOXELIZE:
        multiview_data["multi_scale_p2v"] = multiscsale_voxelize(
            multiview_data["multi_scale_xyz"], voxel_size
        )
    return valids, multiview_data

def process_lang_data(cfg, tokenizer, dataset_dict):
    from univlg.data_video.datasets.ref_coco_utils import (consolidate_spans,
                                                           get_root_and_nouns)
    from univlg.data_video.sentence_utils import (create_positive_map,
                                                  get_positive_tokens)
    
    utterance = dataset_dict["utterance"].lower()
    target_id = dataset_dict["target_id"]
    anchor_ids = dataset_dict['anchor_ids']
    anchor_names = dataset_dict['anchors_types']

    if dataset_dict.get("target_str", None) is not None:
        tokens_positive = get_positive_tokens(utterance, [dataset_dict["target_str"]])
    else:
        _, _, root_spans_spacy, _ = get_root_and_nouns(utterance)
        tokens_positive = [consolidate_spans(root_spans_spacy, utterance)]

    dataset_dict["tokens_positive"] = tokens_positive

    max_len = (
        cfg.MODEL.MAX_SEQ_LEN
        if not cfg.TEXT_ENCODER_TYPE == "clip"
        else 77
    )
    tokenized = tokenizer(
        utterance, return_tensors="pt", max_length=max_len, truncation=True
    )

    positive_map = create_positive_map(
        tokenized, tokens_positive, max_query_len=max_len
    )

    return {
        "text_caption": utterance,
        "target_id": target_id,
        "anchor_ids": anchor_ids,
        "anchors_names": anchor_names,
        "tokens_positive": tokens_positive,
        "tokenized": tokenized,
        "positive_map": positive_map,
        "positive_map_od": None,
        "annotation_id": dataset_dict['ann_id'] if "ann_id" in dataset_dict else None
    }

def get_pred_logits(cfg, lang_data, outputs):
    bs = len(lang_data)
    num_classes = max([len(lang_data[i]['anchor_ids']) + 1 for i in range(len(lang_data))]) # Root noun + anchor ids
    outputs["pred_logits"] = outputs["pred_logits"].sigmoid()
    reduce = "mean"
    outputs["pred_logits"] = torch.cat(
        [
            convert_grounding_to_od_logits_ref(
                logits=outputs["pred_logits"][i][None],
                num_class=num_classes + 1,
                positive_maps=lang_data[i]["positive_map"],
                reduce=reduce,
            )
            for i in range(bs)
        ]
    )
    outputs["pred_scores"] = outputs["pred_logits"]
    return outputs

def get_pred_boxes(outputs, scannet_pc, max_valid_points):
    masks = outputs['pred_masks'] > 0

    # remove padded tokens
    for j in range(len(max_valid_points)):
        masks[j, :, max_valid_points[j]:] = False

    pc = scannet_pc
    pc = pc[:, None].repeat(1, masks.shape[1], 1, 1)
    pc[torch.where(masks == 0)] = torch.nan

    boxes = torch.cat([
        nanmin(pc, dim=2)[0], nanmax(pc, dim=2)[0]
    ], 2)

    # if only one point is in the mask, the box will still be too small
    boxes[masks.sum(2) <= 1] = torch.tensor([0, 0, 0, 1e-2, 1e-2, 1e-2], device=boxes.device)
    outputs['pred_boxes'] = boxes
    return outputs


def get_dummy_data(cfg, device):
    bs, v, H_padded, W_padded = 1, 15, 448, 448
    max_valid_points = [100000]
    captions = ["This is a test caption."]
    images_tensor = torch.zeros(bs, v, 3, H_padded, W_padded).to(device) # [B, V, C, H, W]
    images_tensor = rearrange(images_tensor, "b v c h w -> (b v) c h w")
    depths = torch.zeros(bs, v, H_padded, W_padded).to(device) # [B, V, H, W]
    poses = torch.eye(4).repeat(bs, v, 1, 1).to(device) # tensor [B, V, 4, 4]
    intrinsics = torch.eye(4).repeat(bs, v, 1, 1).to(device) # tensor [B, V, 4, 4]
    align_matrix = torch.eye(4).to(device)
    is_train = False

    assert bs == 1
    batched_inputs = []
    for i in range(bs):
        multi_scale_xyz, scannet_pc, original_xyz = get_multiview_xyz(
            shape=(v, H_padded, W_padded),
            size_divisibility=cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            depths=[x for x in depths[0]],
            poses=[x for x in poses[0]],
            intrinsics=[x for x in intrinsics[0]],
            is_train=is_train,
            augment_3d=cfg.INPUT.AUGMENT_3D,
            interpolation_method=cfg.MODEL.INTERPOLATION_METHOD,
            mask_valid=cfg.MASK_VALID,
            mean_center=cfg.MEAN_CENTER,
            do_rot_scale=cfg.DO_ROT_SCALE,
            scannet_pc=None,
            align_matrix=align_matrix,
            vil3d=cfg.VIL3D,
            scales=cfg.MULTIVIEW_XYZ_SCALES,
        )
        multiview_data = {
            "multi_scale_xyz": multi_scale_xyz,
        }
        batched_inputs.append(multiview_data)

    _, multiview_data = load_3d_data(
        cfg,
        batched_inputs,
        images_shape=[bs, v, H_padded, W_padded],
        device=images_tensor.device
    )

    multi_scale_xyz = multiview_data["multi_scale_xyz"]
    multi_scale_p2v = multiview_data["multi_scale_p2v"]
    mask_features_xyz = [x.reshape(bs, -1, 3) for x in multi_scale_xyz]
    mask_features_p2v = [x.reshape(bs, -1) for x in multi_scale_p2v]
    scannet_pc = mask_features_xyz[0]
    scannet_p2v = mask_features_p2v[0]

    bs, v, H_padded, W_padded = 1, 15, 448, 448
    shape = (bs, v, H_padded, W_padded)
    # Return all needed variables
    return images_tensor, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape

def get_saved_data(cfg):
    output_path = Path('ckpts') / 'misc' / 'data_sample.pth'
    data = torch.load(output_path)
    images_tensor = data["images_tensor"]
    multiview_data = data["multiview_data"]
    scannet_pc = data["scannet_pc"]
    scannet_p2v = data["scannet_p2v"]
    captions = data["captions"]
    max_valid_points = data["max_valid_points"]
    shape = data["shape"]
    
    return images_tensor, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape

@torch.inference_mode()
def fwd(cfg, model):
    use_data = True
    assert not cfg.USE_SEGMENTS
    device = next(model.parameters()).device
    max_bs = 4
    
    images_tensor, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape = get_saved_data(cfg) if use_data else get_dummy_data(cfg, device)
    bs, v, H_padded, W_padded = shape

    if max_bs is not None and max_bs < bs:
        images_tensor = rearrange(rearrange(images_tensor, '(bs v) ... -> bs v ...', bs=bs)[:max_bs], 'bs v ... -> (bs v) ...')
        for k in multiview_data.keys():
            for i in range(len(multiview_data[k])):
                multiview_data[k][i] = multiview_data[k][i][:max_bs]

        scannet_pc = scannet_pc[:max_bs]
        scannet_p2v = scannet_p2v[:max_bs]
        captions = captions[:max_bs]
        max_valid_points = max_valid_points[:max_bs]
        bs = max_bs
    
    
    # Get multi_scale data from multiview_data
    multi_scale_xyz = multiview_data["multi_scale_xyz"]
    multi_scale_p2v = multiview_data["multi_scale_p2v"]

    tokenizer = model.mask_decoder.lang_encoder.tokenizer
    lang_data = []
    
    # To automatically find the root noun, set each element to None. Alternatively, specify the target string for each element.
    target_strs = [None] * bs
    for i in range(bs):
        if target_strs[i] is not None: assert target_strs[i] in captions[i]
        _dataset_dict = {
            "utterance": captions[i], "target_str": target_strs[i], "target_id": -1, "anchor_ids": [], "anchors_types": [],
        }
        lang_data.append(process_lang_data(cfg, tokenizer, _dataset_dict))

    mask_features, multi_scale_features = model.visual_backbone(
        images=images_tensor, # torch.Size([30, 3, 448, 448])
        multi_scale_xyz=multi_scale_xyz, # [torch.Size([2, 15, 32, 32, 3]), ...]
        multi_scale_p2v=multi_scale_p2v, # [torch.Size([2, 15360]), ...]
        shape=[bs, v, H_padded, W_padded],
        decoder_3d=True,
        actual_decoder_3d=True,
        mesh_pc=scannet_pc, # torch.Size([2, 38882, 3])
        mesh_p2v=scannet_p2v # torch.Size([2, 38882])
    )

    scannet_pc = scatter_mean(scannet_pc, scannet_p2v, dim=1)
    scannet_p2v = (
        torch.arange(scannet_pc.shape[1], device=scannet_pc.device)
        .unsqueeze(0)
        .repeat(scannet_pc.shape[0], 1)
    )

    outputs = model.mask_decoder(
        mask_features, # torch.Size([2, 256, 38882, 1])
        shape=[bs, v],
        mask_features_xyz=scannet_pc,
        mask_features_p2v=scannet_p2v,
        segments=None,
        decoder_3d=True,
        captions=captions,
        actual_decoder_3d=True,
        scannet_all_masks_batched=None,
        max_valid_points=max_valid_points,
        tokenized_answer=None,
    )

    outputs = get_pred_boxes(outputs, scannet_pc, max_valid_points)
    outputs = get_pred_logits(cfg, lang_data, outputs)

    assert outputs['pred_scores'].ndim == 3 # (bs, queries, num_classes + 1)
    scores = outputs['pred_scores'][:, :, 0] # Get the root noun (always first)
    downsampmed_images_tensor = F.interpolate(images_tensor, size=(24, 32), mode='bilinear', align_corners=False)
    viz_color = rearrange(downsampmed_images_tensor, "(bs v) c h w -> bs (v h w) c", bs=bs, v=v)
    
    for i in range(bs):
        viz_pc = rearrange(multi_scale_xyz[-1][[i]], "b v h w c -> (b v h w) c").cpu()
        assert multi_scale_xyz[-1][[i]].shape[2:4] == (24, 32)

        pred_mask = outputs['pred_masks'][i]
        pred_mask = pred_mask[:, scannet_p2v[i]]
        # remove padding
        max_valid_point = max_valid_points[i]
        pred_mask = pred_mask[:, :max_valid_point]
        masks = F.sigmoid(pred_mask) > 0.5
        
        bboxes = outputs["pred_boxes"][i]
        
        # Average confidence of each prediction mask.
        top_k_weighted_scores = scores[i]
        max_k = 3
        top_ids = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        top_masks = masks[top_ids, :].cpu().numpy()
        top_bboxes = bboxes[top_ids, :].cpu().numpy()
        
        visualize_pc_masks_and_bbox(
            pc=viz_pc.numpy(),
            color=(viz_color[i].cpu().numpy() * 255).astype(np.uint8),
            captions=captions[i],
            pred_bboxs=top_bboxes,
            pred_masks=top_masks,
            mask_pc=scannet_pc[i].cpu().numpy(),
            data_dir='outputs',
        )


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    print(f"World size: {comm.get_world_size()}")

    # Currently, each rank will run inference on the same data but this can be modified.
    model = create_ddp_model(
        model,
        broadcast_buffers=False,
        find_unused_parameters=cfg.MULTI_TASK_TRAINING
        or cfg.FIND_UNUSED_PARAMETERS,
    )

    model.eval()
    model.requires_grad_(False)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    with breakpoint_on_error():
        res = fwd(cfg, model)
        if wandb.run is not None:
            wandb.finish()
        return res

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    args = default_argument_parser().parse_args()
    print(f"Opts: {args.opts}")
    print("Command Line Args:", args)

    _kwargs = dict()
    
    if "launcher=" in args.opts[0]:
        args.opts = args.opts[1:]

    launcher = launch
    _kwargs["args"] = (args,)
    
    # this is needed to prevent memory leak in conv2d layers
    # see: https://github.com/pytorch/pytorch/issues/98688#issuecomment-1869290827
    os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
    launcher(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        **_kwargs
    )