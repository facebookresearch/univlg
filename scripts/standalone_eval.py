import gc
import logging
import os
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

from datetime import datetime
import detectron2.utils.comm as comm
import ipdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyviz3d.visualizer as viz
import torch
import torch.distributed
import torch.multiprocessing as mp
import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import _log_api_usage
from einops import rearrange
from univlg import (ReferrentialGroundingEvaluator,
                               Scannet3DEvaluator, ScannetDatasetMapper,
                               ScannetSemantic3DEvaluator, Sr3dDatasetMapper,
                               VQAEvaluator, add_maskformer2_config,
                               add_maskformer2_video_config,
                               build_detection_test_loader,
                               build_detection_train_loader,
                               build_detection_train_loader_multi_task,
                               get_detection_dataset_dicts)
from univlg.data_video.build import (
    get_multiple_train_3d_dataset_dicts, merge_datasets)
from univlg.data_video.data_utils import get_multiview_xyz
# MaskFormer
from univlg.global_vars import SCANNET_LIKE_DATASET
from univlg.modeling.backproject.backproject import \
    multiscsale_voxelize
from univlg.utils.vis_utils import visualize_query_dist
from univlg.video_maskformer_model import UniVLG
from torch.nn.parallel import DistributedDataParallel
from torch_scatter import scatter_mean

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


class Trainer:
    @classmethod
    def build_evaluator(
        cls,
        cfg,
        dataset_name,
        output_folder=None,
        use_2d_evaluators_only=False,
        use_3d_evaluators_only=False,
        use_refexp_evaluator_only=False,
    ):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluators = []

        if cfg.TEST.EVAL_3D and cfg.MODEL.DECODER_3D and not use_2d_evaluators_only:
            if 'scanqa' in dataset_name or 'sqa3d' in dataset_name:
                evaluators.append(VQAEvaluator(
                    dataset_name=dataset_name,
                    evaluate_detection="scanqa" in dataset_name.lower(),
                    cfg=cfg
                ))
                return evaluators
            if 'ref' in dataset_name or use_refexp_evaluator_only:
                evaluators.append(ReferrentialGroundingEvaluator(
                    dataset_name,
                    thresholds=[0.25, 0.5, 0.75],
                    topks=[1, 2, 5],
                    cfg=cfg
                ))
                return evaluators
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                evaluators.append(
                    ScannetSemantic3DEvaluator(
                        dataset_name,
                        output_dir=output_folder,
                        eval_sparse=cfg.TEST.EVAL_SPARSE,
                        cfg=cfg,
                    )
                )
                if cfg.USE_CLASSIFICATION_ONLY_LOSS:
                    evaluators.append(
                        ScannetSemantic3DEvaluator(
                            dataset_name,
                            output_dir=output_folder,
                            eval_sparse=cfg.TEST.EVAL_SPARSE,
                            cfg=cfg,
                            cls_only_logits=True,
                        )
                    )
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                evaluators.append(
                    Scannet3DEvaluator(
                        dataset_name,
                        output_dir=output_folder,
                        eval_sparse=cfg.TEST.EVAL_SPARSE,
                        cfg=cfg,
                    )
                )
        return evaluators

    @classmethod
    def build_train_loader(cls, cfg):
        print("build_train_loader...")
        dataset_dicts = get_multiple_train_3d_dataset_dicts(cfg)
        return build_detection_train_loader(cfg, mapper=None, dataset=dataset_dicts)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, cortex_cfg=None):
        print(f"build_test_loader: {dataset_name}, start method: {mp.get_start_method()}")
        
        scannet_like = False
        for scannet_like_dataset in SCANNET_LIKE_DATASET:
            if scannet_like_dataset in dataset_name and 'ref' not in dataset_name:
                scannet_like = True
                break

        if scannet_like:
            dataset_dict = get_detection_dataset_dicts(
                [dataset_name],
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[
                        list(cfg.DATASETS.TEST).index(dataset_name)
                    ]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subsample_data=cfg.TEST.SUBSAMPLE_DATA
                if dataset_name in cfg.DATASETS.TEST_SUBSAMPLED
                else None,
            )
            mapper = ScannetDatasetMapper(
                cfg,
                is_train=False,
                dataset_name=dataset_name,
                dataset_dict=dataset_dict,
                decoder_3d=False
                if dataset_name in cfg.DATASETS.TEST_2D_ONLY
                else cfg.MODEL.DECODER_3D,
            )
            return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)
        elif "ref" in dataset_name:
            dataset_dict = get_detection_dataset_dicts(
                [dataset_name],
                proposal_files=[
                    cfg.DATASETS.PROPOSAL_FILES_TEST[
                        list(cfg.DATASETS.TEST).index(dataset_name)
                    ]
                ]
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
                subsample_data=cfg.TEST.SUBSAMPLE_DATA
                if dataset_name in cfg.DATASETS.TEST_SUBSAMPLED
                else None,
            )
            mapper = Sr3dDatasetMapper(
                cfg,
                is_train=False,
                dataset_name=dataset_name,
                scannet_dict=dataset_dict[1],
                scene_to_id_map=dataset_dict[2],
                decoder_3d=False
                if dataset_name in cfg.DATASETS.TEST_2D_ONLY
                else cfg.MODEL.DECODER_3D,
            )
            return build_detection_test_loader(
                cfg, mapper=mapper, dataset=dataset_dict[0]
            )
        else:
            raise NotImplementedError

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()

        dataset_names = list(cfg.DATASETS.TEST)
        
        if "cortex" in dataset_names:
            cortex_cfg_dict = split_cortex_datasets(cfg)
            dataset_names.remove("cortex")
            dataset_names += list(cortex_cfg_dict.keys())
        else:
            cortex_cfg_dict = {}

        for idx, dataset_name in enumerate(dataset_names):
            print(f"Evaluating on {dataset_name}, idx: {idx}")
            data_loader = cls.build_test_loader(
                cfg,
                dataset_name,
                cortex_cfg_dict[dataset_name]
                if dataset_name in cortex_cfg_dict
                else None,
            )

            if cfg.DATALOADER_ONLY:
                from tqdm import tqdm
                for data in tqdm(data_loader):
                    pass
                continue
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(
                        cfg,
                        dataset_name,
                        use_2d_evaluators_only=dataset_name in cfg.DATASETS.TEST_2D_ONLY
                        if cfg.MULTI_TASK_TRAINING
                        else False,
                        use_3d_evaluators_only=dataset_name in cfg.DATASETS.TEST_3D_ONLY
                        if cfg.MULTI_TASK_TRAINING
                        else False,
                        use_refexp_evaluator_only=dataset_name in cfg.DATASETS.TEST_REFEXP_ONLY,
                    )
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i

        if cfg.VISUALIZE_QUERY_DIST:
            visualize_query_dist(
                model.query_dist_tracker,
                model.metadata.thing_classes,
                cfg.EVALUATE_SUBSET,
            )
            model.query_dist = []

        gc.collect()
        torch.cuda.empty_cache()

        results_structured = {}
        for dataset_name in dataset_names:
            if dataset_name in cfg.DATASETS.TEST_2D_ONLY:
                suffix = "train" if "train_eval" in dataset_name else "val"
            else:
                suffix = (
                    "train_full" if "train_eval" in dataset_name else "val_full"
                )

            suffix += f'_{dataset_name.split("_")[0]}'
            results_val = results[dataset_name].copy()
            results_val = {
                f'{suffix}' + k: v
                for k, v in results_val.items()
            }
            results_structured.update(results_val)
        return results_structured



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

def convert_bbox_to_corners_with_colors(bboxes):
    """
    Convert bounding boxes to a format with 8 corners and deterministically generate a color for each box.

    Args:
    - bboxes (np.array): An array of shape [N, 6] containing bounding boxes.

    Returns:
    - np.array: An array of dictionaries, each with 'corners' and 'color' keys.
    """
    converted = []
    colors = get_color(len(bboxes))
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        corners = np.array([
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
        ])
        converted.append({"corners": corners.tolist(), "color": colors[idx]})
    return np.array(converted)

def crop_pcd_to_combined_bbox(pcd: torch.Tensor, bboxes: torch.Tensor, extra_extent: float) -> torch.Tensor:
    """
    Crop point cloud data (PCD) to match the combined extent of all bounding boxes.

    Parameters:
    - pcd: torch.Tensor of shape [M, 6] representing the point cloud data with [xyz rgb].
    - bboxes: torch.Tensor of shape [N, 6] representing the bounding boxes with [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
    - torch.Tensor: Cropped PCD matching the combined extent of all bounding boxes.
    """
    # Calculate the min/max extent of all bounding boxes
    combined_min = torch.min(bboxes[:, :3], dim=0).values - extra_extent
    combined_max = torch.max(bboxes[:, 3:], dim=0).values + extra_extent

    # Check if points are within the combined bbox extents
    in_combined_bbox = (pcd[:, 0] >= combined_min[0]) & (pcd[:, 0] <= combined_max[0]) & \
                        (pcd[:, 1] >= combined_min[1]) & (pcd[:, 1] <= combined_max[1]) & \
                        (pcd[:, 2] >= combined_min[2]) & (pcd[:, 2] <= combined_max[2])

    return pcd[in_combined_bbox]

def sample_k_rows(tensor, K):
    indices = torch.randperm(tensor.size(0))[:K]
    return tensor[indices]


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
    pc, color, caption, pred_bbox, data_dir=None,
):
    """
    Visualize a point cloud and its predicted bounding box.
    
    Parameters:
      pc: N x 3 numpy array representing the point cloud.
      color: N x 3 numpy array (0-255) with colors corresponding to each point.
      pred_bbox: numpy array representing a bounding box in [xmin, ymin, zmin, xmax, ymax, zmax] format.
      data_dir: (optional) Base directory to save the visualization.
      sample_name: (optional) Name of the sample (used to structure the output directory).
      inputs: (optional) List of dictionaries containing metadata (e.g., 'dataset_name').
    """
    point_size = 25
    v = viz.Visualizer()
    v.add_points("RGB", pc, colors=color, alpha=0.8, visible=True, point_size=point_size)

    # Convert predicted bounding box to center-size format and add to visualization
    if pred_bbox is not None:
        pred_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(pred_bbox)).numpy()
        v.add_bounding_box(
            "Boxes (Pred)",
            position=pred_bbox[..., :3][0],
            size=pred_bbox[..., 3:][0],
            color=np.array([255, 0, 0]),
            alpha=0.8,
            visible=True,
            edge_width=0.03
        )

    data_dir = Path(data_dir)
    datetime_str =  datetime.now().strftime("%Y_%m_%d-%H_%M_%S.%f")[:-3]
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

def fwd(cfg, model):
    use_data = True
    assert not cfg.USE_SEGMENTS
    device = next(model.parameters()).device
    if use_data:
        data = torch.load("docs/wip/ignore/data.pth")
        images, multiview_data, scannet_pc, scannet_p2v, captions, max_valid_points, shape = data["images"], data["multiview_data"], data["scannet_pc"], data["scannet_p2v"], data["captions"], data["max_valid_points"], data["shape"]
        bs, v, H_padded, W_padded = shape #  (2, 15, 448, 448)
        images_tensor = images.tensor # torch.Size([30, 3, 448, 448])
        multi_scale_xyz = multiview_data["multi_scale_xyz"]
        multi_scale_p2v = multiview_data["multi_scale_p2v"]
    else:
        bs, v, H_padded, W_padded = 1, 15, 448, 448
        max_valid_points = 100000
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
        max_valid_points = [max_valid_points]

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

    for i in range(bs):
        viz_pc = rearrange(multi_scale_xyz[0][[i]], "b v h w c -> (b v h w) c").cpu()
        viz_color = torch.zeros_like(viz_pc)
        # TODO: Get the right bounding box. We need do a dot product w/tokens and get the root token(s).
        visualize_pc_masks_and_bbox(
            pc=viz_pc.numpy(),
            color=viz_color.numpy(),
            caption=captions[i],
            pred_bbox=outputs["pred_boxes"][i],
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
    model = create_ddp_model(
        model,
        broadcast_buffers=False,
        find_unused_parameters=cfg.MULTI_TASK_TRAINING
        or cfg.FIND_UNUSED_PARAMETERS,
    )
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    try:
        from univlg.utils.decoupled_utils import \
            set_global_breakpoint
        set_global_breakpoint()
        res = fwd(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError

        if wandb.run is not None:
            wandb.finish()
        return res
    except Exception as e:
        import traceback
        print(f"Exception: {e}")
        print(traceback.format_exc())
        breakpoint(traceback=e.__traceback__)
        raise e
    finally:
        pass



if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    args = default_argument_parser().parse_args()
    print(f"Opts: {args.opts}")
    print("Command Line Args:", args)

    breakpoint()
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.backends.cudnn.deterministic = True  #needed
    # torch.use_deterministic_algorithms(True, warn_only=True)

    _launcher = "main"
    _kwargs = dict()
    
    if "launcher=" in args.opts[0]:
        _launcher = args.opts[0].split("=")[1]
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
