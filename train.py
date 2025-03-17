# Modified from: https://github.com/facebookresearch/Mask2Former/blob/main/train_net_video.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import socket
import copy
import gc
import itertools
import logging
import os
import time
import warnings
import weakref
import contextlib
from collections import OrderedDict
from typing import Any, Dict, List, Set
import contextlib
import logging
import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import detectron2.utils.comm as comm
import torch.distributed
import wandb
from viztracer import VizTracer
import detectron2.utils.comm as comm
import ipdb
import torch
import wandb
import torch.multiprocessing as mp
from viztracer import VizTracer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.engine.defaults import hooks
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, inference_on_dataset
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from fvcore.nn.precise_bn import get_bn_modules
from univlg import (
    COCOEvaluatorMemoryEfficient,
    Scannet3DEvaluator,
    ReferrentialGroundingEvaluator,
    RefCOCOEvaluator,
    ScannetDatasetMapper,
    ScannetSemantic3DEvaluator,
    VQAEvaluator,
    ScanqaDatasetMapper,
    Sqa3dDatasetMapper,
    Sr3dDatasetMapper,
    RefCocoDatasetMapper,
    add_maskformer2_config,
    add_maskformer2_video_config,
    build_detection_test_loader,
    build_detection_train_loader,
    build_detection_train_loader_multi_task,
    get_detection_dataset_dicts,
)
from univlg.data_video.build import (
    get_multiple_train_2d_dataset_dicts,
    get_multiple_train_3d_dataset_dicts,
    merge_datasets,
)

# MaskFormer
from univlg.data_video.dataset_mapper_coco import (
    COCOInstanceNewBaselineDatasetMapper,
)
from univlg.global_vars import SCANNET_LIKE_DATASET
from univlg.utils.vis_utils import visualize_query_dist
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


st = ipdb.set_trace

class OneCycleLr_D2(torch.optim.lr_scheduler.OneCycleLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}
    
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
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        # super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(
            model,
            broadcast_buffers=False,
            find_unused_parameters=cfg.MULTI_TASK_TRAINING
            or cfg.FIND_UNUSED_PARAMETERS,
        )
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

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
        
        if (cfg.TEST.EVAL_2D or cfg.EVAL_PER_IMAGE) and not use_3d_evaluators_only:
            if 'refcoco' in dataset_name:
                print(f"Adding RefCOCO Evaluator for {dataset_name}")
                evaluators.append(
                    RefCOCOEvaluator(
                        dataset_name, 
                        thresholds=[0.25, 0.5, 0.75],
                        topks=[1, 5, 10],
                        cfg=cfg
                    )
                )
            else:
                if cfg.INPUT.ORIGINAL_EVAL:
                    print("Using original COCO Eval, potentially is RAM hungry")
                    evaluators.append(
                        COCOEvaluator(
                            dataset_name, output_dir=output_folder, use_fast_impl=False
                        )
                    )
                else:
                    evaluators.append(
                        COCOEvaluatorMemoryEfficient(
                            dataset_name,
                            output_dir=output_folder,
                            use_fast_impl=False,
                            per_image_eval=cfg.EVAL_PER_IMAGE,
                            evaluate_subset=cfg.EVALUATE_SUBSET,
                        )
                    )
        return evaluators

    @classmethod
    def build_train_loader(cls, cfg):
        print("build_train_loader...")
        dataset_dicts_3d = get_multiple_train_3d_dataset_dicts(cfg)
        dataset_dicts_2d = get_multiple_train_2d_dataset_dicts(cfg)
        
        if cfg.MULTI_TASK_TRAINING:
            return build_detection_train_loader_multi_task(
                cfg,
                mapper_3d=None,
                mapper_2d=None,
                dataset_3d=dataset_dicts_3d,
                dataset_2d=dataset_dicts_2d,
            )
        else:
            dataset_dicts = [dataset_dicts_3d, dataset_dicts_2d]
            mappers = [None, None]
            dataset = merge_datasets(dataset_dicts, mappers, balance=True, dataset_mul=cfg.DATASET_MUL)
            return build_detection_train_loader(cfg, mapper=None, dataset=dataset)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
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
        elif 'refcoco' in dataset_name:
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
            mapper = RefCocoDatasetMapper(
                cfg,
                is_train=False,
                dataset_name=dataset_name,
                decoder_3d=cfg.FORCE_DECODER_3D and not cfg.PSEUDO_2D_AUG
                if dataset_name in cfg.DATASETS.TEST_2D_ONLY
                else cfg.MODEL.DECODER_3D,
            )
            return build_detection_test_loader(cfg, mapper=mapper, dataset=dataset_dict)
        elif "coco" in dataset_name or "sam" in dataset_name or 'paco' in dataset_name:
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
            mapper = COCOInstanceNewBaselineDatasetMapper(
                cfg, is_train=False, dataset_name=dataset_name,
                decoder_3d=cfg.FORCE_DECODER_3D and not cfg.PSEUDO_2D_AUG
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

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
            hooks.LRScheduler(self.optimzer_2d, self.scheduler_2d)
            if cfg.SEPERATE_OPTIMIZERS
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD,
                    max_to_keep=15,
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        if cfg.SOLVER.LR_SCHEDULER_NAME == "onecyclelr":
            return OneCycleLr_D2(
                optimizer,
                max_lr=cfg.SOLVER.BASE_LR,
                total_steps=cfg.SOLVER.MAX_ITER,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()

        print(summary(model))

        panet_resnet_layers = ["cross_view_attn", "res_to_trans", "trans_to_res"]
        panet_swin_layers = [
            "cross_view_attn",
            "cross_layer_norm",
            "res_to_trans",
            "trans_to_res",
        ]
        panet_dino_layers = [
            "cross_view_attn",
            "token_to_trans",
            "trans_to_token",
        ]

        if cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone":
            backbone_panet_layers = panet_resnet_layers
        elif cfg.MODEL.BACKBONE.NAME == "D2SwinTransformer":
            backbone_panet_layers = panet_swin_layers
        elif cfg.MODEL.BACKBONE.NAME == "DINOv2":
            backbone_panet_layers = panet_dino_layers
        else:
            raise NotImplementedError

        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    # panet layers are initialize from scratch so use default lr
                    panet_found = False
                    for panet_name in backbone_panet_layers:
                        if panet_name in module_name:
                            hyperparams["lr"] = hyperparams["lr"]
                            panet_found = True
                            break

                    if not panet_found:
                        hyperparams["lr"] = (
                            hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                        )
                        if cfg.NOT_USE_WD_PRETRAINED:
                            assert not cfg.MODEL.FREEZE_BACKBONE, "Won't work"
                            hyperparams["weight_decay"] = 0.0
                            print("Not using weight decay for backbone")
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

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
        

        for idx, dataset_name in enumerate(dataset_names):
            print(f"Evaluating on {dataset_name}, idx: {idx}")
            data_loader = cls.build_test_loader(
                cfg,
                dataset_name,
            )

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
                
            if ('coco' in dataset_name or 'sam' in dataset_name or 'paco' in dataset_name) and "full" in suffix:
                suffix = suffix.split("_")[0] # remove _full

            suffix += f'_{dataset_name.split("_")[0]}'
            results_val = results[dataset_name].copy()
            results_val = {
                f'{suffix}' + k: v
                for k, v in results_val.items()
            }
            results_structured.update(results_val)
        return results_structured

    def run_fwd_bwd(self):
        """
        Implement the AMP training logic.
        """
        self._trainer.iter = self.iter

        assert (
            self._trainer.model.training
        ), "[AMPTrainer] model was changed to eval mode!"
        assert (
            torch.cuda.is_available()
        ), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        assert self.cfg.SOLVER.AMP.ENABLED

        if self.cfg.SYNC_DATALOADER_TIMING:
            torch.cuda.synchronize()
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        if self.cfg.SYNC_DATALOADER_TIMING:
            torch.cuda.synchronize()

        data_time = time.perf_counter() - start
        if self.cfg.SYNC_DATALOADER_TIMING:
            print(f"Data time: {data_time*1000:.2f}ms")

        with autocast(dtype=self._trainer.precision):
            loss_dict = self._trainer.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                loss_custom = None
                if "loss_3d" in loss_dict or "loss_2d" in loss_dict:
                    loss_name = "loss_3d" if "loss_3d" in loss_dict else "loss_2d"
                    loss_custom = loss_dict[loss_name]
                    loss_dict.pop("loss_3d", None)
                    loss_dict.pop("loss_2d", None)
                losses = sum(loss_dict.values())

                if loss_custom is not None:
                    loss_dict[loss_name] = loss_custom

        if self.cfg.SEPERATE_OPTIMIZERS:
            decoder_3d = data[0]["actual_decoder_3d"]
            if decoder_3d:
                optimizer = self._trainer.optimizer
            else:
                optimizer = self.optimzer_2d
        else:
            optimizer = self._trainer.optimizer
        optimizer.zero_grad()
        self._trainer.grad_scaler.scale(losses).backward()

        self._trainer.after_backward()

        self._trainer._write_metrics(loss_dict, data_time)
        return optimizer

    def run_step(self):
        if self.cfg.DATALOADER_ONLY:
            if self.cfg.SYNC_DATALOADER_TIMING:
                torch.cuda.synchronize()
            start = time.perf_counter()
            data = next(self._trainer._data_loader_iter)
            if self.cfg.SYNC_DATALOADER_TIMING:
                torch.cuda.synchronize()

            data_time = time.perf_counter() - start
            if self.cfg.SYNC_DATALOADER_TIMING:
                print(f"Data time: {data_time*1000:.2f}ms")

            self._trainer._write_metrics({"random_": torch.tensor(1.0)}, data_time)
            return

        if self.cfg.GRAD_ACCUMULATION_STEPS > 1:
            if (self.iter + 1) % self.cfg.GRAD_ACCUMULATION_STEPS == 0:
                optimizer = self.run_fwd_bwd()
                self._trainer.grad_scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)

                self._trainer.grad_scaler.update()
            else:
                if comm.get_world_size() == 1:
                    self.run_fwd_bwd()
                else:
                    with self.model.no_sync():
                        self.run_fwd_bwd()
        else:
            rank = comm.get_rank()
            custom_viz_track = VizTracer(output_file=f"profile_{rank}.json") if self.cfg.USE_VIZTRACER and (self.iter > 10) else contextlib.nullcontext()
            with custom_viz_track:
                optimizer = self.run_fwd_bwd()

            if self.cfg.USE_VIZTRACER and (self.iter > 10):
                print(f"Finished writing viztracer file for rank {rank}")
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                exit()

            self._trainer.grad_scaler.step(optimizer)
            self._trainer.grad_scaler.update()

        if self.cfg.LOG_GRADIENTS:
            wandb.log({"random": 1.0})

        if self.iter % 10 == 0 and self.cfg.LOG_MEMORY:
            metrics_dict = {}
            metrics_dict["max_reserved_mem_gb"] = torch.tensor(torch.cuda.max_memory_reserved() / (1024**3))
            metrics_dict["reserved_mem_gb"] = torch.tensor(torch.cuda.memory_reserved() / (1024**3))
            metrics_dict["max_allocated_mem_gb"] = torch.tensor(torch.cuda.max_memory_allocated() / (1024**3))
            metrics_dict["allocated_mem_gb"] = torch.tensor(torch.cuda.memory_allocated() / (1024**3))
            metrics_dict["global_step"] = self.iter
            if wandb.run is not None:
                wandb.log(metrics_dict)
            elif self.iter % 100 == 0:
                print(f"Logged memory metrics at global step {self.iter}: {metrics_dict}")

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
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="mask2former_video",
    )
    return cfg


def main(args):
    cfg = setup(args)
    
    if cfg.OOM_OBSERVER:
        print(f"Attaching OOM observer to {cfg.OUTPUT_DIR}")
        from torchtnt.utils.oom import attach_oom_observer
        attach_oom_observer(output_dir=str(cfg.OUTPUT_DIR), trace_max_entries=500000)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        # if comm.is_main_process():
        # verify_results(cfg, res)
        if wandb.run is not None:
            wandb.finish()
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    print(f"Found start method: {mp.get_start_method()}")
    if cfg.USE_FORKSERVER_START_METHOD:
        mp.set_start_method("forkserver", force=True)
        print(f"Set start method: {mp.get_start_method()}")

    print(f"World size: {comm.get_world_size()}")
    
    if cfg.BREAKPOINT_ON_ERROR:
        try:
            from univlg.utils.decoupled_utils import set_global_breakpoint
            set_global_breakpoint()
            return trainer.train()
        except Exception as e:
            import traceback
            print(f"Exception: {e}")
            print(traceback.format_exc())
            breakpoint(traceback=e.__traceback__)
            raise e
        finally:
            pass
    else:
        return trainer.train()

    

DEFAULT_TIMEOUT = timedelta(minutes=30)

def slurm_launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    port=None,
    backend="nccl",
    cfg=(),
    timeout=DEFAULT_TIMEOUT,
    one_process_per_gpu=True,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    This function checks slurm variables and sets environment variables for torch.distributed.
    It expects to be called with the following
        #SBATCH --gpus-per-node={ngpu}
        #SBATCH --nodes={args.nodes}
        #SBATCH --ntasks-per-node={ngpu}

    Note:
        It expects that each task on a node sees all the GPUs.
        [Sasha]: My experience with hydra on the fair cluster is that it instead prefers we set
            ntasks = n_gpus and n_gpus_per_task = 1
        If use hydra for launching, then we might have to adjust this launch function, or create a new one

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    print(f"Launcher got args: {num_gpus_per_machine=}, {num_machines=}, {machine_rank=}, {dist_url=}, {port=}, {backend=}, {cfg=}, {timeout=}, {one_process_per_gpu=}")
    logger = logging.getLogger(__name__)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
        
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        """Initialize slurm distributed training environment.

        If argument ``port`` is not specified, then the master port will be system
        environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
        environment variable, then a default port ``29500`` will be used.

        Args:
            backend (str): Backend of torch.distributed.
            port (int, optional): Master port. Defaults to None.
        """
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank_env = os.environ.get("SLURM_LOCALID", None)

        if one_process_per_gpu:
            if local_rank_env is not None:
                local_rank = int(local_rank_env)
                print(f"Using local rank env: {local_rank}")
            else:
                assert False, "This is wrong"
                num_gpus = torch.cuda.device_count()
                local_rank = proc_id % num_gpus
                # This seems wrong...and not necessary
                print(f"Using local rank: {local_rank} given proc_id: {proc_id} and num_gpus: {num_gpus} visible.")
            torch.cuda.set_device(local_rank)
            machine_rank = proc_id // num_gpus
        else:
            machine_rank = int(os.environ["SLURM_NODEID"])

        print(f"{local_rank_env=}, {local_rank=}, {proc_id=}, {ntasks=}, {node_list=}, {num_gpus=}, {machine_rank=}, {world_size=}, {num_gpus_per_machine=}")

        # Hydra on the fair cluster instead prefers we set ntasks = n_gpus and n_gpus_per_task = 1
        # if use hydra for launching, then we might have to adjust the device settings here

        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # 29500 is torch.distributed default port
            os.environ["MASTER_PORT"] = "29500"
            # if dist_url.startswith("tcp://"):
            #     port = dist_url.split(":")[-1]
            #     print("dist_url: ", dist_url, " port: ", port)
            # os.environ["MASTER_PORT"] = port

        # use MASTER_ADDR in the environment variable if it already exists
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        if one_process_per_gpu:
            os.environ["WORLD_SIZE"] = str(ntasks)
        else:
            os.environ["WORLD_SIZE"] = str(ntasks * num_gpus_per_machine)

        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(proc_id)

        print(f"torch.distributed {world_size=}: {machine_rank=} {local_rank=}")
        print(f'{os.environ["SLURM_NODEID"]=}')
        print(f'{os.environ["SLURM_PROCID"]=}')
        print(f'{os.environ["SLURM_NTASKS"]=}')
        print(f'{os.environ["SLURM_NODELIST"]=}')
        print(f'{os.environ["MASTER_ADDR"]=}')
        print(f'{os.environ["MASTER_PORT"]=}')
        print(f'{os.environ["WORLD_SIZE"]=}')
        print(f'{os.environ["LOCAL_RANK"]=}')
        print(f'{os.environ["RANK"]=}')

        dist.init_process_group(backend=backend, timeout=timeout)
        comm.synchronize()

        assert comm._LOCAL_PROCESS_GROUP is None
        num_machines = world_size // num_gpus_per_machine
        print(f"num_machines: {num_machines}, world_size: {world_size}, num_gpus_per_machine: {num_gpus_per_machine}")
        for i in range(num_machines):
            ranks_on_i = list(
                range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
            )
            print(f"ranks_on_i: {ranks_on_i}")
            pg = dist.new_group(ranks_on_i)
            if i == machine_rank:
                comm._LOCAL_PROCESS_GROUP = pg

        main_func(*cfg)
    else:
        main_func(*cfg)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    args = default_argument_parser().parse_args()
    print(f"Opts: {args.opts}")
    print("Command Line Args:", args)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.backends.cudnn.deterministic = True  #needed
    # torch.use_deterministic_algorithms(True, warn_only=True)

    _launcher = "main"
    _kwargs = dict()
    
    if "launcher=" in args.opts[0]:
        _launcher = args.opts[0].split("=")[1]
        args.opts = args.opts[1:]

    if _launcher == "slurm":
        print(f"Using SLURM launcher on host: {socket.gethostname()}")
        launcher = slurm_launch
        _kwargs["cfg"] = (args,)

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

    else:
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
