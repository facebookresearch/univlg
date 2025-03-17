# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

from .build import *  # noqa: F401, F403
from .coco_evaluation import COCOEvaluatorMemoryEfficient  # noqa: F401
from .dataset_mapper_language import (  # noqa: F401
    ScanqaDatasetMapper,
    Sqa3dDatasetMapper,
    Sr3dDatasetMapper,
)
from .dataset_mapper_scannet import ScannetDatasetMapper  # noqa: F401
from .datasets import *  # noqa: F401, F403
from .scannet_3d_eval import Scannet3DEvaluator  # noqa: F401
from .referrential_grounding_evaluator import ReferrentialGroundingEvaluator  # noqa: F401
from .scannet_3d_eval_semantic import ScannetSemantic3DEvaluator  # noqa: F401
from .refcoco_eval import RefCOCOEvaluator  # noqa: F401
from .vqa_evaluator import VQAEvaluator
