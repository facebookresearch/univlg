# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    # cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    # cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    # cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = 32

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
    cfg.MODEL.MASK_FORMER.GENERATION_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    # cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    # cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    # cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False
    # cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    # cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    # cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    # cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = (
        "MultiScaleMaskedTransformerDecoder"
    )

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = [
        "res3",
        "res4",
        "res5",
    ]


    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    cfg.MODEL.CROSS_VIEW_BACKBONE = False
    cfg.MODEL.DECODER_2D = True
    cfg.MODEL.DECODER_3D = False
    cfg.MODEL.FREEZE_BACKBONE = False
    cfg.INPUT.FRAME_LEFT = 0
    cfg.INPUT.FRAME_RIGHT = 0
    cfg.INPUT.SAMPLING_FRAME_NUM = 1

    cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE = False
    cfg.MODEL.SUPERVISE_SPARSE = False
    cfg.TEST.EVAL_SPARSE = False
    cfg.MODEL.OPEN_VOCAB = False
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D = cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS


def add_maskformer2_video_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.FRAME_RIGHT = 2
    cfg.INPUT.FRAME_LEFT = 2
    cfg.INPUT.SAMPLING_FRAME_NUM = cfg.INPUT.FRAME_RIGHT + cfg.INPUT.FRAME_LEFT + 1
    cfg.MODEL.DECODER_3D = False
    cfg.TEST.EVAL_3D = False
    cfg.MODEL.CROSS_VIEW_CONTEXTUALIZE = False
    cfg.INPUT.INPAINT_DEPTH = False
    cfg.INPUT.STRONG_AUGS = False
    cfg.INPUT.CAMERA_DROP = False
    cfg.INPUT.CAMERA_DROP_PROB = 0.5
    cfg.INPUT.CAMERA_DROP_MIN_FRAMES_KEEP = 1
    cfg.INPUT.ALWAYS_KEEP_FIRST_FRAME = False
    cfg.INPUT.MIN_SIZE_TEST_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.SUPERVISE_SPARSE = False
    cfg.TEST.EVAL_SPARSE = False
    cfg.MODEL.KNN = 8
    cfg.INPUT.AUGMENT_3D = False
    cfg.MODEL.FREEZE_BACKBONE = False
    cfg.INPUT.SAMPLE_CHUNK_AUG = False
    cfg.INPUT.VOXELIZE = False
    cfg.INPUT.VOXEL_SIZE = [0.02, 0.04, 0.08, 0.16]
    cfg.DATALOADER.TEST_NUM_WORKERS = 4
    cfg.MODEL.CROSS_VIEW_BACKBONE = False
    cfg.INPUT.ORIGINAL_EVAL = False
    cfg.INPUT.UNIFORM_SAMPLE = False
    cfg.SOLVER.TEST_IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH
    cfg.INPUT.CHUNK_AUG_MAX = 5
    cfg.MODEL.CROSS_VIEW_NUM_LAYERS = [2, 2, 6, 2]
    cfg.USE_GHOST_POINTS = (
        False  # featurizes the ghost points and do dot product with them
    )
    cfg.SCANNET_DATA_DIR = "/path/to/mask3d_processed/scannet/train_validation_database.yaml"
    cfg.S3DIS_DATA_DIR = "/path/to/SEMSEG_100k/s3dis/train_validation_database.yaml"
    cfg.INPUT.RENDER_COLOR = False
    cfg.INPUT.RENDER_DEPTH = False
    cfg.SKIP_CLASSES = None
    cfg.VISUALIZE = False
    cfg.FEATURE_VIS = False
    cfg.VISUALIZE_LOG_DIR = "/path/to/language_grounding/visualizations/default"
    cfg.DO_TRILINEAR_INTERPOLATION = True
    cfg.INTERP_NEIGHBORS = 8
    cfg.MODEL.INTERPOLATION_METHOD = "nearest"
    cfg.TEST.EVAL_2D = False
    cfg.DECODER_NUM_LAYERS = 1
    cfg.SAMPLING_STRATEGY = "consecutive"
    cfg.MAX_FRAME_NUM = -1
    cfg.USE_SEGMENTS = False
    cfg.USE_SHORTCUT = False
    cfg.KNN_THRESH = 1e-3
    cfg.MODEL.DECODER_2D = False
    cfg.ADD_TEMPORAL = True
    cfg.INPUT.COLOR_AUG = True
    cfg.MASK_VALID = False
    cfg.IGNORE_DEPTH_MAX = -1.0
    cfg.MULTI_TASK_TRAINING = False
    cfg.DATASETS.TRAIN_3D = []
    cfg.DATASETS.TRAIN_2D = []
    cfg.TRAIN_3D = False
    cfg.TRAIN_2D = False
    cfg.FIND_UNUSED_PARAMETERS = False
    cfg.HIGH_RES = False
    cfg.NO_POINTREND = False
    cfg.HIGH_RES_SUBSAMPLE = False
    cfg.HIGH_RES_INPUT = False
    cfg.DATASETS.TEST_3D_ONLY = []
    cfg.DATASETS.TEST_2D_ONLY = []
    cfg.EVALUATE_SUBSET = None
    cfg.EVAL_PER_IMAGE = False
    cfg.DEPTH_PREFIX = "depth_inpainted"
    cfg.MEAN_CENTER = False
    cfg.TRAIN_SUBSET_3D = None
    cfg.TRAIN_SUBSET_2D = None
    cfg.SOLVER.IMS_PER_BATCH_2D = cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.IMS_PER_BATCH_3D = cfg.SOLVER.IMS_PER_BATCH
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D = cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS
    cfg.INPUT.FRAME_LEFT_2D = cfg.INPUT.FRAME_LEFT
    cfg.INPUT.FRAME_RIGHT_2D = cfg.INPUT.FRAME_RIGHT
    cfg.INPUT.SAMPLING_FRAME_NUM_2D = cfg.INPUT.SAMPLING_FRAME_NUM
    cfg.TEST.SUBSAMPLE_DATA = None
    cfg.TRAIN_SUBSAMPLE_DATA = None
    cfg.DATASETS.TEST_SUBSAMPLED = []
    cfg.DATASETS.TRAIN_SUBSAMPLED = []
    cfg.NOT_USE_WD_PRETRAINED = False
    cfg.MEAN_CENTER = False
    cfg.FORCE_DECODER_3D = False
    cfg.SKIP_CLASSES_2D = cfg.SKIP_CLASSES
    cfg.VISUALIZE_PRED = False
    cfg.INPUT.MIN_SIZE_TEST_2D = cfg.INPUT.MIN_SIZE_TEST
    cfg.INPUT.MAX_SIZE_TEST_2D = cfg.INPUT.MAX_SIZE_TEST
    cfg.INPUT.IMAGE_SIZE_2D = cfg.INPUT.IMAGE_SIZE
    cfg.MATTERPORT_DATA_DIR = "/path/to/mask3d_processed/matterport/train_validation_database.yaml"
    cfg.SCANNET200_DATA_DIR = "/path/to/mask3d_processed/scannet200/train_validation_database.yaml"
    cfg.AUGMENT_WITH_3D_SCALE = False
    cfg.BALANCE_3D_DATASETS = False
    cfg.MODEL.NO_DECODER_PANET = False
    cfg.EXPORT_BENCHMARK_DATA = False
    cfg.MATTERPORT_ALL_CLASSES_TO_21 = False
    cfg.FORCE_SUBSET_EVAL = False
    cfg.USE_CONV1D = False
    cfg.PASS_2D_CROSS_VIEW = False
    cfg.USE_MULTITASK_WEIGHT = None
    cfg.FORCE_DECODER_2D = False
    cfg.IID_MULTITASK_TRAINING = False
    cfg.PSEUDO_2D_AUG = False
    cfg.DO_ROT_SCALE = True
    cfg.USE_2D_LOSS_IMPL = False
    cfg.USE_ESTIMATED_DEPTH_FOR_2D = False
    cfg.USE_WANDB = False
    cfg.NO_POS = False
    cfg.WANDB_NAME = None
    cfg.PROB = None
    cfg.NON_PARAM_RELATIVE = False
    cfg.EVALUATE_SUBSET_ZERO_OUT = False
    cfg.USE_CLASSIFICATION_ONLY_LOSS = False
    cfg.EVAL_TIME_INSPECT_LOSS = False
    cfg.SEPERATE_OPTIMIZERS = False
    cfg.SINGLE_PROJ_MLP = False
    cfg.LOG_GRADIENTS = False
    cfg.CHEATING_SAMPLING = False
    cfg.FREEZE_PREDICTOR = False
    cfg.FREEZE_FEATURE_GEN = False
    cfg.GRAD_ACCUMULATION_STEPS = 1
    cfg.FREEZE_QUERY = False
    cfg.FREEZE_TRANSFORMER = False
    cfg.FREEZE_SELF_ATTN = False
    cfg.FREEZE_CROSS_ATTN = False
    cfg.FREEZE_FFN = False
    cfg.SELF_CONSISTENCY_LOSS = False
    cfg.MODEL.MASK_FORMER.CLASSIFICATION_WEIGHT = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
    cfg.ONLY_SUPERVISE_CONF_SEGS = False
    cfg.HARD_TARGETS = False
    cfg.MODEL.MASK_FORMER.CONSISTENCY_LOSS = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
    cfg.FREEZE_FFN_MLP = False
    cfg.FREEZE_FFN_LN = False
    cfg.FREEZE_MASK_EMBED = False
    cfg.FREEZE_TEXT_PROJ = False
    cfg.LOAD_ALL_GTS = False
    cfg.GT_PSEUDO_SEG = False
    cfg.ONLY_SUP_INVALID = False
    cfg.SEPERATE_2D_3D_QUERIES = False
    cfg.CLS_ONLY = False
    cfg.INDV_CLASS_ENCODE = False
    cfg.ONLY_USE_2D_QUERIES = False
    cfg.ONLY_USE_3D_QUERIES = False
    cfg.USE_ESTIMATED_CAMERA_FOR_2D = False
    cfg.POSE_PREFIX = None
    cfg.INTRINSICS_PREFIX = None
    cfg.USE_VIZTRACER = False
    cfg.DATALOADER.PIN_MEMORY = False
    cfg.DATALOADER.PREFETCH_FACTOR = 2
    cfg.POINT_PROMPT = False
    cfg.MULTI_POINT_PROMPT = None
    cfg.TREAT_FORCE_DECODER_3D_AS_2D = False
    cfg.ADD_DETECTION_PRETEXT = False

    # Open Vocab configs
    cfg.MODEL.OPEN_VOCAB = False
    cfg.MODEL.LANG_FREEZE_BACKBONE = True
    cfg.MODEL.MAX_SEQ_LEN = 256
    cfg.USE_SPECIAL_BIAS = False
    cfg.NON_PARAM_SOFTMAX = False
    cfg.DISABLE_SHUFFLE = True
    cfg.RANDOM_SELECT_CLASSES = False
    cfg.OPEN_VOCAB_SIGMOID = False
    cfg.USE_GLIP_BIAS = False
    cfg.DETIC = False
    cfg.NO_NO_OBJ_SUP = False
    cfg.TEXT_ENCODER_TYPE = "roberta"
    cfg.OPEN_VOCAB_SOFTMAX = False

    # Referrential Grounding Configs
    cfg.SCANNET_ALIGN_MATRIX_PATH = 'univlg/data_video/scans_axis_alignment_matrices.json'
    cfg.USE_SCAN_ALIGN_MATRIX = False
    cfg.SAMPLING_FRACTION_RELEVANT_FRAMES = 0.5
    cfg.NO_GRAD = False
    cfg.SAMPLING_STRATEGY_REF = False
    cfg.MATCHING_CLASS_WEIGHT = None
    cfg.MATCHING_MASK_WEIGHT = None
    cfg.MATCHING_DICE_WEIGHT = None
    cfg.VIS_LANG_ATTN = False
    cfg.VISUALIZE_REF = False
    cfg.FORCE_SUBSAMPLE = False
    cfg.USE_MASK_FEATURES_FOR_ATTN = False
    cfg.LOAD_SCANENTS = False
    cfg.MAX_CLASSES = None
    cfg.USE_SOFT_TOKEN = False
    cfg.DATALOADER_ONLY = False
    cfg.ADD_DISTRACTOR_RELEVANT_FRAMES = False
    cfg.SYNC_DATALOADER_TIMING = False
    cfg.PROFILE_MULTIVIEW_XYZ = False
    cfg.USE_VIZTRACER = False
    cfg.USE_CUDA_DATALOADER = False
    cfg.USE_FORKSERVER_START_METHOD = True
    cfg.ADD_RELEVANT_OBJECTS = False
    cfg.DATASET_MUL = None
    cfg.DATASET_MUL_2D = None
    cfg.USE_WANDB_NAME_AS_ID = False
    cfg.SAMPLING_REQUIRE_N_TARGET_FRAMES = 0
    cfg.ONLY_ATTEND_LANG = False
    cfg.BREAKPOINT = False
    cfg.FORCE_USE_REFERENTIAL_AUGS = False
    cfg.FORCE_USE_DETECTION_AUGS = False
    cfg.BYPASS_TARGET_ANCHOR_CHECK = False
    cfg.SAMPLING_MAX_FRAMES_PER_RELEVANT_ID = None
    cfg.VIZ_EXTRA_REF = False
    cfg.USE_CLIP_RELEVANT_FRAMES = False
    cfg.FORCE_FULL_RANDOM_RELEVANT_FRAMES = False
    cfg.USE_CLIP_RELEVANT_FRAMES_CLIP_ONLY = False
    cfg.USE_GT_MASKS = False
    cfg.LOG_MEMORY = False
    cfg.VIL3D = False
    cfg.OOM_OBSERVER = False
    cfg.USE_SCANNET200_IN_REF = False
    cfg.REMOVE_ROOM = False
    cfg.DONT_REMOVE_ATTN_MASK = False
    cfg.VIS_LANG_ATTN_V2 = False
    cfg.COCO_REF_EVAL_USE_MASK_IOU = False
    cfg.FAKE_VOXELIZE = False
    cfg.NO_VOXELIZE_2D = False
    cfg.DATASETS.TEST_REFEXP_ONLY = []
    cfg.NO_POS_IN_PANETS = False
    cfg.RETURN_ORIGINAL_PROB = None
    cfg.USE_DIFF_DET_REF_MATCHING = False
    cfg.MLP_TEXT_PROJECTOR = False
    cfg.VISUALIZE_MATCHES = False
    cfg.USE_MOGE_DEPTH = False
    cfg.PANET = False
    cfg.FIX_PANET_BUG = False
    cfg.USE_AUTO_NOUN_DETECTION = False
    

    cfg.WANDB_PROJECT = "univlg"
    cfg.DETACH_GENERATION_LOSS = False
    cfg.USE_NEW_EVAL_METRICS_FORMAT = False

    # box loss configs
    cfg.USE_BOX_LOSS = False
    cfg.MODEL.MASK_FORMER.BOX_WEIGHT = 3.0
    cfg.MODEL.MASK_FORMER.GIOU_WEIGHT = 3.0
    cfg.MATCHING_BOX_WEIGHT = None
    cfg.MATCHING_GIOU_WEIGHT = None
    cfg.USE_BOX_LOSS_IN_MATCHING_ONLY = False
    cfg.SLURM_JOB_ID = None
    cfg.SLURM_SUBMIT_DIR = None
    cfg.BREAKPOINT_ON_ERROR = False
    cfg.FORCE_VIT_XYZ_SCALE = False
    cfg.MULTIVIEW_XYZ_SCALES = [4, 8, 16, 32]
    cfg.NO_BOX_COST_IN_MATCHING_FOR_DET = False
    cfg.ADD_DETECTION_PRETEXT = False

    # Box Decoding Head
    cfg.USE_BOX_HEAD = False

    # VQA settings
    cfg.GENERATION = False

    cfg.DINO_GROUP_NORM = False
    cfg.NON_ZERO_DINO_RESIDUAL = False
    cfg.DINO_VERSION = "dinov2_vitl14_reg"
    cfg.FORCE_VIT_XYZ_SCALE = False
    cfg.DINO_UNFREEZE_LAYERS = []
    cfg.FORCE_SINGLE_FPN_LEVEL = False
    cfg.DINO_V2 = False
    cfg.USE_GENERIC_DINO = False
    cfg.VITDET_WINDOW_SIZE = None
    cfg.DINO_GENERIC_DISABLE_INFERENCE_MODE = False
    cfg.DISABLE_VIT_PANET = False

    cfg.ENABLE_POSE_NOISE = False
    cfg.POSE_TRANSLATION_NOISE = 0.005
    cfg.POSE_ROTATION_NOISE = 0.005
    cfg.SAVE_PCD = False
    cfg.ENABLE_DEPTH_NOISE = False
    cfg.DEPTH_NOISE_STD = 0.01
    cfg.DINO_EVAL_BATCH = False
    cfg.DINO_EVAL_BATCH_SIZE = 1
    cfg.TEST_DATASET_INFERENCE = False
    cfg.TEST_RESULT_EXPORT_PATH = None
    cfg.WANDB_ENTITY = None
    cfg.AR_LLM = False
    cfg.AR_EMBED = False
    cfg.AR_INSTRUCT = False
