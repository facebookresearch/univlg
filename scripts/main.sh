#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#SBATCH --nodes=2
#SBATCH --open-mode=append
#SBATCH --time=5-0:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7G
#SBATCH --ntasks-per-node=8
#SBATCH -o output/logs/%x_%j_%n.out
#SBATCH -e output/logs/%x_%j_%n.out
#SBATCH --requeue

printenv

get_cuda_device_count() {
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    else
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
    fi
}

if [ "${DEBUGRUN:-0}" -eq 1 ]; then
    export IGNORERUN=1
    export CUDA_VISIBLE_DEVICES=0
    export NUM_DATALOADERS=${NUM_DATALOADERS:-0}
    export NUM_VAL_DATALOADERS=${NUM_VAL_DATALOADERS:-0}
    export BS=${BS:-2}
    export EVAL_PERIOD=${EVAL_PERIOD:-5}
    export NUM_GPUS=1
fi

if [ "${DEBUGFAST:-0}" -eq 1 ]; then
    export IGNORERUN=1
    export BS=${BS:-4}
    export EVAL_PERIOD=${EVAL_PERIOD:-30}
fi

DIR="$(dirname "$PWD")"
export PYTHONPATH=$DIR:$DIR/pretrain
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

NUM_GPUS=${NUM_GPUS:-$(get_cuda_device_count)}
BS=${BS:-7}
SAMPLING_FRAME_NUM=${SAMPLING_FRAME_NUM:-15}
SIDE_FRAMES=$(( (SAMPLING_FRAME_NUM - 1) / 2 ))
CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-8000}
EVAL_PERIOD=${EVAL_PERIOD:-8000}
IGNORERUN=${IGNORERUN:-0}
NAME=${NAME:-"univlg"}
NUM_DATALOADERS=${NUM_DATALOADERS:-16}
NUM_VAL_DATALOADERS=${NUM_VAL_DATALOADERS:-4}
NUM_MACHINES=${NUM_MACHINES:-1}
BREAKPOINT_ON_ERROR=${BREAKPOINT_ON_ERROR:-False}
USE_STANDALONE=${USE_STANDALONE:-0}

USE_SLURM=${USE_SLURM:-0}
EVAL_ONLY=${EVAL_ONLY:-0}
RESUME=${RESUME:-1}
USE_SWIN=${USE_SWIN:-0}
USE_DINO=${USE_DINO:-1}

echo "BS: ${BS}, NUM_GPUS: ${NUM_GPUS}, SAMPLING_FRAME_NUM: ${SAMPLING_FRAME_NUM}, NUM_DATALOADERS: ${NUM_DATALOADERS}, VAL: ${NUM_VAL_DATALOADERS}"

OUTPUT_DIR_PREFIX="outputs/train"

if [[ "$NAME" = "univlg" && "$EVAL_ONLY" -eq 1 ]]; then
    NAME="univlg_eval"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -z "$OUTPUT_DIR" ]]; then
    if [[ "$IGNORERUN" -eq 1 ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/debug/ignore_${TIMESTAMP}_${NAME}"
        BREAKPOINT_ON_ERROR=True
    elif [[ "$NAME" = "univlg" || "$NAME" = "univlg_eval" ]]; then
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${TIMESTAMP}_${NAME}"
    else
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${NAME}"
    fi
    echo "Using output dir: ${OUTPUT_DIR}"
else
    echo "Using existing output dir: ${OUTPUT_DIR}"
fi

if [[ "$USE_SLURM" -eq 1 ]]; then
    SLURM_ARG="launcher=slurm"
    CMD="srun python"

    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr
    echo "MASTER_ADDR="$MASTER_ADDR
    export MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    echo "MASTER_PORT="$MASTER_PORT

    export RANK=${SLURM_PROCID}
    echo "NODES="$NODES
    echo "RANK="$RANK
    echo "GPUS_PER_NODE="$GPUS_PER_NODE
    echo "SLURM_NODEID="$SLURM_NODEID
else
    SLURM_ARG=""
    CMD="python"
fi

if [[ "$USE_SWIN" -eq 1 ]]; then
    CONFIG_FILE="univlg/configs/swin_3d.yaml"
elif [[ "$USE_DINO" -eq 1 ]]; then
    echo "Using DINOv2"
    CONFIG_FILE="univlg/configs/dinov2_3d.yaml"
else
    CONFIG_FILE="univlg/configs/3d.yaml"
fi

echo "EVAL_ONLY: $EVAL_ONLY RESUME: $RESUME"

if [[ "$EVAL_ONLY" -eq 1 ]]; then
    EVAL_ARG="--eval-only"
else
    EVAL_ARG=""
fi

if [[ "$RESUME" -eq 1 ]]; then
    RESUME_ARG="--resume"
else
    RESUME_ARG=""
fi

if [[ "$USE_STANDALONE" -eq 1 ]]; then
    PYTHON_FILE="standalone_eval.py"
else
    PYTHON_FILE="train.py"
fi

# Checks if CUDA is available on the node and if not, requeues job (if it detects we are inside a SLURM job).
python mask2former_video/slurm_requeue.py

$CMD $PYTHON_FILE --dist-url="tcp://127.0.0.1:$RANDOM" --num-gpus $NUM_GPUS --num-machines $NUM_MACHINES --config-file $CONFIG_FILE $EVAL_ARG $RESUME_ARG $SLURM_ARG \
OUTPUT_DIR $OUTPUT_DIR SOLVER.IMS_PER_BATCH $((NUM_GPUS * NUM_MACHINES * BS)) \
SOLVER.CHECKPOINT_PERIOD $CHECKPOINT_PERIOD TEST.EVAL_PERIOD $EVAL_PERIOD \
INPUT.FRAME_LEFT $SIDE_FRAMES INPUT.FRAME_RIGHT $SIDE_FRAMES INPUT.SAMPLING_FRAME_NUM $SAMPLING_FRAME_NUM \
INPUT.FRAME_LEFT_2D 0 INPUT.FRAME_RIGHT_2D 0 INPUT.SAMPLING_FRAME_NUM_2D 1 \
MODEL.WEIGHTS $CKPTS_PATH/misc/m2f_coco_swin.pth \
SOLVER.BASE_LR 1e-4 \
MODEL.CROSS_VIEW_CONTEXTUALIZE True \
INPUT.CAMERA_DROP False \
INPUT.STRONG_AUGS True \
INPUT.COLOR_AUG False \
MODEL.KNN 8 \
INPUT.AUGMENT_3D True \
INPUT.SAMPLE_CHUNK_AUG True \
MODEL.MASK_FORMER.TRAIN_NUM_POINTS 50000 \
INPUT.VOXELIZE True \
MODEL.CROSS_VIEW_BACKBONE True \
MODEL.CROSS_VIEW_NUM_LAYERS "[2,2,6,2]" \
DO_TRILINEAR_INTERPOLATION True \
INTERP_NEIGHBORS 8 \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 20 \
MODEL.MASK_FORMER.TEST.SEMANTIC_ON True \
SKIP_CLASSES None \
MODEL.FREEZE_BACKBONE False \
SOLVER.TEST_IMS_PER_BATCH $((NUM_GPUS * NUM_MACHINES)) \
SAMPLING_STRATEGY_REF True \
INPUT.CHUNK_AUG_MAX 5 \
SOLVER.MAX_ITER 200000 \
DATALOADER.NUM_WORKERS $NUM_DATALOADERS \
DATALOADER.TEST_NUM_WORKERS $NUM_VAL_DATALOADERS \
INPUT.INPAINT_DEPTH True \
IGNORE_DEPTH_MAX 15.0 \
MODEL.SUPERVISE_SPARSE False \
TEST.EVAL_SPARSE False \
USE_WANDB True \
USE_GHOST_POINTS True \
SCANNET_DATA_DIR $SCANNET_DATA_DIR \
SCANNET200_DATA_DIR $SCANNET200_DATA_DIR \
MATTERPORT_DATA_DIR $MATTERPORT_DATA_DIR \
S3DIS_DATA_DIR $S3DIS_DATA_DIR \
USE_SEGMENTS True \
MODEL.MASK_FORMER.DICE_WEIGHT 6.0 \
MODEL.MASK_FORMER.MASK_WEIGHT 15.0 \
MODEL.OPEN_VOCAB True \
MATCHING_CLASS_WEIGHT 2.0 \
MATCHING_MASK_WEIGHT 0.2 \
MATCHING_DICE_WEIGHT 0.2 \
OPEN_VOCAB_SIGMOID True \
VIS_LANG_ATTN True \
FIND_UNUSED_PARAMETERS True \
USE_MASK_FEATURES_FOR_ATTN True \
SAMPLING_FRACTION_RELEVANT_FRAMES 0.5 \
ADD_DISTRACTOR_RELEVANT_FRAMES False \
ADD_RELEVANT_OBJECTS True \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single','matterport_train_single','scannet200_context_instance_train_200cls_single_highres_100k',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched','scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched','matterport_train_single','scannet200_context_instance_train_200cls_single_highres_100k',)" \
USE_WANDB_NAME_AS_ID False \
SAMPLING_REQUIRE_N_TARGET_FRAMES 1 \
SLURM_JOB_ID "\"${SLURM_JOB_ID:-}\"" \
SLURM_SUBMIT_DIR "\"${SLURM_SUBMIT_DIR:-}\"" \
SOLVER.IMS_PER_BATCH_2D $((NUM_GPUS * NUM_MACHINES * BS2D)) \
SOLVER.IMS_PER_BATCH_3D $((NUM_GPUS * NUM_MACHINES * BS3D)) \
USE_SCAN_ALIGN_MATRIX True \
USE_NEW_EVAL_METRICS_FORMAT True \
LOAD_SCANENTS True \
TEXT_ENCODER_TYPE jina \
USE_CLIP_RELEVANT_FRAMES True \
ADD_RELEVANT_OBJECTS False \
BYPASS_TARGET_ANCHOR_CHECK True \
USE_CLIP_RELEVANT_FRAMES_CLIP_ONLY True \
FORCE_USE_REFERENTIAL_AUGS True \
HIGH_RES_SUBSAMPLE True \
HIGH_RES_INPUT True \
MODEL.MAX_SEQ_LEN 520 \
VIL3D True \
MAX_FRAME_NUM 450 \
USE_BOX_LOSS True \
MODEL.MASK_FORMER.BOX_WEIGHT 3.0 \
MODEL.MASK_FORMER.GIOU_WEIGHT 3.0 \
MATCHING_BOX_WEIGHT 0.1 \
MATCHING_GIOU_WEIGHT 0.1 \
BREAKPOINT_ON_ERROR $BREAKPOINT_ON_ERROR \
SOLVER.LR_SCHEDULER_NAME WarmupCosineLR SOLVER.WARMUP_ITERS 1000 FORCE_DECODER_3D False USE_BOX_LOSS True \
MULTIVIEW_XYZ_SCALES '[14,14,14,14]' DINO_VERSION "dinov2_vitl14_reg" FORCE_VIT_XYZ_SCALE True \
INPUT.IMAGE_SIZE 448 INPUT.MIN_SIZE_TEST 448 INPUT.MAX_SIZE_TEST 448 \
INPUT.SIZE_DIVISIBILITY 14 MODEL.MASK_FORMER.SIZE_DIVISIBILITY 14 \
MODEL.PIXEL_MEAN '[0.0,0.0,0.0]' MODEL.PIXEL_STD '[255.0,255.0,255.0]' \
USE_GENERIC_DINO True \
USE_AUTO_NOUN_DETECTION True \
INPUT.RANDOM_FLIP "none" \
TRAIN_3D True \
MODEL.DECODER_3D True \
TEST.EVAL_3D True $@

if [[ $? == 124 ]]; then 
  scontrol requeue $SLURM_JOB_ID
fi