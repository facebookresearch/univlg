# Training

## 3D Only Training

```bash
source scripts/setup.sh
configure_local # To use SLURM: replace with "configure_slurm --partition=$SLURM_PARTITION_NAME"
BS=2 NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=10 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh
```

To test only one dataset, e.g., sr3d, add run the following command:
```bash
source scripts/setup.sh
configure_local
BS=2 NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=10 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched',)"
```

## 2D-3D Training

```bash
source scripts/setup.sh
configure_local # To use SLURM: replace with "configure_slurm --partition=$SLURM_PARTITION_NAME"
BS=2 BS2D=2 BS3D=2 NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=10 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
INPUT.FRAME_LEFT_2D 0 INPUT.FRAME_RIGHT_2D 0 INPUT.SAMPLING_FRAME_NUM_2D 1 \
INPUT.RANDOM_FLIP "none" \
MULTI_TASK_TRAINING True \
TRAIN_3D True \
TRAIN_2D True \
MODEL.DECODER_2D True \
MODEL.DECODER_3D True \
TEST.EVAL_3D True \
TEST.EVAL_2D True \
DATASETS.TRAIN_2D "('refcoco_train','refcoco+_train','refcocog_train','coco_2017_train',)" \
DATASETS.TEST_2D_ONLY "('coco_2017_val','refcoco_val','refcoco_train_eval','refcoco+_val','refcoco+_train_eval','refcocog_val','refcocog_train_eval',)" \
DATASETS.TRAIN_3D "('sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single','matterport_train_single','scannet200_context_instance_train_200cls_single_highres_100k',)" \
DATASETS.TEST_3D_ONLY "('sr3d_ref_scannet_val_single_batched','sr3d_ref_scannet_train_eval_single_batched','scanrefer_scannet_anchor_val_single_batched','scanrefer_scannet_anchor_train_eval_single_batched','nr3d_ref_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_train_eval_single_batched','matterport_val_single','scannet200_context_instance_val_200cls_single_highres_100k','ScannetPPDataset',)" \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single','scanrefer_scannet_anchor_train_single','nr3d_ref_scannet_anchor_train_single','matterport_train_single','scannet200_context_instance_train_200cls_single_highres_100k','refcoco_train','refcoco+_train','refcocog_train','coco_2017_train',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched','scanrefer_scannet_anchor_val_single_batched','nr3d_ref_scannet_anchor_val_single_batched','coco_2017_val','refcoco_val','refcoco+_val','refcocog_val','refcocog_train_eval',)" \
DATASET_MUL '[1,1,1,10,10,1,1,1,1]' \
USE_MOGE_DEPTH True \
FORCE_DECODER_3D True
```

## Evaluation

To evaluate the 3D only baseline, replace the `$CKPT_PATH` with the path to the 3D only baseline checkpoint.

To evaluate, simply modify the training script with `EVAL_ONLY=1`, and define the checkpoint path `$CKPT_PATH`. You may also want to set `RETURN_SCENE_BATCH_SIZE` to >1 if you have a GPU with >40GB of VRAM.

For example:
```bash
CKPT_PATH="ckpts/univlg.pth"
...
# Then, add the following configs to the end of the command:
USE_AUTO_NOUN_DETECTION True \
VISUALIZE_REF False VISUALIZE_LOG_DIR "$OUTPUT_DIR/viz_ref" \
DINO_EVAL_BATCH True DINO_EVAL_BATCH_SIZE 8
```

- To visualize the results, set `VISUALIZE_REF` to `True`. We use [Pyviz3D](https://github.com/francisengelmann/PyViz3D) and instructions will be printed to the console explaining how to view the results.

**Note:** `BS` is not used for evaluation. Each rank has a "batch size" of 1 scene, and a maximum number of captions controlled by the `RETURN_SCENE_BATCH_SIZE` environment variable. Thus, we process a single visual scene but multiple captions in each forward pass during evaluation. For faster evaluation, try increasing `RETURN_SCENE_BATCH_SIZE`, although given the differences in scene size and number of captions, this can cause OOM later on. The caption batching per scene is done in `univlg/data_video/datasets/load_sr3d.py`.

A minimal example for SR3D-only evaluation is below:

**Note: The `_5` suffix is used to subsample the dataset to 5 scenes for evaluation, only for debugging purposes.**

```bash
export CKPT_PATH="ckpts/univlg.pth"
source scripts/setup.sh
configure_local
NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=10 EVAL_ONLY=1 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
DATASETS.TRAIN "('sr3d_ref_scannet_train_5_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_5_single_batched',)"
```

A more complete example for 2D-3D evaluation is below:
```bash
export CKPT_PATH="ckpts/univlg.pth"
source scripts/setup.sh
configure_local
BS2D=2 BS3D=2 NUM_VAL_DATALOADERS=2 NUM_DATALOADERS=10 EVAL_ONLY=1 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
USE_AUTO_NOUN_DETECTION True \
VISUALIZE_REF True VISUALIZE_LOG_DIR "$OUTPUT_DIR/viz_ref" \
DINO_EVAL_BATCH True DINO_EVAL_BATCH_SIZE 8
```

### ScanRefer Evaluation
# TODO: Make path generic.

To evaluate on ScanRefer test set, add:
```bash
TEST_DATASET_INFERENCE True \
TEST_RESULT_EXPORT_PATH "$OUTPUT_DIR/test_results" \
SCANNET_DATA_DIR "/path/to/mask3d_processed/scannet/test_database.yaml" \
SCANNET200_DATA_DIR "/path/to/mask3d_processed/scannet200/test_database.yaml" \
```

### Generation Evaluation
```bash
export CKPT_PATH="ckpts/ckpt.pth"
source scripts/setup.sh
configure_local
EVAL_ONLY=1 NUM_DATALOADERS=0 NUM_VAL_DATALOADERS=2 EVAL_ONLY=1 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
DATASETS.TRAIN "('scanqa_ref_scannet_train_single',)" \
DATASETS.TEST "('scanqa_ref_scannet_val_single_batched',)" \
DINO_EVAL_BATCH True DINO_EVAL_BATCH_SIZE 64 \
GENERATION True DETACH_GENERATION_LOSS False SOLVER.BASE_LR 3e-4 BREAKPOINT_ON_ERROR True \
AR_LLM True MODEL.MASK_FORMER.GENERATION_WEIGHT 1000.0 AR_EMBED True AR_INSTRUCT True
```

and set the test dataset to:
```
DATASETS.TEST "('scanrefer_scannet_anchor_test_single_batched',)"
```

## Standalone Evaluation

We additionally provide a standalone evaluation script that uses a minimal amount of pre-processing and post-processing code. We have a pre-saved data sample, and demonstrate how to load from raw sensor data.

To use the standalone evaluation script, run:
```bash
export CKPT_PATH="ckpts/univlg.pth"
source scripts/setup.sh
configure_local
USE_STANDALONE=1 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
USE_AUTO_NOUN_DETECTION False \
USE_SEGMENTS False
```

To save a different data sample (assuming you have the proper datasets configured), run:
```bash
export CKPT_PATH="ckpts/univlg.pth"
source scripts/setup.sh
configure_local
SHUFFLE_BATCHED_CAPTIONS=1 RETURN_SCENE_BATCH_SIZE=32 EVAL_ONLY=1 $PREFIX "${PREFIX_ARGS[@]}" scripts/main.sh \
DATASETS.TRAIN "('sr3d_ref_scannet_train_single',)" \
DATASETS.TEST "('sr3d_ref_scannet_val_single_batched',)" \
SAVE_DATA_SAMPLE True
```

## Notes

- The dataloader is CPU bound, so increase `NUM_DATALOADERS` to the number of CPUs (divided by `NUM_GPUS`) on the machine.
- Training requires a lot of CPU memory, so ensure at least 64GB per GPU. If you run out of CPU memory, try reducing `NUM_DATALOADERS` or `NUM_VAL_DATALOADERS`.
- To use SLURM, replace `configure_local` with `configure_slurm --partition=$SLURM_PARTITION_NAME`. Make sure to set the desired number of GPUs and nodes beforehand (e.g., `export NUM_GPUS=8` and `export NUM_MACHINES=2`).