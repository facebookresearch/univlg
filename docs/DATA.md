# ScanNet

- For setting up the ScanNet dataset (and optionally Matterport3D dataset), follow instrictions from [here](https://github.com/ayushjain1144/odin/blob/main/data_preparation/scannet/README.md).

- Next, Download the ScanEnts3D version of the NR3D and ScanRefer dataset from [here](https://scanents3d.github.io/dataset.html) and unzip `ScanEnts3D_ScanRefer.zip`.

```bash
mkdir data/refer_it_3d
cd data/refer_it_3d
wget https://scanents3d.github.io/ScanEnts3D_Nr3D.csv https://scanents3d.github.io/ScanEnts3D_ScanRefer.zip
unzip ScanEnts3D_ScanRefer.zip
```

- Download the SR3D dataset from [here](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV) provided by Referit3D

- Generate the train-test splits for Referi3D and ScanRefer:
```bash
export REF_DATASET='data/refer_it_3d'
uv run data_preparation/refexp/make_splits.py
```

Note: We train with Matterport3D dataset as well, but you can skip it if you want -- it doesn't help in the 3D grounding performance.

## ScanQA and SQA3D datasets
- Download the ScanQA dataset from [here](https://github.com/ATR-DBI/ScanQA?tab=readme-ov-file) and the SQA3D dataset from [here](https://github.com/SilongYong/SQA3D)

### Scannet Metadata

You can download some pre-computed data for scannet. If you want to generate it from scratch, see below.

For all scripts, set the following environment variables:
```bash
REF_DATASET="ckpts/scannet"
UNIVLG_DATA_PATH="..."
```

### Embeddings

To improve frame subsampling during training, we pre-compute embeddings for all text and images:

```bash
uv run accelerate launch --main_process_port $RANDOM tools/generate_object_pixel_counts.py
uv run accelerate launch --main_process_port $RANDOM tools/generate_scene_image_embeddings.py # This will take approximately 30 minutes
uv run accelerate launch --main_process_port $RANDOM tools/generate_scene_text_embeddings.py

# Less tested: You can use tools/generate_clip_sampling_data.py which computes image+text at the same time
```

### Span Prediction

To pre-compute span predictions:
```bash
uv run accelerate launch --main_process_port $RANDOM tools/generate_predicted_spans.py
```

## 2D Datasets

The above steps are enough if you just want to run inference on 3D language grounding datasets or train the 3D only baseline. However, if you want to train with 2D datasets too, you follow the below steps:

Make a new folder: `mkdir data/datasets_2d; cd data/datasets_2d`

Inside of it, download the following: 
- Download COCO dataset from [here](https://cocodataset.org/#download)
- Download the RefCOCO, RefCOCO+, and RefCOCOg datasets from [here](https://github.com/lichengunc/refer). This [issue](https://github.com/lichengunc/refer/issues/14#issuecomment-1258318183) might be relevant if download links don't work. 
- Download the 3D pointmap data for COCO dataset:

```bash
uvx --from huggingface_hub huggingface-cli download katefgroup/UniVLG_ScanNet_MonoDepth --local-dir data/datasets_2d/coco_3d_moge
```

# Set the path to the setup datasets in the main.sh script. Specifically:
CKPTS_PATH -- path to checkpoints
PRECOMPUTED_SCANNET_PATH -- path to image/text embeddings and other metadata for scannet
DETECTRON2_DATASETS_2D -- path to datasets_2d
DETECTRON2_DATASETS -- path to the folder which contains the RGB-D images from scannet and matterport3D datasets
REF_DATASET -- path to the `refer_it_3d` folder
SCANNET200_DATA_DIR -- path to the `train_validation_database.yaml` of `scannet200` dataset
MATTERPORT_DATA_DIR -- path to the `train_validation_database.yaml` of `matterport3d` dataset
OUTPUT_DIR_PREFIX -- path to the folder which will store your logs and checkpoints