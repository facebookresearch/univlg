# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

export DATA_ROOT=/path/to/scannet_sens/groundtruth # path to scannet sens data (change here)
export TARGET=/path/to/SEMSEG_100k/frames_square_highres   # data destination (change here)

reader() {
    filename=$1
    frame_skip=20

    scene=$(basename -- "$filename")
    scene="${scene%.*}" 
    echo "Find sens data: $filename $scene"
    python -u reader.py --filename $filename --output_path $TARGET/$scene --frame_skip $frame_skip 
}

export -f reader

parallel -j 16 --linebuffer time reader ::: `find $DATA_ROOT/scene*/*.sens`
# reader '/scratch/scans/scene0191_00/scene0191_00.sens'
