#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# add root folder to python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# compile custom operators
cd libs/pointops2
rm -rf build
python setup.py install
cd -
cd mask2former_video/modeling/pixel_decoder/ops
rm -rf build
sh make.sh
cd -
