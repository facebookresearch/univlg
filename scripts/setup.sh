#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


configure_shared() {
    export TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    export OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/${NAME}_${TIMESTAMP}"
}

configure_slurm() {
    local PARTITION="general"
    local QOS="normal"
    local ARRAY_ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --partition=*)
                PARTITION="${1#*=}"
                if [[ "$PARTITION" == "array" ]]; then
                    QOS="array_qos"
                    ARRAY_ARGS="--array=0-0%1"
                elif [[ "$PARTITION" == "preempt" ]]; then
                    QOS="preempt_qos"
                fi
                shift
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done
    
    export NUM_GPUS=${NUM_GPUS:-8}
    export NUM_MACHINES=${NUM_MACHINES:-1}
    export NAME=${NAME:-"univlg"}
    export COMMENT=${COMMENT:-"univlg"}
    export EVAL_PERIOD=${EVAL_PERIOD:-5000}
    export CHECKPOINT_PERIOD=${CHECKPOINT_PERIOD:-10000}
    export NUM_GPUS=$NUM_GPUS
    export NUM_MACHINES=$NUM_MACHINES
    export USE_SLURM=1
    export PREFIX="sbatch"
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_SHM_DISABLE=1
    export NCCL_DEBUG=INFO
    
    
    PREFIX_ARGS=(
        "--qos=$QOS"
        "--job-name=$NAME"
        "--partition=$PARTITION"
        $([[ -n $ARRAY_ARGS ]] && echo "$ARRAY_ARGS")  # Removed extra quotes
        $(hostname | grep -q "babel" && echo "--account=kfragki2")
        "--nodes=$NUM_MACHINES"
        "--gpus-per-node=$NUM_GPUS"
        "--ntasks-per-node=$NUM_GPUS"
        "--time=1-0:00"
        "--comment=$COMMENT"
    )
    # Remove empty elements from array
    PREFIX_ARGS=("${PREFIX_ARGS[@]/#empty/}")
    export PREFIX_ARGS

    echo "Args: $PREFIX" "${PREFIX_ARGS[@]}"
    configure_shared
}

configure_local() {
    unset SLURM; unset NAME; unset EVAL_PERIOD; unset NUM_GPUS; unset NUM_MACHINES; unset PREFIX; unset DEBUGRUN; unset PREFIX_ARGS;
    export DEBUGRUN=0;
    export EVAL_PERIOD=100;
    export NUM_GPUS=1;
    export NUM_MACHINES=1;
    export SLURM=0;
    export USE_SLURM=0;
    export PREFIX="bash";
    export PREFIX_ARGS="";
    export NAME="local"
    export PREFIX_ARGS=()
    configure_shared
}