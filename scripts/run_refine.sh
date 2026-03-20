#!/bin/bash

output_folder=$1
load_folder=$2
speed=$3
scene=$4

if [ $# -ne 4 ]
then
    echo "Four arguments are required, exiting"
    exit 1
fi

sequence=$"$speed"_$"$scene"
script_folder=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
python3 $script_folder/../src/trgs_slam/run_refine.py \
    --slam-config.load-dir $load_folder/$sequence/ \
    --slam-config.result-dir $output_folder/$sequence/ \
    --slam-config.trajectory-config.device cuda:0 \
    --slam-config.trajectory-config.use-cpp False \
    --slam-config.trajectory-config.bias-lr 1e-4 \
    --slam-config.trajectory-config.linear-accel-weight 1e-2 \
    --slam-config.trajectory-config.bias-accel-weight 0.0 \
    --slam-config.trajectory-config.bias-gyro-weight 0.0 \
    --slam-config.renderer-config.num-rasters 15 \
    --slam-config.renderer-config.integration-interval 40e-3 \
    trnerf-dataset \
        --slam-config.dataset-config.downsample-factor 1
