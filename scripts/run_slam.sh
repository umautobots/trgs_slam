#!/bin/bash

output_folder=$1
trnerf_dataset_folder=$2
speed=$3
scene=$4

if [ $# -ne 4 ]
then
    echo "Four arguments are required, exiting"
    exit 1
fi

sequence=$"$speed"_$"$scene"
if [ ${sequence} = slow_outdoor ]
then
    timestamp_begin=1505.29
    timestamp_end=1735.29
    threshold_min=22530.0
    threshold_max=24878.0
elif [ ${sequence} = medium_outdoor ]
then
    timestamp_begin=2003.35
    timestamp_end=2163.35
    threshold_min=22716.0
    threshold_max=25546.0
elif [ ${sequence} = fast_outdoor ]
then
    timestamp_begin=2438.4
    timestamp_end=2538.4
    threshold_min=22844.0
    threshold_max=25644.0
elif [ ${sequence} = slow_indoor ]
then
    timestamp_begin=1237.73
    timestamp_end=1467.73
    threshold_min=22904.0
    threshold_max=23412.0
elif [ ${sequence} = medium_indoor ]
then
    timestamp_begin=1839.44
    timestamp_end=1999.44
    threshold_min=22934.0
    threshold_max=23410.0
elif [ ${sequence} = fast_indoor ]
then
    timestamp_begin=2398.75
    timestamp_end=2498.75
    threshold_min=22956.0
    threshold_max=23486.0
else
    echo Unsupported sequence: $sequence
    exit 1
fi

script_folder=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
trgs_slam_data_folder=$script_folder/../assets/data/trnerf_dataset/
python3 $script_folder/../src/trgs_slam/run_slam.py \
    --result-dir $output_folder/$sequence \
    trnerf-dataset \
        --dataset-config.time-begin $timestamp_begin \
        --dataset-config.time-end $timestamp_end \
        --dataset-config.downsample-factor 4 \
        --dataset-config.path-images $trnerf_dataset_folder/$sequence/adk_right.h5 \
        --dataset-config.path-calibration-results $trnerf_dataset_folder/cam_calibration/camchain.yaml \
        --dataset-config.path-imu $trnerf_dataset_folder/$sequence/imu.h5 \
        --dataset-config.path-imu-cam-calibration $trnerf_dataset_folder/imu_cam_calibration/transformation_imu_to_mono_left.yaml \
        --dataset-config.path-imu-noise-calibration $trnerf_dataset_folder/imu_noise_calibration/imu.yaml \
        --dataset-config.path-ground-truth-poses $trgs_slam_data_folder/$sequence/pseudo_ground_truth_poses_trgs_slam.h5 \
        --dataset-config.threshold-minimum $threshold_min \
        --dataset-config.threshold-maximum $threshold_max
