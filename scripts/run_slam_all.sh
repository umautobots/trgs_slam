#!/bin/bash

output_folder=$1
trnerf_dataset_folder=$2

if [ $# -ne 2 ]
then
    echo "Two arguments are required, exiting"
    exit 1
fi

for speed in slow medium fast
do
    for scene in indoor outdoor
    do
        $(dirname "$0")/run_slam.sh \
            $output_folder \
            $trnerf_dataset_folder \
            $speed \
            $scene
    done
done
