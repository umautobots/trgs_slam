#!/bin/bash

output_folder=$1
load_folder=$2

if [ $# -ne 2 ]
then
    echo "Two arguments are required, exiting"
    exit 1
fi

for speed in slow medium fast
do
    for scene in indoor outdoor
    do
        $(dirname "$0")/run_refine.sh \
            $output_folder \
            $load_folder \
            $speed \
            $scene
    done
done
