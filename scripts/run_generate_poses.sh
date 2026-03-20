#!/bin/bash

########################################################################################################################
# Parameters to set
########################################################################################################################

output_folder=$1
trnerf_dataset_folder=$2
trnerf_repo=$3

if [ $# -ne 3 ]
then
    echo "Three arguments are required, exiting"
    exit 1
fi

for scene in indoor outdoor
do
    if [ $scene == indoor ]
    then
        sequence_list=(\
            'fast_indoor' \
            'medium_indoor' \
            'slow_indoor')
        t_start_list=(\
            '2398.0' \
            '1839.0' \
            '1237.0')
        t_stop_list=(\
            '2499.0' \
            '2000.0 ' \
            '1471.5')
    elif [ $scene == outdoor ]
    then
        sequence_list=(\
            'fast_outdoor' \
            'medium_outdoor' \
            'slow_outdoor')
        t_start_list=(\
            '2438.0' \
            '2003.0' \
            '1505.0')
        t_stop_list=(\
            '2539.0' \
            '2164.0' \
            '1740.0')
    else
        echo "Invalid scene"
        exit 1
    fi

    script_folder=$trnerf_repo/dataset_prep/
    n_images_colmap_list=(\
        '2000' \
        '1500' \
        '500')

    # NOTE: also change --ImageReader.camera_params below if needed

    ####################################################################################################################
    # Set folder paths and create new folders
    ####################################################################################################################

    image_folder=$output_folder/pose_estimation/images/$scene/
    colmap_folder=$output_folder/pose_estimation/colmap/$scene/
    hloc_folder=$output_folder/pose_estimation/hloc/$scene/
    calibration_results_file=$trnerf_dataset_folder/cam_calibration/camchain.yaml

    mkdir -p $image_folder
    mkdir -p $colmap_folder

    ####################################################################################################################
    # Create folder of images and image lists for COLMAP and HLOC to process
    ####################################################################################################################

    paths_colmap_lists=""
    paths_hloc_lists=""
    for i in {0..2}
    do
        sequence=${sequence_list[$i]}
        n_images_colmap=${n_images_colmap_list[$i]}
        t_start=${t_start_list[$i]}
        t_stop=${t_stop_list[$i]}

        python3 $script_folder/h5_to_image_folders.py \
            --path-input-folder $trnerf_dataset_folder/$sequence/ \
            --path-output-folder $image_folder/$sequence/ \
            --n-images-colmap $n_images_colmap \
            --t-start $t_start \
            --t-stop $t_stop

        paths_colmap_lists="$paths_colmap_lists $image_folder/$sequence/image_list_mono_left_colmap.txt"
        paths_hloc_lists="$paths_hloc_lists $image_folder/$sequence/image_list_mono_left_hloc.txt $image_folder/$sequence/image_list_mono_right_hloc.txt"
    done

    python3 $script_folder/combine_image_lists.py \
        --paths-file-to-combine $paths_colmap_lists \
        --path-output-file $image_folder/image_list_mono_left_colmap.txt

    python3 $script_folder/combine_image_lists.py \
        --paths-file-to-combine $paths_hloc_lists \
        --path-output-file $image_folder/image_list_hloc.txt

    ####################################################################################################################
    # Run COLMAP and GLOMAP to estimate sparse left monochrome poses
    ####################################################################################################################

    # Feature extraction
    # NOTE: --ImageReader.camera_params is the mono left Kalibr results:
    # fx, fy, cx, cy, k1, k2, p1, p2
    # where the principal points used here should be the ones estimated from Kalibr + 0.5
    colmap feature_extractor \
        --database_path $colmap_folder/database.db \
        --image_path $image_folder \
        --image_list_path $image_folder/image_list_mono_left_colmap.txt \
        --camera_mode 1 \
        --ImageReader.camera_model OPENCV \
        --ImageReader.camera_params "1089.9885334237777, 1085.471966486138, 720.7611271971043, 565.7831924847984, -0.3639052622561226, 0.1317757082193622, -0.000081248788859175, -0.0001429163673316175" \
        --SiftExtraction.max_num_features 2048

    # Perform feature matching
    # Using sequential matching to find matches between neighboring images in time and using (vocab tree based) loop
    # detection to find additional matches efficiently
    # The transitive_matcher could be subsequently used if more matches are needed.
    colmap sequential_matcher \
        --database_path $colmap_folder/database.db \
        --SequentialMatching.loop_detection 1 \
        --SequentialMatching.loop_detection_period 5 \
        --SequentialMatching.loop_detection_num_images 50 \
        --SequentialMatching.loop_detection_num_nearest_neighbors 5

    # Sparse reconstruction
    mkdir -p $colmap_folder/sparse/
    num_images=$(wc -l < $image_folder/image_list_mono_left_colmap.txt | awk '{print $1}')
    max_num_tracks=$((num_images * 1000)) # Recommended here: https://github.com/colmap/glomap/blob/main/docs/getting_started.md
    glomap mapper \
        --database_path $colmap_folder/database.db \
        --image_path $image_folder \
        --output_path $colmap_folder/sparse/ \
        --skip_retriangulation 1 \
        --skip_pruning 1 \
        --TrackEstablishment.max_num_tracks $max_num_tracks \
        --BundleAdjustment.optimize_intrinsics 0 \
        --BundleAdjustment.optimize_principal_point 0 \
        --GlobalPositioning.use_gpu 1 \
        --BundleAdjustment.use_gpu 1

    ####################################################################################################################
    # Run HLOC to estimate remaining left monochrome poses and all right monochrome camera poses
    ####################################################################################################################

    python3 $script_folder/run_hloc.py \
        --path-colmap-model $colmap_folder/sparse/0/ \
        --path-reference-image-list $image_folder/image_list_mono_left_colmap.txt \
        --path-query-image-list $image_folder/image_list_hloc.txt \
        --path-image-parent-folder $image_folder \
        --path-calibration-results $calibration_results_file \
        --path-outputs $hloc_folder \
        --parallel \
        --overwrite \
        --n-pairs 10

    ####################################################################################################################
    # Finalize poses (combine COLMAP and HLOC poses, scale them, and write them out for each individual sequence)
    ####################################################################################################################

    python3 $script_folder/finalize_poses.py \
        --path-calibration-results $calibration_results_file \
        --path-colmap-model $colmap_folder/sparse/0/ \
        --path-hloc-poses $hloc_folder/poses_query.pickle \
        --path-image-folders $image_folder \
        --path-output-folder $output_folder
done
