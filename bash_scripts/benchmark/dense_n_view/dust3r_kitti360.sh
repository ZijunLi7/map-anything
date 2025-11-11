#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

export HYDRA_FULL_ERROR=1

# Define the batch sizes and number of views to loop over
batch_sizes_and_views=(
    "1 5 benchmark_224_kitti360"
)

# Loop through each combination
for combo in "${batch_sizes_and_views[@]}"; do
    # Split the string into batch_size and num_views
    read -r batch_size num_views dataset <<< "$combo"

    echo "Running $dataset with batch_size=$batch_size and num_views=$num_views"

    python3 \
        benchmarking/dense_n_view/benchmark.py \
        machine=aws \
        dataset=$dataset \
        dataset.num_workers=12 \
        dataset.num_views=$num_views \
        batch_size=$batch_size \
        model=dust3r224 \
        amp=0 \
        hydra.run.dir='${root_experiments_dir}/mapanything/benchmarking/dense_'"${num_views}"'_view/dust3r'

    echo "Finished running $dataset with batch_size=$batch_size and num_views=$num_views"
done
