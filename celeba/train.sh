#!/bin/bash

# Change the following:
# - dataset_dir, note that torch FolderLoader is used, so with default path the images should be in folder /datasets/celeba/0
# - steps_per_chpt, set to (number of images // 64) to save every epoch
# - chpt_dir and sum_dir, where to write results

# After that, remove the following lines:
echo "Tips: inspect and edit any script you download before running it."
echo "Edit some important paths in this script before training, please."
exit 0

python3 train_cleaned.py --dataset_dir /datasets/celeba --steps_per_chpt 3009 --chpt_dir /output/dde-celeba/chpts --sum_dir /output/dde-celeba/summaries --g_act swish --dde_act swish --linear --normalize --normalize_data --sigma_real 0.6 --sigma_fake 0.6