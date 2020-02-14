#!/bin/bash

# Change the following:
# - model_name, saved checkpoint
# - store_single, if not set, outputs a grid of images to out_file
# -               otherwise, output batch_size number of images to the path

# After that, remove the following lines:
echo "Tips: inspect and edit any script you download before running it."
echo "Edit some important paths in this script before testing, please."
exit 0

python3 test_cleaned.py --model_name /output/dde-celeba/chpts/run_00/checkpoint_e50-158250 --out_file /output/dde-celeba/dde_results.png --dde_act swish --g_act swish --linear --normalize
python3 test_cleaned.py --model_name /output/dde-celeba/chpts/run_00/checkpoint_e50-158250 --store_single --out_single_file /output/dde-celeba/images/%05d.png --batch_size 2000 --dde_act swish --g_act swish --linear --normalize