#!/bin/bash
python3 dde.py --train --dataset stacked-mnist --dsc_activation lrelu:0.2 --gen_bn True --sigma 0.5 --batch_size 512 --epochs 50 --dataset_dir /data/torch --n_input 100 --lr 0.002 --vis_path output-stacked/
