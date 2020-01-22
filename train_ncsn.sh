#!/bin/bash
python3 dde.py --train --dataset ncsn --activation lrelu:0.2 --batch_size 512 --epochs 50 --dataset_dir /data/torch --lr 0.0001 --vis_path output-ncsn/