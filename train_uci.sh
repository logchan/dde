#!/bin/bash

python3 dde.py --train --dataset_dir /data/uci \
                --dataset uci_hepmass --n_input 21 --min_sigma 0.02 --n_hidden 128 --lr 0.0005 --save_to hepmass.bin --visualize_every 1000 \
                --lr_step 100 --lr_step_gamma 0.5 \
                --sigma 0.1 --reduce_sigma_every 10000 --reduce_sigma_gamma 0.9091 \
                --batch_size 512 --epochs 400

#--dataset uci_hepmass --n_input 21 --min_sigma 0.02 --n_hidden 128 --lr 0.0005 --save_to hepmass.bin --visualize_every 1000
#--dataset uci_miniboon --n_input 43 --min_sigma 0.015 --n_hidden 64 --lr 0.00025 --save_to miniboon.bin --visualize_every 100
#--dataset uci_power --n_input 6 --min_sigma 0.05 --n_hidden 64 --lr 0.00025 --save_to power.bin --visualize_every 1000
#--dataset uci_gas --n_input 8 --min_sigma 0.04 --n_hidden 128 --lr 0.0005 --save_to gas.bin --visualize_every 1000 --n_uci_samples 512000
