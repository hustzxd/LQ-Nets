#!/usr/bin/env bash
time CUDA_VISIBLE_DEVICES=1 \
python ../train_cifar10.py \
--batch-size 100 \
--model vgg_small_lq --lr 0.02 \
--lr-decay-epoch 80,160,300 \
--num-epochs 400 \
--num-gpus 1 \
--wd 0.0005 \
-qw 2 -qa -1 \
--save-dir vgg_small_cifar10_w2a32_params