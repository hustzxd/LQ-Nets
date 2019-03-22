#!/usr/bin/env bash
time CUDA_VISIBLE_DEVICES=0 \
python ../train_cifar10.py \
--batch-size 100 \
--model vgg_small --lr 0.02 \
--lr-decay-epoch 80,160,300 \
--num-epochs 400 \
--num-gpus 1 \
--wd 0.0005 \
--save-dir vgg_small_cifar10_base_params
# 0.9355-cifar-vgg_small-306-best.params
