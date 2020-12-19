#!/bin/bash
for i in `seq 0 9`
    do
        python compress_classifier.py -a resnet56_cifar ../../../data.cifar10 \
        --resume ~/open_lth_data/lottery_cifar_resnet56_old/replicate_1/level_$i/main/model_best.pth \
        --evaluate --lth --pruned -mp ~/open_lth_data/lottery_cifar_resnet56_old/replicate_1/level_$i/main/mask.pth --qe
done