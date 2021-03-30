#!/bin/bash
# for i in `seq 0 15`
#     do
#         for j in `seq 3 8`
#             do
#                 python compress_classifier.py -a simplenet_mnist ../../../data.mnist \
#                 --resume ~/open_lth_data/lottery_184ace1d6901ace6854b0a595cbd6b27/replicate_1/level_9/main/model_best.pth \
#                 --evaluate --lth --pruned -mp ~/open_lth_data/lottery_184ace1d6901ace6854b0a595cbd6b27/replicate_1/level_9/main/mask.pth \
#                 --qe --qebw 8 --qeba 8 --qem asym_u
#         done
# done
for i in `seq 0 15`
    do
        python compress_classifier.py -a simplenet_mnist ../../../data.mnist \
                        --resume ~/open_lth_data/lottery_184ace1d6901ace6854b0a595cbd6b27/replicate_1/level_$i/main/model_best.pth \
                        --evaluate --lth --pruned -mp ~/open_lth_data/lottery_184ace1d6901ace6854b0a595cbd6b27/replicate_1/level_$i/main/mask.pth 
                        # --qe --qebw 8 --qeba 8 --qem asym_u
done