#!/bin/bash
MVS_TRAINING="/home/works/Self-Supervised-MVS/dtu"
python train.py \
	--gpu_device "0" \
        --dataset "dtu_yao" \
        --batch_size 2 \
        --trainpath $MVS_TRAINING \
        --trainlist "lists/dtu/train.txt" \
        --testlist "lists/dtu/test.txt" \
        --numdepth 192 \
        --logdir "./log-first-run-7-14-25" \
        --refine False \
        --lr 0.001 \
        --epochs 10 \
        --lrepochs "2,4,6,8:2" \
	--w_aug 0.01 \
	--seg_clusters 4 \
	--w_seg 0.01 \
	--summary_freq 500 \
	--val_freq 5000
