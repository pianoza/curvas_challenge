#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

device='cuda'
batch_size=1
warmup_epoch=100
max_epoch=1000
store_num=100
word_embedding='/home/kaisar/AbdomenAtlas/pretrained_weights/txt_encoding.pth'
out_dir='/home/kaisar/AbdomenAtlas/out'

savepath='/home/kaisar/CURVAS_results_annotator_3'

annotator=3
for fold in 1 2 3 4 5; do

    echo "------------------------Fold ${fold}-------------------------"

    log_name_train="annotator_${annotator}_train_fold_${fold}"
    log_name_test="annotator_${annotator}_val_fold_${fold}"
    resume='/home/kaisar/AbdomenAtlas/pretrained_checkpoints/unet.pth'
    dataset_list_train="annotator_${annotator}_train_fold_${fold}"
    dataset_list_test="annotator_${annotator}_val_fold_${fold}"
    data_root_path='/home/kaisar/Datasets/CURVAS/training_set_annotator_2'
    backbone='unet'
    data_txt_path='/home/kaisar/AbdomenAtlas/dataset/dataset_list'

    echo "------------------------Training-------------------------"

    docker run -it --shm-size=100GB --gpus '"device=4"' -v /home/kaisar:/home/kaisar atlas:latest python -W ignore /home/kaisar/AbdomenAtlas/train_curvas.py \
                    --device="$device" \
                    --batch_size="$batch_size" \
                    --warmup_epoch="$warmup_epoch" \
                    --max_epoch="$max_epoch" \
                    --log_name="$log_name_train" \
                    --resume="$resume" \
                    --dataset_list="$dataset_list_train" \
                    --data_root_path="$data_root_path" \
                    --backbone="$backbone" \
                    --phase="train" \
                    --data_txt_path="$data_txt_path" \
                    --store_num="$store_num" \
                    --word_embedding="$word_embedding" \
                    --out_dir="$out_dir"

    max_epoch_test=$((max_epoch - 1))

    echo "------------------------Testing-------------------------"

    docker run -it --shm-size=100GB --gpus '"device=4"' -v /home/kaisar:/home/kaisar atlas:latest python -W ignore /home/kaisar/AbdomenAtlas/test_curvas.py \
                    --resume="/home/kaisar/AbdomenAtlas/out/${log_name_train}/epoch_${max_epoch_test}.pth" \
                    --backbone="$backbone" \
                    --save_dir="$savepath" \
                    --dataset_list="$dataset_list_test" \
                    --data_root_path="$data_root_path" \
                    --batch_size="$batch_size" \
                    --phase="test" \
                    --data_txt_path="$data_txt_path" \
                    --cpu \
                    --original_label \
                    --store_soft_pred \
                    --store_result >> /home/kaisar/AbdomenAtlas/logs/${log_name_test}.log
done || exit 1