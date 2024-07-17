#!/bin/bash

device='cuda'
batch_size=1
warmup_epoch=100
max_epoch=1000
store_num=100

savepath='/experiments/CURVAS_results'

annotator=1
for fold in 1 2 3 4 5
do

echo "------------------------Fold ${fold}-------------------------"

log_name_train="annotator_${annotator}_train_fold_${fold}"
log_name_test="annotator_${annotator}_val_fold_${fold}"
resume='pretrained_checkpoints/unet.pth'
dataset_list_train="annotator_${annotator}_train_fold_${fold}"
dataset_list_test="annotator_${annotator}_val_fold_${fold}"
data_root_path='/experiments/CURVAS_training_set'
backbone='unet'
data_txt_path='/home/kaisar/Research/Coding/TransferLearning/Bias/CURVAS/AbdomenAtlas/dataset/dataset_list'


python -W ignore train_curvas.py \
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
                --store_num="$store_num"
max_epoch_test=$((max_epoch - 1))
python -W ignore test_curvas.py \
                --resume="out/${log_name_train}/epoch_${max_epoch_test}.pth" \
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
                --store_result >> logs/${log_name_test}.log
done