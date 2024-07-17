import os
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse

def generate_list(args, subjects):
    images, labels = [], []
    for folder in subjects:
        images.append(f'{folder}/image.nii.gz')
        labels.append(f'{folder}/annotation_{args.annotator}.nii.gz')
    
    with open(os.path.join(args.out, args.save_file), 'w') as f:
        for image, label in zip(images, labels):
            f.write(f'{image}\t{label}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_json', default='/experiments/CURVAS_training_set/splits_final.json')
    parser.add_argument('--data_path', default='/experiments/CURVAS_training_set', help='The path of your data')
    # parser.add_argument('--dataset_name', default='', help='The dataset name for generating')
    parser.add_argument('--out', default='/home/kaisar/Research/Coding/TransferLearning/Bias/CURVAS/AbdomenAtlas/dataset/dataset_list')
    # parser.add_argument('--save_file', default='annotator_1_fold1.txt')
    parser.add_argument('--annotator', default='1', help='The annotation number 1, 2, 3')

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    with open(args.splits_json, 'r') as f:
        splits = json.load(f)
    
    # print(len(splits))

    for k, split in enumerate(splits):
        for key, value in split.items():
            print(k, key, value)
            args.save_file = f'annotator_{args.annotator}_{key}_fold_{k+1}.txt'
            generate_list(args, value)

if __name__ == "__main__":
    main()