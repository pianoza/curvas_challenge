import os
from pathlib import Path
import nibabel as nib
import numpy as np
import argparse

def generate_list(args):
    # list all folders inside data_path
    data_path = Path(args.data_path)
    folders = [f.name for f in data_path.iterdir() if f.is_dir()]
    images, labels = [], []
    for folder in folders:
        images.append(f'{folder}/image.nii.gz')
        labels.append(f'{folder}/annotation_{args.annotator}.nii.gz')
    
    with open(os.path.join(args.out, args.save_file), 'w') as f:
        for image, label in zip(images, labels):
            f.write(f'{image}\t{label}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/kaisar/Datasets/CURVAS/training_set', help='The path of your data')
    parser.add_argument('--dataset_name', default='CURVAS_annotator_1', help='The dataset name for generating')
    parser.add_argument('--folder', nargs='+', default=None, help='folder to filter the files(img,train,imagesTr)')
    parser.add_argument('--filetype', default='.nii.gz', help='.nii.gz,.mhd')
    parser.add_argument('--out', default='/home/kaisar/Research/Coding/TransferLearning/Bias/CURVAS/AbdomenAtlas/dataset/dataset_list')
    parser.add_argument('--save_file', default='CURVAS_annotator_1.txt')
    parser.add_argument('--annotator', default='1', help='The annotation number 1, 2, 3')

    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    generate_list(args)

if __name__ == "__main__":
    main()