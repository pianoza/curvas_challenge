import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import shutil
import nibabel as nib
import csv

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from model.Universal_model import Universal_model
from dataset.dataloader_test import get_loader
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key, pseudo_label_all_organ, pseudo_label_single_organ, save_organ_label
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS,ORGAN_NAME_LOW
from utils.utils import organ_post_process, threshold_organ,create_entropy_map,save_soft_pred,invert_transform

from torchsummary import summary



model = Universal_model(img_size=(96, 96, 96),
                in_channels=1,
                out_channels=NUM_CLASS,
                backbone="unet",
                encoding='word_embedding'
                ).cuda()

summary(model, (1, 96, 96, 96), depth=3)
output = model(torch.randn(1, 1, 96, 96, 96).cuda())
print(output.shape)

# count the number of parameters in the model
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")
# number of biases in the model
num_bias = 0
for name, param in model.named_parameters():
    if 'bias' in name:
        num_bias += param.numel()
print(f"Number of biases: {num_bias}")

# UNET3D
# Number of parameters: 19416417
# Number of biases: 6817

# SWINUNETR
# Number of parameters: 62595315
# 220867