import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.Universal_model import Universal_model
from model.Unet import UNet3D
# from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS,PSEUDO_LABEL_ALL,containing_totemplate,merge_organ
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from torchsummary import summary

torch.multiprocessing.set_sharing_strategy('file_system')

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, lbl, name = batch["image"].to(args.device), batch["label"].float(), batch['name']
        B, C, W, H, D = lbl.shape
        y = torch.zeros(B,NUM_CLASS,W,H,D)
        for b in range(B):
            for src,tgt in enumerate(TEMPLATE['curvas']):
                y[b][src][lbl[b][0]==tgt] = 1
        # curvas ground truth is already all organ with tumours and vessels, etc. so no need to merge
        # y = merge_organ(args,y,containing_totemplate)
        y = y.to(args.device)
        logit_map = model(x)
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

class UNet3DSegmentation(nn.Module):
    def __init__(self, model, num_classes=3):
        super(UNet3DSegmentation, self).__init__()
        self.backbone = model
        # freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.segmentation_head = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        _, x = self.backbone(x)
        x = self.segmentation_head(x)
        return x

def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    model = UNet3D()
    # criterion and optimizer
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        store_dict = model.state_dict()
        model_dict = checkpoint['net']
        for key in model_dict.keys():
            store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
        # keep only the backbone weights in store_dict
        store_dict = {'.'.join(k.split('.')[1:]): v for k, v in store_dict.items() if 'backbone' in k}
        model.load_state_dict(store_dict)
        print('success resume from ', args.resume)

    # add the segmenation head
    model = UNet3DSegmentation(model, num_classes=NUM_CLASS).to(args.device)
    
    summary(model, (1, args.roi_x, args.roi_y, args.roi_z))
    # # freeze the backbone
    # for name, param in model.named_parameters():
    #     if 'backbone' in name:
    #         param.requires_grad = False
    # unfreeze the biases in the backbone
    # number of trainable parameters
    print('number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # number_of_biases = 0
    # for name, param in model.backbone.named_parameters():
    #     if 'bias' in name:
    #         param.requires_grad = True
    #         number_of_biases += param.numel()
    #     else:
    #         param.requires_grad = False
    # print('number of trainable biases in the backbone:', number_of_biases)  # 5568
    # print('number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # with all on there are 19,416,417 trainable parameters
    # without backbone there are 342,817 trainable parameters
    # with backbone biases only there are 348385 trainable parameters
    
    # summary(model, (1, args.roi_x, args.roi_y, args.roi_z))

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join('out' , args.log_name))
        print('Writing Tensorboard logs to ', os.path.join('out' , args.log_name))
        writer.add_graph(model, torch.randn(1, 1, args.roi_x, args.roi_y, args.roi_z).to(args.device))

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if ((args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0) or (args.epoch == args.max_epoch - 1):
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir(os.path.join('out' , args.log_name)):
                os.mkdir(os.path.join('out' , args.log_name))
            torch.save(checkpoint, os.path.join('out' , args.log_name,'epoch_' + str(args.epoch) + '.pth'))
            print('save model success')

        args.epoch += 1

    # dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device", dest='device', type=str, default='cuda:0')
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='curvas', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=2000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=100, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    ### for cross_validation 'cross_validation/PAOT_0' 1 2 3 4
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--internal_organ', default=True , type=bool, help='Ourdata or internal organ')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":
    main()

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --dist True --uniform_sample
