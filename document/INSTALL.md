# Installation

### Create Environments

```bash
conda create -n atlas python=3.9
source activate atlas
cd AbdomenAtlas/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### [Optional] If You are using ASU GPU Cluster

Please first read the document at [Essence for Linux newbies](https://github.com/MrGiovanni/Eureka/blob/master/Essence%20for%20Linux%20newbies.md#access-asu-gpu-cluster)

```bash
module load anaconda3/5.3.0 # only for Agave

module load mamba/latest # only for Sol
mamba create -n atlas python=3.9
```

### Download Pretrained Weights

```bash
cd pretrained_weights/
wget https://www.dropbox.com/s/po2zvqylwr0fuek/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
wget https://www.dropbox.com/s/lh5kuyjxwjsxjpl/Genesis_Chest_CT.pt
cd ../
cd pretrained_checkpoints/
wget https://www.dropbox.com/s/jdsodw2vemsy8sz/swinunetr.pth
wget https://www.dropbox.com/s/lyunaue0wwhmv5w/unet.pth
cd ..
```

### Define Variables

```bash
dataname=01_Multi-Atlas_Labeling # an example
datapath=/medical_backup/PublicAbdominalData/
savepath=/medical_backup/Users/zzhou82/outs
```

### Syncronize

rsync -avz --exclude 'out/' --exclude '.git' AbdomenAtlas/ kaisar@10.101.242.2:/home/kaisar/AbdomenAtlas

### Train

python -W ignore train_curvas.py --device cuda --resume pretrained_checkpoints/unet.pth --dataset_list $dataname --data_root_path $datapath --backbone unet --phase train

python -W ignore train_curvas.py --device cuda --resume pretrained_checkpoints/unet.pth --dataset_list CURVAS_reoriented_annotator_1_train_fold1 --data_root_path /experiments/AbdomenAtlas_results/CURVAS_reoriented --backbone unet --phase train --data_txt_path /home/kaisar/Research/Coding/TransferLearning/Bias/CURVAS/AbdomenAtlas/dataset/dataset_list/CURVAS_reoriented_annotator_1_train_fold1.txt

### Test

export dataname=CURVAS_reoriented_annotator_1_test_fold1
export datapath=/experiments/AbdomenAtlas_results/CURVAS_reoriented
export savepath=/experiments/AbdomenAtlas_results/results_CURVAS_reoriented_annotator_1_fold1
python -W ignore test_curvas.py --resume out/curvas_annotator_1/epoch_490.pth --backbone unet --save_dir $savepath --dataset_list $dataname --data_root_path $datapath --cpu --store_result >> logs/$dataname.unet.txt
