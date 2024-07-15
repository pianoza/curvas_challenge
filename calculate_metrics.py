import os
import numpy as np
import pandas as pd
import nibabel as nib
import seg_metrics.seg_metrics as sg
from pathlib import Path

# data_path = Path('/home/kaisar/Datasets/CURVAS/training_set')
data_path = Path('/experiments/AbdomenAtlas_results/CURVAS_reoriented')
# results_path = Path('/experiments/AbdomenAtlas_results/results_CURVAS_reoriented_annotator_1')
# results_path = Path('/experiments/AbdomenAtlas_results/results_CURVAS_reoriented_annotator_1_fold1')
results_path = Path('/experiments/AbdomenAtlas_results/results_CURVAS_reoriented_annotator_1_TL_fold1')
backbone = 'unet'
annotator = '1'
label_num_to_name = {
    1: 'pancreas',
    2: 'kidneys',
    3: 'liver'
}
label_name_to_num = {
    'pancreas': 1,
    'kidneys': 2,
    'liver': 3
}

results = {
    'patient': [],
    'pancreas_dice': [],
    'kidneys_dice': [],
    'liver_dice': []
}

def main():
    # get the folder names from data_path
    folder_names = sorted([f for f in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, f))])
    # iterate over the folders
    for folder in folder_names:
        print(f'Processing {folder}')
        gt_nii = nib.load(os.path.join(data_path, folder, f'annotation_{annotator}.nii.gz'))
        gt = gt_nii.get_fdata().astype(np.uint8)
        pred_nii = nib.load(os.path.join(results_path, folder, 'image/backbones/', backbone, 'pseudo_label.nii.gz'))
        pred = pred_nii.get_fdata().astype(np.uint8)
        proc_pred = np.zeros_like(pred)
        for organ in ['pancreas', 'kidneys', 'liver']:
            organ_nii = nib.load(os.path.join(results_path, folder, 'image/backbones/', backbone, 'segmentations', f'{organ}.nii.gz'))
            organ_data = organ_nii.get_fdata()
            proc_pred[organ_data == 1] = label_name_to_num[organ]
        # calculate the metrics
        metrics = sg.write_metrics(
            labels=[1, 2, 3],
            gdth_img=gt,
            pred_img=proc_pred,
            spacing=gt_nii.header.get_zooms()[:3],
            metrics=['dice']
        )
        # append the results
        results['patient'].append(folder)
        results['pancreas_dice'].append(metrics[0]['dice'][0])
        results['kidneys_dice'].append(metrics[0]['dice'][1])
        results['liver_dice'].append(metrics[0]['dice'][2])
    df = pd.DataFrame(results)
    df_save_path = os.path.join(results_path, f'results_{backbone}_annotator_{annotator}.csv')
    df.to_csv(df_save_path, index=False)

if __name__ == '__main__':
    main()

