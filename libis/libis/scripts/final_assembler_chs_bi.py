from libis.LegNetMax import LitLegNetMax
from libis.utils import Lib_Dataset_A2G
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import glob
from Bio import SeqIO
import argparse
import os

MODEL_CLASS = LitLegNetMax
AGG_LENGTHS = [51, 151, 301]
parser = argparse.ArgumentParser()
parser.add_argument('--subm_name', type = str, required= True)
parser.add_argument('--device_id', type = int, required= True)
parser.add_argument('--template_dir', type = str, default='final_test')# directory with templates, script saves submit files here

args = parser.parse_args()
exp_type = 'CHS'
subm_name = args.subm_name
device_id = args.device_id
template_dir = args.template_dir

if not os.path.exists(os.path.join('./',template_dir,subm_name)):
    os.mkdir(os.path.join('./',template_dir,subm_name))

models_dict = dict()
paths = glob.glob(f'./checkpoints/*/*/{subm_name}/*.ckpt')
template = pd.read_csv(os.path.join(template_dir, f'{exp_type}_aaa_template.tsv',sep='\t', index_col=0)

skipped_tfs = []
for tf in template.columns.values:
    runs = list(filter(lambda x: tf in x, paths))
    if len(runs) == 0:
        skipped_tfs.append(tf)
        continue
    models_dict[tf] = [MODEL_CLASS.load_from_checkpoint(j) for j in runs]
print(f'The checkpoints for the following TFs are not presented: {skipped_tfs}')
assert all([len(models) == 5 for tf, models in models_dict.items()]) 

parser = SeqIO.parse(os.path.join(template_dir, f'{exp_type}_participants.fasta', format = 'fasta'))
tags, seqs = [], []
for record in parser:
    tags.append(record.name)
    seqs.append(str(record.seq))
    
test = pd.DataFrame(dict(tags=tags, seq=seqs))
test.set_index('tags', inplace=True)
test = test.loc[template.index.values,:]
lst_dataloaders = []
for ln in AGG_LENGTHS:
    test_dataset =  Lib_Dataset_A2G(data=test.seq.values, target_len=ln)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = 4096,num_workers=8)
    lst_dataloaders.append(test_dataloader)

for tf, models in models_dict.items():
    lst_tf_pred = []
    for model in models:
        lst_agg = []
        for test_dataloader in lst_dataloaders:
            my_trainer = L.Trainer(devices=[device_id], accelerator='gpu', enable_progress_bar=True)
            if regr_mode:
                pred = (torch.cat(my_trainer.predict(model= model ,dataloaders= test_dataloader))).numpy(force= True)
            else:
                pred = (F.sigmoid(torch.cat(my_trainer.predict(model= model ,dataloaders= test_dataloader)))).numpy(force= True)
            lst_agg.append(pred)
        preds = np.stack(lst_agg, axis=0).mean(0)
        lst_tf_pred.append(preds)
    pre_pred = np.stack(lst_tf_pred, axis=0).mean(0)
    if regr_mode:
        min_value, max_value = min(pre_pred), max(pre_pred)
        template[tf] = ((pre_pred - min_value)/(max_value - min_value)).round(5)
    else:
        template[tf] = pre_pred.round(5)



if not os.path.isfile(submit_path):
    template.to_csv(submit_path, sep = '\t')
else:
    old_subm = pd.read_csv(submit_path,delimiter='\t', index_col=0)
    new_subm = old_subm.join(selected_tfs)
    new_subm.to_csv(submit_path, sep = '\t')
