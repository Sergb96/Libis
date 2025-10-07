from libis.LegNetMax import LitLegNetMax
from libis.utils import Lib_Dataset
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Bio import SeqIO
import glob
import argparse
import os

MODEL_CLASS = LitLegNetMax
parser = argparse.ArgumentParser()
parser.add_argument('--subm_name', type = str, required= True)
parser.add_argument('--exp_type', type = str, required= True)
parser.add_argument('--device_id', type = int, default = 0)
parser.add_argument('--template_dir', type = str, default='final_test') # directory with templates, script saves submit files here

args = parser.parse_args()
exp_type = args.exp_type
subm_name = args.subm_name
device_id = args.device_id
template_dir = args.template_dir

if not os.path.exists(f'./{template_dir}/{subm_name}'):
    os.mkdir(f'./{template_dir}/{subm_name}')

paths = glob.glob(f'./checkpoints/*/*/{subm_name}/*.ckpt')
submit_path = os.path.join('./',template_dir,subm_name,f'{subm_name}.tsv')

models_dict = dict()
A2G_lst = [path.split('/')[-1] for path in glob.glob('./checkpoints/A2G/*')]
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

parser = SeqIO.parse(f'./{template_dir}/{exp_type}_participants.fasta', format = 'fasta')
tags, seqs = [], []
for record in parser:
    tags.append(record.name)
    seqs.append(str(record.seq))
test = pd.DataFrame(dict(tags=tags, seq=seqs))

test_dataset = Lib_Dataset(data=test.seq.values)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = 4096,num_workers=8)
for tf, models in models_dict.items():
    my_trainer = L.Trainer(devices=[device_id], accelerator='gpu', enable_progress_bar=True)
    lst_preds = []
    for model in models:
        if not regr_mode:
            pred = F.sigmoid(torch.cat(my_trainer.predict(model= model ,dataloaders= test_dataloader))).numpy(force= True)
        else:
            pred = torch.cat(my_trainer.predict(model= model ,dataloaders= test_dataloader)).numpy(force= True)
        lst_preds.append(pred)
    preds = np.stack(lst_preds, axis=0).mean(0)
    if regr_mode: 
        mins, maxs = preds.min(), preds.max()
        preds = (preds - mins)/(maxs - mins)
    template[tf] = preds.round(5)

black_lst = template.iloc[0,[True if i=='nodata' else False for i in template.iloc[0,:]]].index.to_list()
selected_tfs = template.loc[:, [i for i in template.columns if i not in black_lst]]
if not os.path.isfile(submit_path):
    selected_tfs.to_csv(submit_path, sep = '\t')
else:
    old_subm = pd.read_csv(submit_path,delimiter='\t', index_col=0)
    new_subm = old_subm.join(selected_tfs)
    new_subm.to_csv(submit_path, sep = '\t')
