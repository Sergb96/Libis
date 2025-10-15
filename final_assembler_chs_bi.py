from libis.general_max import LitLegNetMax
# # from libis.general_new_mean import LitLegNet_new
# # from libis.general_max_lstm import LitLegNet_LSTM
# # from libis.general import LitLegNet
# # from libis.BHI import BHI_wrapper
# # from libis.UNLOCK_DNA import UNLOCK_DNA_wrapper
# # from libis.DREAM_RNN import DREAM_RNN_wrapper
# from libis.Malinois import MalinoisWrapper



from libis.utils import LibDatasetExp2Exp
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
parser.add_argument('--template_dir', type = str, default='final')# directory with templates, script saves submit files here

args = parser.parse_args()
exp_type = 'CHS'
subm_name = args.subm_name
device_id = args.device_id
template_dir = args.template_dir
os.makedirs(f'./{template_dir}/{subm_name}',exist_ok=True)

paths = glob.glob(f'./selected_checkpoints/*/*/{subm_name}/*.ckpt')
A2G_lst = [path.split('/')[-1] for path in glob.glob('./selected_checkpoints/A2G/*')]
submit_path = f'./{template_dir}/{subm_name}/{exp_type}.tsv'
template = pd.read_csv(f'./{template_dir}/{exp_type}_aaa_template.tsv',sep='\t', index_col=0)
models_dict = dict()

skipped_tfs = []
for tf in template.columns.values:
    runs = list(filter(lambda x: tf in x, paths))
    if len(runs) == 0:
        skipped_tfs.append(tf)
        continue
    models_dict[tf] = [MODEL_CLASS.load_from_checkpoint(j, map_location=torch.device(device_id)) for j in runs]
print(f'The clear_checkpoints of the following TFs are not presented: {skipped_tfs}')

parser = SeqIO.parse(f'./{template_dir}/{exp_type}_participants.fasta', format = 'fasta')
tags, seqs = [], []
for record in parser:
    tags.append(record.name)
    seqs.append(str(record.seq))
test = pd.DataFrame(dict(tags=tags, seq=seqs))

test.set_index('tags', inplace=True)
test = test.loc[template.index.values,:]
lst_dataloaders = []
for ln in AGG_LENGTHS:
    test_dataset =  LibDatasetExp2Exp(data=test.seq.values, target_len=ln)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = 4096,num_workers=8)
    test_dataset_rev = LibDatasetExp2Exp(data=test.seq.values, target_len=ln, reverse_always=True)
    test_dataloader_rev = DataLoader(test_dataset_rev, shuffle=False, batch_size = 4096,num_workers=8)
    lst_dataloaders.append(test_dataloader)
    lst_dataloaders.append(test_dataloader_rev)

for tf, models in models_dict.items():
    regr_mode = True if tf in A2G_lst else False
    lst_tf_pred = []
    for model in models:
        lst_agg = []
        for test_dataloader in lst_dataloaders:
            my_trainer = L.Trainer(devices=[device_id], accelerator='gpu', enable_progress_bar=True)
            if not regr_mode:
                pred = F.sigmoid(torch.cat(my_trainer.predict(model= model ,dataloaders= test_dataloader))).numpy(force= True)
            else:
                pred = torch.cat(my_trainer.predict(model= model ,dataloaders= test_dataloader)).numpy(force= True)

            pred = pred.astype(np.float32)
            lst_agg.append(pred)
        preds = np.stack(lst_agg, axis=0).mean(0)
        lst_tf_pred.append(preds)
    pre_pred = np.stack(lst_tf_pred, axis=0).mean(0)
    if regr_mode:
        min_value, max_value = min(pre_pred), max(pre_pred)
        template[tf] = ((pre_pred - min_value)/(max_value - min_value)).round(5)
    else:
        template[tf] = pre_pred.round(5)

black_lst = template.iloc[0,[True if i=='nodata' else False for i in template.iloc[0,:]]].index.to_list()
selected_tfs = template.loc[:, [i for i in template.columns if i not in black_lst]]
if not os.path.isfile(submit_path):
    selected_tfs.to_csv(submit_path, sep = '\t')
else:
    old_subm = pd.read_csv(submit_path,delimiter='\t', index_col=0)
    new_subm = old_subm.join(selected_tfs)
    new_subm.to_csv(submit_path, sep = '\t')