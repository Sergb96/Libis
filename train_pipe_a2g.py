import argparse
import os
import sys
sys.path.append('./ibis-challenge/')

from libis.general_new import LitLegNet_new, get_default_params_A2G
from libis.utils import AUROC_callback, Lib_Dataset, generate_name, read_configs_new
import lightning as L
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
import re
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--neg_type',required=True)
parser.add_argument('--tf_name',required=True) 
parser.add_argument('--seed', default=29, type=int)
parser.add_argument('--n_workers', default=1, type=int)
parser.add_argument('--device_id', type=int, required=True)
parser.add_argument('--config_file', default= 'not_stated')
parser.add_argument('--exp_name', required=True)
parser.add_argument('-tune_mode', action= 'store_true')

args = parser.parse_args()

EXP_TYPE = 'HTS'
dicipline = 'A2G'
neg_type = args.neg_type
tf_name = args.tf_name
seed = args.seed
n_workers = args.n_workers
exp_name = args.exp_name
TUNE_MODE = args.tune_mode
config_file = args.config_file
if config_file == 'not_stated':
    config_file = exp_name
device_id = args.device_id
torch.set_float32_matmul_precision('high')

np.random.seed(seed)

BATCH_SIZE = 1024
MODEL_CLASS = LitLegNet_new
path_to_raw = './train/'
model_name = 'LegNet_new'
log_file = f'{model_name}_{EXP_TYPE}_{tf_name}'
log_path = f'./mlruns/{log_file}/'
check_dir = f'./checkpoints/{dicipline}/{tf_name}/{exp_name}/'
dataset_path = f'./datasets/{EXP_TYPE}/{neg_type}/{tf_name}.parquet.gzip'
dataset_dir = f'./datasets/{EXP_TYPE}/{neg_type}/'
config_path = f'./configs/{EXP_TYPE}/{model_name}/{tf_name}/{config_file}.json'
version_path = glob(f'{log_path}*')

if not os.path.exists(dataset_path):
    from libis.prep import dataset_generation
    data = dataset_generation(EXP_TYPE=EXP_TYPE, tf_name=tf_name ,num_workers=n_workers, 
                            path_to_data= path_to_raw, neg_type = neg_type)
    os.makedirs(dataset_dir) if not os.path.exists(dataset_dir) else None
    data.to_parquet(dataset_path, index=False)
else:
    data = pd.read_parquet(dataset_path)


if not os.path.exists(check_dir) and not TUNE_MODE:
    os.makedirs(check_dir)

# select the biggest replic 
data = data.loc[data.replic == 0,:]
data.index = list(range(data.shape[0]))

Kfolder = KFold(n_splits = 5, shuffle=True)
for i, (train_idx, val_idx) in enumerate(Kfolder.split(data.seq.values)):
    s = seed + i
    X_train, y_train = data.loc[train_idx,'seq'].values, data.loc[train_idx,'cycle'].values
    X_val, y_val = data.loc[val_idx,'seq'].values, data.loc[val_idx,'cycle'].values
    train_dataset = Lib_Dataset(data=X_train, target=y_train, rev_compl_aug=True, noisy=False)
    val_dataset = Lib_Dataset(data=X_val, target=y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=n_workers,shuffle=True) 
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE, num_workers=n_workers,shuffle=True)
    if os.path.isfile(config_path):
        model_kws, hparams = read_configs_new(config_path)
    else:
        print("This launch will be based on default hyperparameters")
        hparams, model_kws = get_default_params_A2G()

    
    model = MODEL_CLASS( model_kws, hparams, criterion=nn.MSELoss)
    run_name = generate_name(check_dir, model_name = 'LegNet_new')
    logger = MLFlowLogger(save_dir=log_path ,experiment_name = f'{tf_name}_{exp_name}', 
                        run_name= run_name,
                        tracking_uri='http://localhost:5005' ) 
    checkpoint_callback = ModelCheckpoint(
        save_top_k= 1,
        monitor="val/AUROC" if val_dataloader is not None else 'train/AUROC',
        mode="max",
        dirpath = check_dir,
        filename = run_name,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + '_last_{epoch}'
    ea_stop = EarlyStopping(monitor='val/loss', patience=40, mode='min')

    callbacks = [ AUROC_callback(),checkpoint_callback,ea_stop ] if not TUNE_MODE else [AUROC_callback(),ea_stop]
    my_trainer = L.Trainer(accelerator='gpu', 
                        devices= [device_id], max_epochs=hparams['epochs'], default_root_dir=check_dir,
                        enable_checkpointing= (not TUNE_MODE), 
                        callbacks=callbacks, 
                        logger=logger, log_every_n_steps=1,
                        limit_train_batches=hparams['num_steps']//hparams['epochs'])

    my_trainer.fit(model=model,train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
