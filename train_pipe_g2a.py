import argparse
import os
import sys
sys.path.append('./ibis-challenge/')

from libis.general_max import LitLegNetMax, get_default_params_G2A
from libis.prep import dataset_generation
from libis.utils import AUROC_callback, LibDatasetExp2Exp, generate_name, read_configs_new
import lightning as L
from pytorch_lightning.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
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

dicipline = 'G2A'
EXP_TYPE_1 = 'CHS'
EXP_TYPE_2 = 'GHTS'

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

MODEL_CLASS = LitLegNetMax
BATCH_SIZE = 64
path_to_raw = 'train/'
model_name = 'LegNetMax'
log_file = f'{model_name}_{dicipline}_{tf_name}'
log_path = os.path.join('mlruns', log_file)
check_dir = os.path.join('checkpoints',dicipline, tf_name,exp_name)
dataset_path_chs = os.path.join('datasets',EXP_TYPE_1, neg_type, tf_name, f'{tf_name}.parquet.gzip')
dataset_path_ghts = os.path.join('datasets',EXP_TYPE_2, neg_type, tf_name, f'{tf_name}.parquet.gzip')
dataset_dir_chs = os.path.join('datasets',EXP_TYPE_1, neg_type)
dataset_dir_ghts = os.path.join('datasets',EXP_TYPE_2, neg_type)
config_path = os.path.join('configs', dicipline, model_name, tf_name,f'{config_file}.json')
version_path = glob(f'{log_path}*')

if not os.path.exists(dataset_path_ghts):
    data_ghts = dataset_generation(exp_type=EXP_TYPE_2, tf_name=tf_name ,num_workers=n_workers, 
                            path_to_data= path_to_raw, neg_type = neg_type)
    os.makedirs(dataset_dir_ghts, exist_ok=True)
    data_ghts.to_parquet(dataset_path_ghts, index=False)
else:
    data_ghts = pd.read_parquet(dataset_path_ghts)

if not os.path.exists(dataset_path_chs):
    data_chs = dataset_generation(exp_type=EXP_TYPE_1, tf_name=tf_name ,num_workers=n_workers, 
                            path_to_data= path_to_raw, neg_type = neg_type)
    os.makedirs(dataset_dir_chs, exist_ok=True)
    data_chs.to_parquet(dataset_path_chs, index=False)
else:
    data_chs = pd.read_parquet(dataset_path_chs)

os.makedirs(check_dir, exist_ok=True) if not TUNE_MODE else None

data = pd.concat([data_chs, data_ghts], axis = 0)
data.reset_index(drop=True, inplace=True)

Kfolder = GroupKFold(n_splits = 5)
for i, (train_idx, val_idx) in enumerate(Kfolder.split(data.seq.values, groups=data.chr.values)):
    s = seed + i
    X_train, y_train = data.loc[train_idx,'seq'].values, data.loc[train_idx,'label'].values
    X_val, y_val = data.loc[val_idx,'seq'].values, data.loc[val_idx,'label'].values
    train_dataset = LibDatasetExp2Exp(data=X_train, target=y_train,target_len=300 ,rev_compl_aug=True, noisy=False)
    val_dataset = LibDatasetExp2Exp(data=X_val, target=y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=n_workers,shuffle=True) 
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE, num_workers=n_workers,shuffle=True)
    if os.path.isfile(config_path):
        model_kws, hparams = read_configs_new(config_path)
    else:
        print("This launch will be based on default hyperparameters")
        hparams, model_kws = get_default_params_G2A()


    model = MODEL_CLASS( model_kws, hparams, criterion=nn.BCEWithLogitsLoss, seed= s)
    run_name = generate_name(check_dir, model_name = model_name)
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

    callbacks = [ AUROC_callback(),checkpoint_callback ] if not TUNE_MODE else [AUROC_callback()]
    my_trainer = L.Trainer(accelerator='gpu', 
                        devices= [device_id], max_epochs=hparams['epochs'], default_root_dir=check_dir,
                        enable_checkpointing= (not TUNE_MODE), 
                        callbacks=callbacks, 
                        logger=logger, log_every_n_steps=1,
                        limit_train_batches=hparams['num_steps']//hparams['epochs'])

    my_trainer.fit(model=model,train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)