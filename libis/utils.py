from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import lightning as L
from sklearn.metrics import roc_auc_score
import json
from funkybob import RandomNameGenerator
import os


def read_configs(path:str, dataset_size:int):
    with open(path) as fl:
        params = json.load(fl)
    model_kws, hparams = params['model_kws'], params['hparams']
    hparams['num_steps'] = hparams['epochs'] * dataset_size
    model_kws['activation'] = torch.nn.SiLU if model_kws['activation'] == 'SiLU' else None
    if model_kws['activation']  is None:
        raise ValueError('This activation function is not supported')
    return model_kws, hparams


def read_configs_new(path:str,steps_per_epoch:int=512):
    with open(path) as fl:
        params = json.load(fl)
    model_kws, hparams = params['model_kws'], params['hparams']
    hparams['num_steps'] = hparams['epochs'] * steps_per_epoch
    model_kws['activation'] = torch.nn.SiLU if model_kws['activation'] == 'SiLU' else None
    if model_kws['activation']  is None:
        raise ValueError('This activation function is not supported')
    return model_kws, hparams

def read_configs_BI(path:str, dataset_size:int):
    if os.path.isfile(path):
        with open(path) as fl:
            params = json.load(fl)
            hparams = params['hparams']
        hparams['num_steps'] = hparams['epochs'] * dataset_size
    else:
        hparams = None
    return hparams


def read_configs_dream_rnn(path:str, dataset_size:int):
    with open(path) as fl:
        params = json.load(fl)
    model_kws_first, model_kws_core,model_kws_final  = params['model_kws_first'], params['model_kws_core'], params['model_kws_final']
    hparams = params['hparams']
    hparams['num_steps'] = hparams['epochs'] * dataset_size
    return model_kws_first, model_kws_core,model_kws_final, hparams


class Seq2Tensor(torch.nn.Module):
    CODES = dict(zip('ACGTN', range(5)))
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def n2id(n):
        return Seq2Tensor.CODES[n.upper()]

    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [self.n2id(x) for x in seq.upper()]
        code = torch.tensor(seq)
        code = torch.nn.functional.one_hot(code, num_classes=5).float()
        code[code[:, 4] == 1] = 0.25
        code = code[:, :4]
        return code.transpose(0, 1)
    
class Seq2Tensor_noisy(torch.nn.Module):
    CODES = dict(zip('ACGTN', range(5)))
    def __init__(self, mut_tol = 0.1):
        super().__init__()
        self.mut_tol = mut_tol
        
    @staticmethod
    def n2id(n):
        return Seq2Tensor.CODES[n.upper()]

    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [self.n2id(x) for x in seq.upper()]
        code = torch.tensor(seq)
        code = torch.nn.functional.one_hot(code, num_classes=5).float()
        code[code[:, 4] == 1] = 0.25
        code = code[:, :4]
        code[code == 0] = self.mut_tol
        code[code == 1] = 1 - self.mut_tol * 3
        return code.transpose(0, 1)
    




def generate_name(check_dir:str, model_name:str) -> str:
    name = f'{model_name}_{next(iter(RandomNameGenerator()))}'
    used_names = glob(f'{check_dir}{name}*')
    while len(used_names) != 0:
        name = f'LegNet_{next(iter(RandomNameGenerator()))}'
    return name

class LibDatasetArtificial(Dataset):
        
    def __init__(self, data, target = None, rev_compl_aug = False, 
                 noisy = False, mut_tol=0.1, reverse_always=False):
        self.use_reverse = rev_compl_aug
        if noisy:
            self.transform = Seq2Tensor_noisy(mut_tol)
        else:
            self.transform = Seq2Tensor()

        self.reverse_marks = np.zeros(len(data), dtype='bool')
        if self.use_reverse:
            self.reverse_marks = np.concatenate([self.reverse_marks,np.ones(len(data), dtype='bool')], axis = 0) 
        self.x = data
        if target is None:
            self.y = None
        else:    
            assert len(data) == len(target)
            self.y = torch.FloatTensor(target)
        
        self.reverse_always = reverse_always
    @staticmethod
    def reverse_complement(seq_tensor):
        return seq_tensor.flip(dims=[-2,-1])

    def __len__(self):
        return len(self.reverse_marks)
    
    def __getitem__(self, idx):
        rev_out = self.reverse_marks[idx]
        true_idx = idx - len(self.x) if rev_out else idx

        seq, target = (self.x[true_idx], self.y[true_idx]) if self.y is not None else (self.x[true_idx], None)
        seq_tensor = self.transform(seq)

        if rev_out:
            seq_tensor = self.reverse_complement(seq_tensor)
        
        if self.reverse_always:
            seq_tensor = self.reverse_complement(seq_tensor)
        item = (seq_tensor, target) if target is not None else seq_tensor
        return item

# class Lib_Dataset_G2A(Dataset):
        
#     def __init__(self, data, target = None,target_len = 300, rev_compl_aug = False, noisy = False, mut_tol = 0.1,
#                  rev_comp_always = False):
#         self.use_reverse = rev_compl_aug
#         self.rev_always = rev_comp_always
#         if noisy:
#             self.transform = Seq2Tensor_noisy(mut_tol=mut_tol)
#         else:
#             self.transform = Seq2Tensor()
#         self.reverse_marks = np.zeros(len(data), dtype='bool')
#         if self.use_reverse:
#             self.reverse_marks = np.concatenate([self.reverse_marks,np.ones(len(data), dtype='bool')], axis = 0) 
#         self.x = data
#         if target is None:
#             self.y = None
#         else:    
#             assert len(data) == len(target)
#             self.y = torch.FloatTensor(target)

#         self.target_len = target_len
        
#     @staticmethod
#     def reverse_complement(seq_tensor):
#         return seq_tensor.flip(dims=[-2,-1])

#     def __len__(self):
#         return len(self.reverse_marks)
    
#     def __getitem__(self, idx):
#         rev_out = self.reverse_marks[idx]
#         true_idx = idx - len(self.x) if rev_out else idx

#         seq, target = (self.x[true_idx], self.y[true_idx]) if self.y is not None else (self.x[true_idx], None)
#         seq_tensor = self.transform(seq)

#         if self.rev_always:
#             seq_tensor = self.reverse_complement(seq_tensor)
#         elif rev_out:
#             seq_tensor = self.reverse_complement(seq_tensor)
        
#         ln_seq = len(seq)
#         if ln_seq <= self.target_len:
#             l_p, r_p = floor((self.target_len - ln_seq)/2),ceil((self.target_len - ln_seq)/2)
#             seq_tensor = F.pad(seq_tensor,pad=(l_p, r_p))
#         else:
#             ln_seq = seq_tensor.shape[1]
#             len_l = (seq_tensor.shape[1] - self.target_len)//2
#             len_r = (seq_tensor.shape[1] - self.target_len) - len_l
#             seq_tensor = seq_tensor[:,len_l:-len_r]

#         item = (seq_tensor, target) if target is not None else seq_tensor
#         return item

class LibDatasetExp2Exp(Dataset):
    ALPHABET = 'ACGT'
    ALPHABET_encoded = Seq2Tensor()(ALPHABET)
    NUCL_CONTENT = torch.tensor([0.295, 0.205, 0.205, 0.295])
    def __init__(self, data, target_len,target = None, rev_compl_aug = False, reverse_always = False):
        self.target_len = target_len
        self.use_reverse = rev_compl_aug
        self.transform = Seq2Tensor()
        self.reverse_marks = np.zeros(len(data), dtype='bool')
        if self.use_reverse:
            self.reverse_marks = np.concatenate([self.reverse_marks,np.ones(len(data), dtype='bool')], axis = 0) 
        self.x = data
        if target is None:
            self.y = None
        else:    
            assert len(data) == len(target)
            self.y = torch.FloatTensor(target)
        self.reverse_always = reverse_always

    @staticmethod
    def reverse_complement(seq_tensor:torch.Tensor):
        return seq_tensor.flip(dims=[-2,-1])
    
    def prune_seq_tensor(self, seq_tensor:torch.Tensor):
        mid = seq_tensor.shape[1]//2
        l, r = mid - self.target_len//2, mid + self.target_len//2
        return seq_tensor[:,l:r]
    
    def add_flanks(self, seq_tensor: torch.Tensor):
        len_l = (self.target_len - seq_tensor.shape[1])//2
        len_r = (self.target_len - seq_tensor.shape[1]) - len_l
        generated_idx_l = torch.multinomial(LibDatasetExp2Exp.NUCL_CONTENT,len_l , replacement=True)
        generated_idx_r = torch.multinomial(LibDatasetExp2Exp.NUCL_CONTENT, len_r, replacement=True)
        seq_tensor = torch.cat([LibDatasetExp2Exp.ALPHABET_encoded[:,generated_idx_l], 
                              seq_tensor,
                              LibDatasetExp2Exp.ALPHABET_encoded[:,generated_idx_r]], dim = 1)
        return seq_tensor
    

    def __len__(self):
        return len(self.reverse_marks)
    
    def __getitem__(self, idx:int):
        rev_out = self.reverse_marks[idx]
        true_idx = idx - len(self.x) if rev_out else idx

        seq, target = (self.x[true_idx], self.y[true_idx]) if self.y is not None else (self.x[true_idx], None)
        seq_tensor = self.transform(seq)
        if seq_tensor.shape[1] != self.target_len:
            seq_tensor = self.add_flanks(seq_tensor) if seq_tensor.shape[1] < self.target_len else self.prune_seq_tensor(seq_tensor)

        if rev_out:
            seq_tensor = self.reverse_complement(seq_tensor)
        
        if self.reverse_always:
            seq_tensor = self.reverse_complement(seq_tensor)

        item = (seq_tensor, target) if target is not None else seq_tensor
        return item




class AUROC_callback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_actual_labels = []
        self.train_pred = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.train_actual_labels.append(outputs["actual_labels"])
        self.train_pred.append(outputs["pred_labels"])

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_actual_labels = np.concatenate(self.train_actual_labels)
        self.train_pred = np.concatenate(self.train_pred)
        pl_module.log("train/AUROC", roc_auc_score(self.train_actual_labels,
                                                                  y_score=self.train_pred))

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_actual_labels = []
        self.val_pred = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_actual_labels.append(outputs['actual_labels'])
        self.val_pred.append(outputs['pred_labels'])

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_actual_labels = np.concatenate(self.val_actual_labels)
        self.val_pred = np.concatenate(self.val_pred)
        
        score = roc_auc_score(self.val_actual_labels, y_score=self.val_pred)
        pl_module.auroc.append(score)
        pl_module.log("val/AUROC", score)
        self.auroc = score
        
        score_25, score_50 = self.cust_scorer()
        self.auroc25 = score_25
        self.auroc50 = score_50
        self.mean_auc = (score_50+score_25+score)/3
        pl_module.log("val/auroc25", score_25)
        pl_module.log("val/auroc50", score_50)
        pl_module.log("val/mean_auc", self.mean_auc)

    def cust_scorer(self):
        sort_id = np.argsort(self.val_pred)
        self.val_pred = self.val_pred[sort_id]
        self.val_actual_labels = self.val_actual_labels[sort_id]
        neg_mask = np.isclose(self.val_actual_labels, 0)
        negatives = self.val_pred[neg_mask]
        positives = self.val_pred[~neg_mask]

        n25_pos = positives.shape[0]//4
        n25_neg = negatives.shape[0]//4
        n50_pos = positives.shape[0]//2
        n50_neg = negatives.shape[0]//2
        
        pred_scores_25 = np.concatenate([negatives[-n25_neg:], positives[-n25_pos:]])  
        pred_scores_50 = np.concatenate([negatives[-n50_neg:], positives[-n50_pos:]])
        y_true_25 = np.concatenate([np.zeros(len(negatives[-n25_neg:])), np.ones(len(positives[-n25_pos:]))],axis=0)
        y_true_50 = np.concatenate([np.zeros(len(negatives[-n50_neg:])), np.ones(len(positives[-n50_pos:]))],axis=0)

        score_25 = roc_auc_score(y_true = y_true_25,y_score= pred_scores_25)
        score_50 = roc_auc_score(y_true = y_true_50,y_score = pred_scores_50)
        return score_25, score_50
