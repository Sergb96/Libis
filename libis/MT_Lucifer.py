import os
import numpy as np
import pdb
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

import einops
from collections import OrderedDict
from random import randint
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torchmetrics import Accuracy
from lightning import pytorch as pl
from rotary_embedding_torch import RotaryEmbedding


def get_default_params_A2G():
    params = dict(
            lr = 1e-5,
            wd = 1e-3,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 200*512,
            epochs = 200
            )
    return params
def get_default_params_G2A():
    params = dict(
            lr = 1e-5,
            wd = 1e-3,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 100*512,
            epochs = 100
            )
    return params


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", dilation=1, bias=True, gn_num_groups=None, gn_group_size=16):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                             stride=stride, padding=padding, dilation=dilation, bias=bias)
        if gn_num_groups is None:
            gn_num_groups = out_channels // gn_group_size
        self.gn = nn.GroupNorm(gn_num_groups, out_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs):
        seq = inputs
        x = self.gn(F.gelu(self.cnn(seq)))
        x = self.dropout(x)
        
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn_num_groups=None, gn_group_size=16):
        super().__init__()

        stride_for_conv1_and_shortcut = 1

        if in_channels != out_channels:
            stride_for_conv1_and_shortcut = 2

        padding = kernel_size // 2

        if gn_num_groups is None:
            gn_num_groups = out_channels // gn_group_size

        # modules for processing the input
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride_for_conv1_and_shortcut, padding = padding, bias=False)
        self.gn1 = nn.GroupNorm(gn_num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, stride = 1, padding = "same", bias=False)
        self.gn2 = nn.GroupNorm(gn_num_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # short cut connections
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride_for_conv1_and_shortcut, bias=False)

    def forward(self, xl):
        input = self.shortcut(xl)

        xl = self.relu1(self.gn1(self.conv1(xl)))
        xl = self.conv2(xl)

        xlp1 = input + xl

        xlp1 = self.relu2(self.gn2(xlp1))

        return xlp1
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_dim, dropout=0.1, use_position_embedding=True):
        assert d_model % nhead == 0
        super().__init__()
        embedding_dim = d_model
        self.embedding_dim = embedding_dim
        self.num_heads = nhead
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.use_position_embedding = use_position_embedding

        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.xk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.xq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.xv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        if self.use_position_embedding:
            self.rotary_emb = RotaryEmbedding(dim=embedding_dim // self.num_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, inputs):
        x = self.layer_norm1(inputs)
        xk = self.xk(x)
        xq = self.xq(x)
        xv = self.xv(x)

        xk = xk.reshape(xk.shape[0], xk.shape[1], self.num_heads, self.embedding_dim // self.num_heads)
        xq = xq.reshape(xq.shape[0], xq.shape[1], self.num_heads, self.embedding_dim // self.num_heads)
        xv = xv.reshape(xv.shape[0], xv.shape[1], self.num_heads, self.embedding_dim // self.num_heads)

        if self.use_position_embedding:
            # make xq and xk have shape (batch_size, num_heads, seq_len, embedding_dim // num_heads)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
            xq = self.rotary_emb.rotate_queries_or_keys(xq, seq_dim=2)
            xk = self.rotary_emb.rotate_queries_or_keys(xk, seq_dim=2)
            # make xq and xk have shape (batch_size, seq_len, num_heads, embedding_dim // num_heads)
            xq = xq.permute(0, 2, 1, 3)
            xk = xk.permute(0, 2, 1, 3)
        
        attention_weights = einops.einsum(xq, xk, '... q h d, ... k h d -> ... h q k')

        attention_weights = attention_weights / np.sqrt(self.embedding_dim // self.num_heads)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout1(attention_weights)
        attention_output = einops.einsum(attention_weights, xv, '... h q k, ... k h d -> ... q h d')
        attention_output = einops.rearrange(attention_output, '... h d -> ... (h d)')
        attention_output = self.fc1(attention_output)
        attention_output = self.dropout2(attention_output)

        mlp_inputs = attention_output + inputs
        x = self.layer_norm2(mlp_inputs)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = x + mlp_inputs

        return x
    
class MTLucifer(nn.Module):
    def __init__(self, nucleotide_embed_dims=1024, nheads=8, mlp_dim_ratio=4):
        super().__init__()
        self.nheads = nheads
        self.cls_token_embedding = nn.Parameter(torch.normal(mean=0.0, std=0.02, size=(1, 1, nucleotide_embed_dims)))
        self.embed_dims = nucleotide_embed_dims
        self.nheads = nheads
        self.mlp_dim = nucleotide_embed_dims * mlp_dim_ratio
        
        self.promoter_cnn = nn.Sequential(
                                            CNNBlock(in_channels = 4, out_channels = 256, kernel_size = 5, stride = 1, bias=True),
                                            CNNBlock(in_channels = 256, out_channels = 512, kernel_size = 5, stride = 1, bias=True),
                                            CNNBlock(in_channels = 512, out_channels = nucleotide_embed_dims, kernel_size = 5, stride = 1, bias=True)
                                         )
        self.promoter_transformer = nn.Sequential(
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, mlp_dim=self.mlp_dim),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, mlp_dim=self.mlp_dim),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, mlp_dim=self.mlp_dim),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, mlp_dim=self.mlp_dim),
                                            TransformerBlock(d_model=nucleotide_embed_dims, nhead=self.nheads, mlp_dim=self.mlp_dim)
                                        )
        
        self.head = nn.Sequential(nn.Linear(self.mlp_dim//4, self.mlp_dim//8),
                                  nn.BatchNorm1d(self.mlp_dim//8),
                                  nn.ReLU(),
                                  nn.Linear(self.mlp_dim//8, 1))
        
    def forward(self, seq):
        seq = self.promoter_cnn(seq)
        seq = seq.permute(0, 2, 1)
        seq = torch.hstack([self.cls_token_embedding.expand(seq.shape[0], -1, -1), seq])
        outs = self.promoter_transformer(seq)[:, 0]
        outs = self.head(outs).squeeze()
        return outs
    



class MTLuciferWrapper(pl.LightningModule):
    def __init__(self,
        hparams: dict = dict(
            lr = 1e-4,
            wd = 1e-4,
            div_fct = 1e1,
            final_div_fct = 1e4,
            num_steps = 1000,
        ),
        seed = None,
        criterion = nn.BCEWithLogitsLoss):
        
        super().__init__()
        if seed is not None:
            self.seed = seed
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        else:
            seed = randint(1, 100000)
            self.seed = seed
            generator = torch.Generator()
            generator.manual_seed(self.seed)

        for key in hparams.keys():
            self.hparams[key] = hparams[key]
        self.save_hyperparameters()
        
        self.model = MTLucifer() #default hyperparams

        self.criterion = criterion()
        self.auroc = []

        
    @staticmethod
    def initialize_weights(m):
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, (2 / n) ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def compute_loss(self, batch):
        seqs, targets = batch
        pred = self.model(seqs)
        loss = self.criterion(pred, targets)
        with torch.no_grad():
            score = F.sigmoid(pred)
        return loss, score

    def training_step(self, batch, batch_idx):
        loss, preds = self.compute_loss(batch)
        actual_labels = (batch[1] > 0).float()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss, 'actual_labels' : actual_labels.numpy(force=True), 'pred_labels':preds.numpy(force=True)}

    def predict_step(self, batch, batch_idx):
        seqs = batch
        pred = self.model(seqs)
        return pred

    def validation_step(self, batch, batch_idx):
        loss,preds = self.compute_loss(batch)
        actual_labels = (batch[1] > 0).float()
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss, 'actual_labels' : actual_labels.numpy(force=True), 'pred_labels':preds.numpy(force=True)}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'],
                                       weight_decay= self.hparams['wd'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr= self.hparams['lr'],
                                                        total_steps= self.hparams['num_steps'], 
                                                        div_factor=self.hparams['div_fct'],
                                                        final_div_factor=self.hparams['final_div_fct'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
                "strict": True,
                "name": scheduler.__class__.__name__,
            },
        }


