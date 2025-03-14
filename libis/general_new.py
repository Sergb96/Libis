from collections import OrderedDict
from math import ceil
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import pandas as pd

from lightning import pytorch as pl


def get_default_params_A2G():
    hparams = {"arch" :"LegNet_windowed",
            "conf":"conf_1",
            "lr" : 1e-3,
            "wd": 0,
            "div_fct" : 25,
            "final_div_fct": 1e4,
            "epochs":200,
            "num_steps":512*200,}
           
    model_kws = {"seqsize": 40,
            "in_channels": 4,
            "block_sizes": [64,64,32,16],
            "ks": 5,
            "init_ks": 5,
            "resize_factor": 2,
            "activation": nn.SiLU,
            "filter_per_group": 2,
            "se_reduction": 4,
            "se_window_size":15,
            "final_ch": 32,
            "head_window_size":40,
            'head_drop_probs' : [0.2, 0.1],
            "bn_momentum": 0.1,
            "out_size": 1,
            'triangle_mode' : False
            }

    return (hparams, model_kws)

def get_default_params_G2A():
    hparams = {"arch" :"LegNet_windowed",
            "conf":"conf_1",
            "lr" : 1e-3,
            "wd": 1e-5,
            "div_fct" : 25,
            "final_div_fct": 1e4,
            "epochs":100,
            "num_steps":50000}
    model_kws = {"seqsize": 40,
            "in_channels": 4,
            "block_sizes": [64,64,32,16],
            "ks": 5,
            "init_ks": 5,
            "resize_factor": 2,
            "activation": nn.SiLU,
            "filter_per_group": 2,
            "se_reduction": 4,
            "se_window_size":15,
            "final_ch": 32,
            "head_window_size":40,
            "bn_momentum": 0.1,
            "out_size": 1,
            'triangle_mode' : False}

    return (hparams, model_kws)

class StemConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, filter_sizes:list = [3,6,9,12]):
        super(StemConv, self).__init__()
        self.convs_list = []
        for size in filter_sizes:
            conv = nn.Conv1d(in_channels= in_channels, out_channels = out_channels//4, 
                                            kernel_size= size, padding='same')
            self.convs_list.append(conv)
        self.convs_list = nn.ModuleList(self.convs_list)
    
    def forward(self, x):
        lst_out = []
        for conv in self.convs_list:
            lst_out.append(conv(x))
        out = torch.cat(lst_out, dim = -2)
        return out

class WindowedSEBlock1d(nn.Module):
    def __init__(
        self,
        channels: int,
        se_reduction: int,
        window_size: int = 9,
        pooling_mode: str = 'max',
        padding_mode:int = 'same',
        bias: bool = True,
    ):
        super().__init__()
        if padding_mode == 'same':
            self.pad_tuple = (ceil((window_size -1)/2), (window_size-1)//2)
        elif padding_mode == None:
            self.pad_tuple = None
        if pooling_mode == 'max':
            self.windowed_pool = nn.MaxPool1d(
                kernel_size=window_size,
                stride = 1
            )
        elif pooling_mode == 'avg':
            self.windowed_pool = nn.AvgPool1d(
                kernel_size=window_size,
                stride=1,
            )
        else:
            raise ValueError("`pooling_mode` must be either 'max' or 'avg'")
        
        self.fc_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels // se_reduction,
                kernel_size=1,
                bias=bias,
            ),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=channels // se_reduction,
                out_channels=channels,
                kernel_size=1,
                bias=bias,
            ),
            nn.Sigmoid(),
        )

    def forward(self, X):
        if self.pad_tuple is None:
            X_mul = X
        else:
           X_mul = F.pad(X, pad = self.pad_tuple)
        X_mul = self.windowed_pool(X_mul)
        X_mul = self.fc_conv(X_mul)
        X = X * X_mul
        return X    


class WindowedHeadLayer(nn.Module):
    '''
    A layer of a model that scores a sequence in the window, aggregating the obtained logites using the triangle method
    '''

    def __init__(self,final_ch:int,
                out_size:int = 1,
                window_size:int = 40, 
                window_stride: int = 1,
                triangle_mode = False,
                drop_probs = [0.2,0.1]):
        super().__init__()
        self.triangle_mode = triangle_mode
        self.conv = nn.Sequential(nn.MaxPool1d(stride=window_stride, kernel_size=window_size),
                                  nn.Conv1d(kernel_size=1,in_channels=final_ch, out_channels=final_ch//2),
                                  nn.Dropout1d(drop_probs[0]),
                                  nn.SiLU(),
                                  nn.Conv1d(kernel_size=1,in_channels=final_ch//2, out_channels=final_ch//4),
                                  nn.Dropout1d(drop_probs[1]),
                                  nn.SiLU(),
                                  nn.Conv1d(kernel_size=1,in_channels=final_ch//4, out_channels=out_size))

    def forward(self, x ):
        out = self.conv(x)
        out = torch.squeeze(out, dim = 1)
        if self.triangle_mode:
            b_s, n_w = out.shape
            hlf_size, mod = divmod(n_w, 2)
            hlf_size += mod
            weights = torch.ones(n_w, dtype=torch.float32, device = out.device)
            weights[:hlf_size] = torch.linspace(0,1, steps=hlf_size) 
            weights[-hlf_size:] = torch.linspace(1,0, steps=hlf_size)   
            pred = out.matmul(weights)/(weights.sum())  
        else:
            pred = F.adaptive_max_pool1d(out, output_size=1)
        return pred


class SeqNN(nn.Module):
    """
    NoGINet neural network.

    Parameters
    ----------
    seqsize : int
        Sequence length.
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.

    """
    __constants__ = ('resize_factor')

    def __init__(self,
                 seqsize,
                 in_channels,
                 block_sizes=[256, 128],
                 init_ks=5,
                 ks=5,
                 resize_factor=4,
                 activation=nn.SiLU,
                 filter_per_group=2,
                 se_reduction=4,
                 se_window_size = 15,
                 final_ch=128,
                 head_window_size = 40,
                 head_drop_probs = [0.2, 0.1],
                 bn_momentum=0.1,
                 out_size = 1,
                 triangle_mode = False
                 ):
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.se_window_size = se_window_size
        self.seqsize = seqsize
        self.in_channels = in_channels
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        self.out_size = out_size
        self.head_window_size = head_window_size
        self.head_drop_probs = head_drop_probs
        self.triangle_mode = triangle_mode
        seqextblocks = OrderedDict()
        

        block = nn.Sequential(
            StemConv(in_channels=self.in_channels, 
                    out_channels=block_sizes[0]),
            nn.BatchNorm1d(block_sizes[0],
                           momentum=self.bn_momentum),
            activation()  
        )
        seqextblocks[f'blc0'] = block

        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=prev_sz,
                    out_channels=sz * self.resize_factor,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(),  


                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=sz * self.resize_factor,
                    kernel_size=ks,
                    groups=sz * self.resize_factor // filter_per_group,
                    padding='same',
                    bias=False, 
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(), 
                WindowedSEBlock1d(channels = sz * self.resize_factor, se_reduction = se_reduction,
                        window_size = self.se_window_size),
                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=prev_sz,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(prev_sz,
                               momentum=self.bn_momentum),
                activation(), 

            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * prev_sz,
                    out_channels=sz,
                    kernel_size=ks,
                    padding='same',
                    bias=False, 
                ),
                nn.BatchNorm1d(sz,
                               momentum=self.bn_momentum),
                activation(),  
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper = block = nn.Sequential(
            nn.Conv1d(
                in_channels=block_sizes[-1],
                out_channels=self.final_ch,
                kernel_size=1,
                padding='same',
            ),
            activation(), 
        )

        self.head = WindowedHeadLayer(final_ch=self.final_ch,
                                      out_size=self.out_size, 
                                      window_size=self.head_window_size,
                                      triangle_mode=self.triangle_mode,
                                      drop_probs=self.head_drop_probs)

    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)

        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x

    def forward(self, x, predict_score=True):
        f = self.feature_extractor(x)
        x = self.mapper(f)
        logits = torch.squeeze(self.head(x))
        return logits 


class LitLegNet_new(pl.LightningModule):
    def __init__(self,
        model_kws: dict = dict(
                    seqsize = 40,
                    in_channels = 4,
                    block_sizes=[256, 128],
                    init_ks=5,
                    ks=5,
                    resize_factor=4,
                    activation=nn.SiLU,
                    filter_per_group=2,
                    se_reduction=4,
                    se_window_size = 15,
                    final_ch=128,
                    head_window_size = 40,
                    head_drop_probs = [0.2, 0.1],
                    bn_momentum=0.1,
                    out_size = 1,
                    triangle_mode = False
                ),
        hparams: dict = dict(
            lr = 1e-2,
            wd = 1e-4,
            div_fct = 1e1,
            final_div_fct = 1e4,
            num_steps = 1000,
        ),
        seed = None,
        criterion : nn.MSELoss | nn.BCEWithLogitsLoss = nn.MSELoss
       ):
        
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        for key in hparams.keys():
            self.hparams[key] = hparams[key]
        self.save_hyperparameters()

        self.model = SeqNN(**model_kws)
        self.model.apply(self.initialize_weights)
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
        # elif isinstance(m, nn.Linear):
        #     m.weight.data.normal_(0, 0.001)
        #     if m.bias is not None:
        #         nn.init.constant_(m.bias.data, 0)

    def compute_loss(self, batch):
        seqs, targets = batch
        pred = self.model(seqs)
        loss = self.criterion(pred, targets)
        return loss, pred

    def training_step(self, batch, batch_idx):
        loss, preds = self.compute_loss(batch)
        actual_labels = (batch[1] > 0).float()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {'loss': loss, 'actual_labels' : actual_labels.numpy(force=True), 'pred_labels':preds.numpy(force=True)}

    def predict_step(self, batch, batch_idx):
        seqs = batch
        preds = self.model(seqs)
        return preds

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