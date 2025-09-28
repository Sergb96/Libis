from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torchmetrics import Accuracy
import pandas as pd

from lightning import pytorch as pl



def get_default_params_A2G():
    hparams = {"arch" :"LegNet",
            "conf":"conf_1",
            "lr" : 1e-3,
            "wd": 1e-5,
            "div_fct" : 25,
            "final_div_fct": 1e4,
            "epochs":200,
            "num_steps":512*200,}
           
    model_kws = {"seqsize": 40,
            "in_channels": 4,
            "block_sizes": [64,64,32,16],
            "ks": 5,
            "resize_factor": 2,
            "activation": nn.SiLU,
            "filter_per_group": 2,
            "se_reduction": 4,
            "final_ch": 32,
            'head_drop_probs' : [0.2, 0.1],
            "bn_momentum": 0.1,
            "stem_ks": [3,6,9,12]
            }

    return (hparams, model_kws)

def get_default_params_G2A():
    hparams = {"arch" :"LegNet",
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
            "resize_factor": 2,
            "activation": nn.SiLU,
            "filter_per_group": 2,
            "se_reduction": 4,
            "final_ch": 32,
            "head_drop_probs" : [0.2,0.1],
            "bn_momentum": 0.1,
            "stem_ks": [3,6,9,12]}

    return (hparams, model_kws)


class StemConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stem_ks:list = [3,6,9,12]):
        super(StemConv, self).__init__()
        self.convs_list = []
        for size in stem_ks:
            conv = nn.Conv1d(in_channels= in_channels, out_channels = out_channels//len(stem_ks), 
                                            kernel_size= size, padding='same')
            self.convs_list.append(conv)
        self.convs_list = nn.ModuleList(self.convs_list)
    
    def forward(self, x):
        lst_out = []
        for conv in self.convs_list:
            lst_out.append(conv(x))
        out = torch.cat(lst_out, dim = -2)
        return out


class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer.
    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.
    """

    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    


class HeadLayer(nn.Module):
    '''
    Linear layer concatinating the results of max and average pooling
    '''

    def __init__(self,final_ch ,out_size, pooling, 
                 drop_probs = [0.2,0.1], activation = nn.SiLU):
        super().__init__()
        _max_pooler = nn.AdaptiveMaxPool1d(output_size = 1) if pooling == 'all' or pooling == 'max' else None
        _avg_pooler = nn.AdaptiveAvgPool1d(output_size = 1) if pooling == 'all' or pooling == 'avg' else None
        self.poolers = nn.ModuleList([pooler for pooler in [_max_pooler, _avg_pooler] if pooler is not None])
        self.head = nn.Sequential(nn.Linear(in_features= final_ch * len(self.poolers), 
                                            out_features= final_ch * len(self.poolers) // 2),
                                  nn.Dropout1d(drop_probs[0]),
                                  activation(),  
                                  nn.Linear(in_features= final_ch * len(self.poolers) // 2, 
                                            out_features= final_ch * len(self.poolers) // 4),
                                  nn.Dropout1d(drop_probs[1]),
                                  activation(),  
                                  nn.Linear(in_features= final_ch * len(self.poolers)//4, 
                                            out_features= out_size))
    def forward(self, x ):
        out = []
        for pooler in self.poolers:
            out.append(pooler(x))
        out = torch.cat(out,dim = 1)
        out = torch.squeeze(out, dim = 2)
        out = torch.squeeze(self.head(out))     
        return out


class SeqNN(nn.Module):
    """
    NoGINet neural network.

    Parameters
    ----------
    seqsize : int
        Sequence length.
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
    pooling : str
        Polling mode in head. The default is avg
    stem_ks : str
        Sizes of filters in Stem block

    """
    __constants__ = ('resize_factor')

    def __init__(self,
                 seqsize,
                 in_channels,
                 block_sizes=[64,64,32,16],
                 ks=5,
                 resize_factor=4,
                 activation=nn.SiLU,
                 filter_per_group=2,
                 se_reduction=4,
                 final_ch=32,
                 head_drop_probs = [0.2,0.1],
                 bn_momentum=0.1,
                 pooling = 'avg',
                 stem_ks = [3,6,9,12]):
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.in_channels = in_channels
        self.final_ch = final_ch
        self.head_drop_probs = head_drop_probs
        self.bn_momentum = bn_momentum
        self.out_size = 1
        self.pooling = pooling
        seqextblocks = OrderedDict()
        

        block = nn.Sequential(
            StemConv(in_channels=self.in_channels, 
                out_channels=block_sizes[0],
                stem_ks=stem_ks),
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
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor,
                               momentum=self.bn_momentum),
                activation(), 
                SELayer(prev_sz, sz * self.resize_factor,
                        reduction=self.se_reduction),
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
                    bias=False
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

        self.head = HeadLayer(final_ch=self.final_ch,
                               out_size=self.out_size, pooling = self.pooling)

    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)

        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x

    def forward(self, x, predict_score=True):
        f = self.feature_extractor(x)
        x = self.mapper(f)
        logits = self.head(x)
        return logits 


class LitLegNet(pl.LightningModule):
    def __init__(self,
        model_kws: dict = dict(
                    seqsize=40,
                    in_channels=4,
                    block_sizes=[64, 64, 32, 16],
                    ks=5,
                    resize_factor=2,
                    activation=nn.SiLU,
                    filter_per_group=2,
                    se_reduction=4,
                    final_ch=32,
                    head_drop_probs = [0.2,0.1],
                    bn_momentum=0.1,
                    pooling = 'avg'
                ),
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