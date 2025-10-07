from collections import OrderedDict
from random import randint
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from lightning import pytorch as pl

from .prixfixe.bhi import BHIFirstLayersBlock,BHICoreBlock, BHIFinalLayersBlock
from .prixfixe.prixfixe import PrixFixeNet



def get_default_params_A2G():
    params = dict(
            lr = 1.5e-3,
            wd = 1e-2,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 200*512,
            epochs = 200
            )
    return params
def get_default_params_G2A():
    params = dict(
            lr = 1.5e-3,
            wd = 1e-2,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 50000,
            epochs = 100
            )
    return params

class BHI_wrapper(pl.LightningModule):
    def __init__(self,
        hparams: dict = dict(
            lr = 1e-4,
            wd = 1e-4,
            div_fct = 25,
            final_div_fct = 1e4,
            num_steps = 10000,
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

        first = BHIFirstLayersBlock(
        in_channels = 4,
        out_channels = 320,
        seqsize = 40,
        kernel_sizes = [9, 15],
        pool_size = 1,
        dropout = 0.2
        )

        core = BHICoreBlock(
            in_channels = first.out_channels,
            out_channels = 320,
            seqsize = first.infer_outseqsize(),
            lstm_hidden_channels = 320,
            kernel_sizes = [9, 15],
            pool_size = 1,
            dropout1 = 0.2,
            dropout2 = 0.5
            )

        final = BHIFinalLayersBlock(in_channels=core.out_channels, 
                                        seqsize=core.infer_outseqsize())
        self.model = PrixFixeNet(
            first=first,
            core=core,
            final=final,
            generator=generator
            )
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
        return loss, pred

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

