from typing import Any 
import torch
import torch.nn.functional as F
from torch import nn, Generator

from ..prixfixe import FinalLayersBlock
from .utils import initialize_weights


class Head_layer(nn.Module):
    '''
    Linear layer concatinating the results of max and average pooling
    '''

    def __init__(self,final_ch =  256, out_size = 1, 
                 drop_probs = [0.2,0.1], activation = nn.SiLU):
        super().__init__()
        self.avg_pooler = nn.AdaptiveAvgPool1d(output_size = 1) 
        self.head = nn.Sequential(nn.Linear(in_features= final_ch, 
                                            out_features= final_ch//2 ),
                                  nn.Dropout1d(drop_probs[0]),
                                  activation(),  
                                  nn.Linear(in_features= final_ch // 2, 
                                            out_features= final_ch // 4),
                                  nn.Dropout1d(drop_probs[1]),
                                  activation(),  
                                  nn.Linear(in_features= final_ch//4, 
                                            out_features= out_size))
    def forward(self, x):
        out = self.avg_pooler(x)
        out = torch.squeeze(out, dim = 2)
        out = torch.squeeze(self.head(out))     
        return out



class AutosomeFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int = 320,
        final_channels: int = 160,
        seqsize: int =  301,
        out_size: int = 1
    ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)

        self.mapper =  nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=final_channels,
                kernel_size=1,
                padding='same',
            ),
            nn.SiLU(),
        )
        self.head = Head_layer(final_ch = final_channels,
                               out_size = out_size,
                               )

    def forward(self, x):
        x = self.mapper(x)
        out = self.head(x)
        return out
   
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))
