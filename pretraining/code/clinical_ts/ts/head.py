__all__ = ['MLPHead', 'MLPHeadConfig', 'S4Head', 'S4HeadConfig']

import torch
import torch.nn as nn

from ..template_modules import HeadBase, HeadBaseConfig
from .s4_modules.s4_model import S4Model

import dataclasses
from dataclasses import dataclass


class Transpose(nn.Module):
    '''helper module: transpose operation that can be used like a nn.module'''
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
class MLPHead(HeadBase):
    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        self.multi_prediction = hparams_head.multi_prediction
        
        proj = []
        if(hparams_input_shape.length>0 and self.multi_prediction is False):#leave out pool and flatten if sequence is already pooled
            proj += [Transpose(1,2),torch.nn.AdaptiveAvgPool1d(1),nn.Flatten()] #transpose to bring input into expected format B,F,S for pooling

        if(hparams_head.mlp):# additional hidden layer as in simclr
            proj += [nn.Linear(hparams_input_shape.channels, hparams_input_shape.channels),nn.ReLU(inplace=True),nn.Linear(hparams_input_shape.channels, target_dim, bias=hparams_head.bias)]
        else:
            proj += [nn.Linear(hparams_input_shape.channels, target_dim, bias=hparams_head.bias)]
        self.proj = nn.Sequential(*proj)
    
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = self.output_shape.length if self.multi_prediction else 0
        
    def forward(self, **kwargs):
        return {"seq": self.proj(kwargs["seq"])}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class MLPHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.MLPHead"
    multi_prediction: bool = True # sequence level prediction
    
    mlp: bool = False #mlp in prediction head
    bias: bool = True  #bias for final projection in prediction head 


class S4Head(HeadBase):

    def __init__(self, hparams_head, hparams_input_shape, target_dim):
        '''S4 analogue of the pretraining head used in modified CPC https://arxiv.org/abs/2002.02848 (in addition to layer norm) can also be used as global prediction head'''
        super().__init__(hparams_head, hparams_input_shape, target_dim)

        d_input = None
        d_model = hparams_input_shape.channels

        if hparams_head.d_model != -1 and hparams_input_shape.channels != hparams_head.d_model:
            d_input = hparams_input_shape.channels
            d_model = hparams_head.d_model

        print(f"SSL Head: d_input = {d_input}")            

        self.s4 = S4Model(
            d_input = d_input, #matches output dim of the encoder
            d_output = target_dim,
            d_state = hparams_head.state_dim,
            d_model = d_model,
            n_layers = hparams_head.n_layers,
            #dropout = hparams_predictor.dropout, #use default
            #prenorm = hparams_predictor.prenorm, #
            l_max = hparams_input_shape.length,
            transposed_input = False,
            bidirectional=not(hparams_head.causal),
            layer_norm=not(hparams_head.batchnorm),
            pooling = not(hparams_head.multi_prediction),
            backbone = hparams_head.backbone)
        
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = target_dim
        self.output_shape.length = hparams_input_shape.length if hparams_head.multi_prediction else 0

    def forward(self, **kwargs):
        return {"seq": self.s4(kwargs["seq"])}
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class S4HeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.ts.head.S4Head"
    multi_prediction: bool = True # sequence level prediction or not

    state_dim:int = 64
    dropout:float=0.2
    prenorm:bool=False
    batchnorm:bool=False
    backbone:str="s42" #help="s4original/s4new/s4d")  

    causal:bool= True #causal layer (e.g. for CPC)

    n_layers:int = 1
    d_model:int = -1
