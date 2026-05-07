__all__ = ['SupervisedLossConfig', 'BCELossConfig', 'BinaryCrossEntropyLoss']

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass, field
from typing import List

from clinical_ts.template_modules import LossConfig

####################################################################################
# BASIC supervised losses
###################################################################################
@dataclass
class SupervisedLossConfig(LossConfig):
    _target_:str = "" #insert appropriate loss class
    loss_type:str ="supervised"
    supervised_type:str="classification_single"#"classification_multi","regression_quantile"

class BinaryCrossEntropyLoss(nn.Module):
    #standard BCE loss that just passes the pos_weight correctly
    def __init__(self, hparams_loss):
        super().__init__()
        self.ignore_nans = hparams_loss.ignore_nans
        self.pos_weight_set = len(hparams_loss.pos_weight)>0

        if(not self.ignore_nans):
            self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(hparams_loss.pos_weight,dtype=np.float32)) if len(hparams_loss.pos_weight)>0 else None)
        else:
            if(self.pos_weight_set):
                self.bce = torch.nn.ModuleList([torch.nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array([hparams_loss.pos_weight[i]],dtype=np.float32))) for i in range(len(self.pos_weight_set))])
            else:
                self.bce = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targs):
        if(not self.ignore_nans):
            return self.bce(preds,targs)
        else:
            losses = []
            for i in range(preds.size(1)):
                predsi = preds[:,i]
                targsi = targs[:,i]
                maski = ~torch.isnan(targsi)
                predsi = predsi[maski]
                targsi = targsi[maski]
                if(len(predsi)>0):
                    if(self.pos_weight_set):
                        losses.append(self.bce[i](predsi,targsi))
                    else:
                        losses.append(self.bce(predsi,targsi))
                
            return torch.sum(torch.cat(losses)) if(len(losses)>0) else 0.
                    

@dataclass
class BCELossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.loss.supervised.BinaryCrossEntropyLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_multi"
    pos_weight:List[float]=field(default_factory=lambda: [])#class weights e.g. inverse class prevalences
    ignore_nans:bool=False #ignore nans- requires separate BCEs for each label
