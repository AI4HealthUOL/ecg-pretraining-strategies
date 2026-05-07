__all__ = ["MaskingModule", "MaskingConfig"]

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..template_modules import MaskingBaseConfig

class MaskingModule(nn.Module):
    def __init__(self, hparams_masking, hparams_input_shape):
        super().__init__()
        self.mask_probability = hparams_masking.mask_probability/hparams_masking.mask_span
        self.mask_span = hparams_masking.mask_span
        
        self.mask_item = nn.Parameter(torch.randn([hparams_input_shape.channels]).unsqueeze(dim=0))
        self.output_shape = hparams_input_shape
        
    def forward(self, seq, **kwargs):
        #bs,seq,features
        mask = self.calculate_mask(seq)
        mask_ids = torch.nonzero(mask)
        input_encoded_masked = seq.clone()
        input_encoded_masked[torch.where(mask)] = self.mask_item.repeat([len(mask_ids),1]).to(seq.dtype)
        return {"seq":input_encoded_masked, "mask_ids": mask_ids}#all mask_ids have shape bs,ts
    
    def calculate_mask(self, seq):
        bd=torch.distributions.bernoulli.Bernoulli(self.mask_probability)
        mask_sparse=bd.sample([seq.shape[0],seq.shape[1]]).long().to(seq.device)
        mask_full = mask_sparse.clone()
        midpoints= torch.nonzero(mask_sparse==1).cpu().numpy()#midpoints
        steps_after = (self.mask_span-1)//2 if (self.mask_span-1)%2==0 else self.mask_span//2
        steps_before = self.mask_span-steps_after-1
        
        for x in midpoints:
            mask_full[x[0],max(x[1]-steps_before,0):min(x[1]+steps_after+1,mask_sparse.shape[1]-1)]=1
        return mask_full
        #return mask_full, torch.nonzero(mask_sparse==1), torch.nonzero((mask_full-mask_sparse)==1) #full mask, midpoint ids, other masked ids

    def get_output_shape(self):
        return self.output_shape
    
    def __str__(self):
        return self.__class__.__name__+"\toutput shape:"+str(self.get_output_shape())

@dataclass
class MaskingConfig(MaskingBaseConfig):
    _target_:str = "clinical_ts.mask.base.MaskingModule"
    mask_probability:float = 0.065 #probability that a certain position in the sequence gets drawn as midpoint
    mask_span: int = 10 # draw mask_span surrounding positions around midpoints
