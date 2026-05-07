__all__ = ["JEPAMultiBlockMaskingModule", "JEPAMultiBlockMaskingConfig"]

from dataclasses import dataclass
import torch
import torch.nn as nn
from ..template_modules import MaskingBaseConfig

class JEPAMultiBlockMaskingModule(nn.Module):
    def __init__(self, hparams_masking, hparams_input_shape):
        super().__init__()
        self.timesteps = hparams_input_shape.length
        self.channels = hparams_input_shape.channels
        self.enc_mask_scale = hparams_masking.enc_mask_scale
        self.pred_mask_scale = hparams_masking.pred_mask_scale  
        self.nenc = hparams_masking.nenc
        self.npred = hparams_masking.npred
        self.min_keep = hparams_masking.min_keep
        self.allow_overlap = hparams_masking.allow_overlap
        self.mask_token = nn.Parameter(torch.randn([self.channels]))
        self._itr_counter = 0

    def step(self):
        self._itr_counter += 1
        return self._itr_counter - 1

    def _sample_block_size(self, generator, scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        
        max_keep = int(self.timesteps * mask_scale)
        block_length = max_keep
        
        block_length = max(self.min_keep, block_length)
        block_length = min(block_length, self.timesteps - 1)
            
        return block_length

    def _sample_block_masks_vectorized(self, b_size, batch_size, n_blocks, device, acceptable_regions=None):
        if self.timesteps - b_size <= 0:
            starts = torch.zeros(batch_size, n_blocks, dtype=torch.long, device=device)
        else:
            starts = torch.randint(0, self.timesteps - b_size, 
                                  (batch_size, n_blocks), device=device)
        
        timesteps_range = torch.arange(self.timesteps, device=device).view(1, 1, -1)
        starts_expanded = starts.unsqueeze(-1)
        
        masks = (timesteps_range >= starts_expanded) & (timesteps_range < starts_expanded + b_size)
        
        if acceptable_regions is not None:
            masks = masks & acceptable_regions.unsqueeze(1)
        
        num_kept = masks.sum(dim=-1)  # [batch, n_blocks]
        valid = num_kept >= self.min_keep
        
        for b in range(batch_size):
            for n in range(n_blocks):
                if not valid[b, n]:
                    idxs = torch.randperm(self.timesteps, device=device)[:self.min_keep]
                    masks[b, n] = 0
                    masks[b, n, idxs] = 1
        
        return masks

    def forward(self, seq, **kwargs):
        batch_size, timesteps, channels = seq.shape
        assert timesteps == self.timesteps, f"Input timesteps {timesteps} != expected {self.timesteps}"
        
        device = seq.device
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale)
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale)
        
        batch_masks_pred = self._sample_block_masks_vectorized(
            p_size, batch_size, self.npred, device, acceptable_regions=None
        )  # [batch, npred, timesteps]
        
        if not self.allow_overlap:
            masks_C = ~batch_masks_pred.any(dim=1)  # [batch, timesteps]
            acceptable_regions = masks_C
        else:
            acceptable_regions = None
        
        batch_masks_enc = self._sample_block_masks_vectorized(
            e_size, batch_size, self.nenc, device, acceptable_regions=acceptable_regions
        )  # [batch, nenc, timesteps]
        
        context_mask = batch_masks_enc.any(dim=1)  # [batch, timesteps]
        inverse_mask = ~context_mask  # [batch, timesteps]
        
        mask_token_expanded = self.mask_token.view(1, 1, -1).expand(batch_size, timesteps, -1)
        seq_masked = torch.where(inverse_mask.unsqueeze(-1), mask_token_expanded, seq)
        
        mask_ids = torch.nonzero(inverse_mask)
        
        return {
            "seq": seq_masked,
            "masks_enc": batch_masks_enc,
            "masks_pred": batch_masks_pred,
            "mask_ids": mask_ids
        }

@dataclass
class JEPAMultiBlockMaskingConfig(MaskingBaseConfig):
    _target_: str = "clinical_ts.mask.jepa_multiblock_mask.JEPAMultiBlockMaskingModule"
    mask_type: str = "jepa_multi_block_mask"
    enc_mask_scale: tuple = (0.85, 1.0)
    pred_mask_scale: tuple = (0.15, 0.2)
    nenc: int = 1
    npred: int = 4
    min_keep: int = 64
    allow_overlap: bool = False