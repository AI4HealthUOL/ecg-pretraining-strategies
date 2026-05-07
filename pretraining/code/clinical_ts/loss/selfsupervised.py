__all__ = [
    'Data2VecLoss',
    'DinoSRLoss',
    'JEPALoss',
    'CPCLoss',
    'MaskedPredictionLossSinkhornKnoppKMeans',
    'Data2VecLossConfig',
    'DinoSRLossConfig',
    'JEPALossConfig',
    'CPCLossConfig',
    'MaskedPredictionLossSinkhornKnoppKMeansConfig',
    'MaskedLossConfig',
    'MaskedPredictionLossConfig'
]

from dataclasses import dataclass, field
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..template_modules import SSLLossConfig
from ..utils.schedulers import cosine_warmup


#base config class for masking
@dataclass
class MaskedLossConfig(SSLLossConfig):
    _target_:str = ""
    loss_type:str = "masked_"


@dataclass
class MaskedPredictionLossConfig(MaskedLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskedPredictionLoss"
    loss_type:str = "masked_pred"
    kmeans_ks:List[int]=field(default_factory=lambda: [50,100]) # output dimensions for each head
    temperature:float = 0.1 #0.15 # in general: slightly higher than the EMA temperature (default 0.07)
    temperature_warmup_start:float = 0.5 
    temperature_warmup_steps:int = 200 #0 for no warmup
    alpha:float = 1. #1.: only loss on the masked tokens, 0. only loss on the unmasked tokens


class Data2VecLoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.loss_beta = hparams_loss.loss_beta
        self.loss_scale = hparams_loss.loss_scale
        
    def forward(self, input_predicted, ema_features, mask_ids, **kwargs):
        if len(mask_ids) == 0:
            return {"loss": torch.tensor(0.0, device=input_predicted.device, requires_grad=True)}
        
        preds = input_predicted[mask_ids[:, 0], mask_ids[:, 1]]
        targets = ema_features[mask_ids[:, 0], mask_ids[:, 1]]

        targets = targets.detach()
        
        if self.loss_beta == 0:
            loss = F.mse_loss(preds.float(), targets.float(), reduction="none")
        else:
            loss = F.smooth_l1_loss(
                preds.float(),
                targets.float(),
                reduction="none",
                beta=self.loss_beta
            )
        
        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(preds.size(-1))
        
        loss = loss.sum(dim=-1).mean() * scale
        
        return {"loss": loss}

@dataclass
class Data2VecLossConfig(MaskedLossConfig):
    _target_: str = "clinical_ts.loss.selfsupervised.Data2VecLoss"
    loss_type: str = "data2vec"
    loss_beta: float = 1  # 0: MSE else Smooth L1
    loss_scale: float | None = None  # None: auto-scale by 1/sqrt(dim)


class DinoSRLoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.kmeans_ks_cumsum = np.cumsum(hparams_loss.kmeans_ks)
        self.temperature = getattr(hparams_loss, 'temperature', 1.0)
        self.num_codebooks = len(hparams_loss.kmeans_ks)
        
    def forward(self, input_predicted, ema_cluster_assignments, mask_ids, **kwargs):
        if len(mask_ids) == 0:
            return {"loss": torch.tensor(0.0, device=input_predicted.device, requires_grad=True)}

        input_predicted = input_predicted.view(
            input_predicted.size(0),
            input_predicted.size(1),
            -1,
            self.kmeans_ks_cumsum[-1]
        )

        total_loss = 0.0
        num_heads = 0
        
        for i in range(len(ema_cluster_assignments)):
            for j in range(len(ema_cluster_assignments[i])):
                start_idx = 0 if j == 0 else self.kmeans_ks_cumsum[j-1]
                end_idx = self.kmeans_ks_cumsum[j]
                preds = input_predicted[:, :, i, start_idx:end_idx]

                targets = ema_cluster_assignments[i][j]
                targets = targets.detach()
                
                preds_masked = preds[mask_ids[:, 0], mask_ids[:, 1]]
                targets_masked = targets[mask_ids[:, 0], mask_ids[:, 1]]
                
                logits = preds_masked / self.temperature
                log_probs = F.log_softmax(logits, dim=-1)
                
                loss = F.nll_loss(log_probs, targets_masked)
                
                total_loss += loss
                num_heads += 1
        
        final_loss = total_loss / num_heads
        
        return {"loss": final_loss}

@dataclass
class DinoSRLossConfig(MaskedLossConfig):
    _target_: str = "clinical_ts.loss.selfsupervised.DinoSRLoss"
    loss_type: str = "dinosr"
    kmeans_ks: List[int] = field(default_factory=list)  # Must be set based on average_top_k_layers
    temperature: float = 1.0


class JEPALoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.loss_fn_type = getattr(hparams_loss, "loss_fn_type", "smooth_l1")
        self.beta = getattr(hparams_loss, "beta", 1.0)
    
    def forward(self, input_predicted, ema_features, masks_pred, masks_enc=None, **kwargs):
        device = masks_pred.device
        batch_size, npred, timesteps = masks_pred.shape
        
        if npred == 0 or masks_pred.sum() == 0:
            return {"loss": torch.tensor(0.0, device=device, requires_grad=True)}
        
        mask_expanded = masks_pred.unsqueeze(-1)  # [batch, npred, timesteps, 1]
        mask_expanded = mask_expanded.expand(-1, -1, -1, input_predicted.size(-1))
        
        pred_expanded = input_predicted.unsqueeze(1).expand(-1, npred, -1, -1)
        target_expanded = ema_features.unsqueeze(1).expand(-1, npred, -1, -1)
        
        pred_masked = pred_expanded[mask_expanded].view(-1, input_predicted.size(-1))
        target_masked = target_expanded[mask_expanded].view(-1, ema_features.size(-1))
        
        if self.loss_fn_type == "smooth_l1":
            loss = nn.functional.smooth_l1_loss(
                pred_masked, target_masked, 
                reduction='mean', beta=self.beta
            )
        elif self.loss_fn_type == "mse":
            loss = nn.functional.mse_loss(pred_masked, target_masked, reduction='mean')
        else:
            raise ValueError(f"Unknown loss_fn_type: {self.loss_fn_type}")
        
        return {"loss": loss}

@dataclass
class JEPALossConfig(MaskedLossConfig):
    _target_: str = "clinical_ts.loss.selfsupervised.JEPALoss"
    loss_type: str = "jepa"
    loss_fn_type: str = "smooth_l1"  # Options: "smooth_l1", "mse"
    beta: float = 1.0


class CPCLoss(nn.Module):
    def __init__(self, hparams_loss):
        super().__init__()
        self.steps_predicted = hparams_loss.steps_predicted
        self.n_false_negatives = hparams_loss.n_false_negatives
        self.negatives_from_same_seq_only = hparams_loss.negatives_from_same_seq_only
        self.negatives_selection_interval = hparams_loss.negatives_selection_interval
        assert(self.negatives_from_same_seq_only is False or self.negatives_selection_interval==0 or self.negatives_selection_interval*2>= self.n_false_negatives)#make sure to have enough negatives available
        self.t = hparams_loss.temperature
        self.normalize = hparams_loss.normalize

    def forward(self, input_predicted, input_encoded, **kwargs):        
        bs = input_encoded.size(0)
        seq = input_encoded.size(1)
        device = input_predicted.device
        num_steps = seq - self.steps_predicted
        
        positives = input_encoded[:, self.steps_predicted:seq, :].unsqueeze(2)
        
        if self.negatives_selection_interval == 0:
            idxs_seq = torch.randint(0, seq - 1, (bs, num_steps, self.n_false_negatives), device=device)
            pos_idxs = torch.arange(self.steps_predicted, seq, device=device).view(1, num_steps, 1)
            idxs_seq = idxs_seq + (idxs_seq >= pos_idxs).long()
        else:
            i_vals = torch.arange(num_steps, device=device)
            starts = torch.clamp(i_vals - self.negatives_selection_interval, min=0)
            ends = torch.clamp(i_vals + self.negatives_selection_interval + 1, 
                            max=seq if self.negatives_selection_interval < self.steps_predicted else seq - 1)
            rand_vals = torch.rand(bs, num_steps, self.n_false_negatives, device=device)
            idxs_seq = starts.view(1, -1, 1) + (rand_vals * (ends - starts).float().view(1, -1, 1)).long()
            pos_idxs = (i_vals + self.steps_predicted).view(1, -1, 1)
            idxs_seq = torch.where(idxs_seq >= pos_idxs, idxs_seq + 1, idxs_seq)
            idxs_seq = torch.clamp(idxs_seq, max=seq - 1)
        
        if self.negatives_from_same_seq_only:
            idxs_batch = torch.arange(bs, device=device).view(bs, 1, 1).expand(bs, num_steps, self.n_false_negatives)
        else:
            idxs_batch = torch.randint(0, bs, (bs, num_steps, self.n_false_negatives), device=device)
        
        negatives = input_encoded[idxs_batch, idxs_seq, :]  # (bs, num_steps, n_neg, feat)
        candidates = torch.cat([positives, negatives], dim=2)  # (bs, num_steps, 1+n_neg, feat)
        preds = input_predicted[:, :num_steps, :]  # (bs, num_steps, feat)
        
        if self.normalize:
            candidates = F.normalize(candidates, p=2.0, dim=-1)
            preds = F.normalize(preds, p=2.0, dim=-1)
        
        sim = torch.einsum('bsf,bscf->bsc', preds, candidates) / self.t  # (bs, num_steps, 1+n_neg)
        sim_flat = sim.view(-1, 1 + self.n_false_negatives)
        targs_flat = torch.zeros(bs * num_steps, dtype=torch.long, device=device)
        
        loss = F.cross_entropy(sim_flat, targs_flat, reduction='sum') / bs
        tp_cnt = (sim_flat.argmax(dim=-1) == 0).sum()
        
        return {"loss": loss, "metric_acc": tp_cnt.float() / (bs * num_steps)}

@dataclass
class CPCLossConfig(SSLLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.CPCLoss"
    loss_type:str = "cpc"
    steps_predicted: int = 12
    n_false_negatives: int = 128
    negatives_from_same_seq_only:bool = False # help="only draw false negatives from same sequence (as opposed to drawing from everywhere)")
    negatives_selection_interval: int = 0 #only draw negative from -x...x around the current index
    normalize: bool = False #normalize before calculating similarities
    temperature: float = 1.0 #temperature parameter dividing similarities


class MaskedPredictionLossSinkhornKnoppKMeans(nn.Module):
    '''Masked prediction loss using soft SK assignments'''
    
    def __init__(self, hparams_loss):
        super().__init__()
        self.kmeans_ks_cumsum = np.cumsum(hparams_loss.kmeans_ks)
        self.alpha = hparams_loss.alpha
        self.temperature_warmup_start = hparams_loss.temperature_warmup_start
        self.temperature_warmup_end = hparams_loss.temperature
        self.temperature_warmup_steps = hparams_loss.temperature_warmup_steps
        self.temperature = hparams_loss.temperature if hparams_loss.temperature_warmup_steps == 0 else hparams_loss.temperature_warmup_start
        self.step_count = 0
        self.entropy_regularization = hparams_loss.entropy_regularization
    
    def forward(self, input_predicted, ema_soft_cluster_assignments, mask_ids, **kwargs):
        results_dict = {}
        
        # Determine device for empty case
        device = input_predicted[0].device if len(input_predicted) > 0 else 'cpu'

        #reformat combined input_predicted into entries corresponding to codebooks and layers
        #original shape is bs,seq,sum(codebook_dims)*layers
        input_predicted = input_predicted.view(input_predicted.size(0),input_predicted.size(1),-1, self.kmeans_ks_cumsum[-1])
        for i in range(len(ema_soft_cluster_assignments)):#layers
            for j in range(len(ema_soft_cluster_assignments[i])):#codebooks
                ip = input_predicted[:,:,i,(0 if j==0 else self.kmeans_ks_cumsum[j-1]):self.kmeans_ks_cumsum[j]]
                sc = ema_soft_cluster_assignments[i][j]
                
                # ip: [bs, seq, feat]
                # sc: [bs, seq, feat]
                assert ip.shape == sc.shape, f"Shape mismatch: {ip.shape} vs {sc.shape}"
                
                if self.alpha == 1.:
                    if(len(mask_ids)==0):
                        results_dict[f"loss_masked{i}_{j}"] = torch.tensor(0., device=device)
                    else:
                        # Only compute masked loss
                        preds = 10*F.log_softmax(ip[mask_ids[:, 0], mask_ids[:, 1]]/self.temperature, dim=-1)  # [N, feat]
                        targs = sc[mask_ids[:, 0], mask_ids[:, 1]]  # [N, feat]
                        loss_masked = F.kl_div(preds,targs, reduction='batchmean', log_target=False)  # pred is log, target is prob
                        results_dict[f"loss_masked{i}_{j}"] = loss_masked
                else:
                    # Compute both masked and non-masked loss
                    loss_noagg = F.kl_div(F.log_softmax(ip/self.temperature, dim=-1),sc, reduction='none', log_target=False).sum(dim=-1)  # [bs, seq]
                    # Masked loss
                    if(len(mask_ids)==0):
                        loss_masked = torch.tensor(0., device=device)
                        loss_masked_sum = torch.tensor(0., device=device)
                    else:
                        loss_masked_sum = torch.sum(loss_noagg[mask_ids[:, 0], mask_ids[:, 1]])
                        loss_masked = self.alpha * loss_masked_sum / len(mask_ids)
                    
                    # Non-masked loss
                    num_nonmasked = loss_noagg.numel() - len(mask_ids)
                    if(num_nonmasked==0):
                        loss_nonmasked = torch.tensor(0., device=device)
                    else:
                        loss_nonmasked_sum = torch.sum(loss_noagg) - loss_masked_sum
                        loss_nonmasked = (1 - self.alpha) * loss_nonmasked_sum / num_nonmasked
                    
                    results_dict[f"loss_masked{i}_{j}"] = loss_masked
                    results_dict[f"loss_nonmasked{i}_{j}"] = loss_nonmasked
                    
                    #debugging metrics
                    preds = ip.view(-1,ip.size(-1))
                    pred_probs = F.softmax(preds,dim=-1)
                    targs = sc.view(-1,sc.size(-1))
                    kl = F.kl_div( F.log_softmax(preds / self.temperature, dim=-1), targs, reduction='batchmean')
                    target_entropy = -(targs * torch.log(targs + 1e-8)).sum(dim=-1).mean()
                    pred_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean()
                    results_dict[f"metric_teacher_entropy{i}_{j}"]=target_entropy
                    results_dict[f"metric_student_entropy{i}_{j}"]=pred_entropy
                    results_dict[f"metric_ratio_student_entropy_to_teacher_entropy{i}_{j}"]=pred_entropy/target_entropy
                    results_dict[f"metric_kl_student_teacher{i}_{j}"]=kl
                    results_dict[f"metric_teacher_temperature{i}_{j}"]=self.temperature
                    # print("DEBUG",i,j)
                    # print(f"  Target entropy: {target_entropy:.4f}")
                    # print(f"  Pred entropy: {pred_entropy:.4f}")
                    # print(f"  Ratio p/t: {pred_entropy/target_entropy:.4f}")
                    # print(f"  KL(pred||target): {kl:.4f}")

                    #entropy regularization to enforce low student entropy
                    if(self.entropy_regularization>0):
                        results_dict[f"loss_entropy_regularization{i}_{j}"]=self.entropy_regularization*pred_entropy
            
        if self.training:
            self.step_count += 1
            if(self.temperature_warmup_steps>0):
                self.temperature = cosine_warmup(self.temperature_warmup_start, self.temperature_warmup_end, self.step_count, self.temperature_warmup_steps)

        return results_dict

@dataclass
class MaskedPredictionLossSinkhornKnoppKMeansConfig(MaskedPredictionLossConfig):
    _target_:str = "clinical_ts.loss.selfsupervised.MaskedPredictionLossSinkhornKnoppKMeans"
    loss_type:str = "masked_pred_skkmeans"
    entropy_regularization:float = 0.001 #prefactor of entropy regularization loss