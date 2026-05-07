__all__ = [
    'EMATimeSeriesEncoderData2Vec',
    'EMATimeSeriesEncoderDinoSR',
    'EMATimeSeriesEncoderJEPA',
    'EMATimeSeriesEncoderSinkhornKnoppKMeans',
    'EMATimeSeriesEncoderConfigData2Vec',
    'EMATimeSeriesEncoderConfigDinoSR',
    'EMATimeSeriesEncoderConfigJEPA',
    'EMATimeSeriesEncoderConfigSinkhornKnoppKMeans'
]

from dataclasses import dataclass, field
from operator import attrgetter
from typing import List, Union

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..template_modules import TimeSeriesEncoder, EMATimeSeriesEncoderBaseConfig
from ..utils.callbacks import ForwardHook
from ..utils.schedulers import cosine_warmup

###############################################################################################
# Sinkhorn-Knopp algorithm for optimal transport
# adapted from CAPI only clustering from model.py in https://github.com/facebookresearch/capi/
###############################################################################################
exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}

def stable_exp(M: torch.Tensor) -> torch.Tensor:
    """
    Compute numerically stable exponential.
    
    Args:
        M: Input tensor
    Returns:
        Stabilized exponential of M
    """
    if M.numel() == 0:
        raise ValueError("Input tensor is empty")
    if torch.isnan(M).any() or torch.isinf(M).any():
        raise ValueError("Input tensor contains NaN or Inf values")
    
    # Max-shift for numerical stability (per sample)
    shift = M.max(dim=-1, keepdim=True).values
    
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)
    
    # Simple max-shift without exp_max_values
    M_shifted = M - shift 
    return M_shifted.exp()


def reduced_sum(*args, **kwargs):
    """Sum with distributed reduction if initialized."""
    summed = torch.sum(*args, **kwargs)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(summed)
    return summed


@torch.no_grad()
def sinkhorn_knopp(M, num_iters=3, eps=1e-8):
    """
    Sinkhorn-Knopp algorithm for optimal transport.
    Numerically stable with distributed training support.
    
    Args:
        M: [B, K] logit matrix (before softmax)
        num_iters: number of iterations
        eps: epsilon to prevent division by zero
    
    Returns:
        Q: [B, K] doubly stochastic assignment matrix
    """
    M = stable_exp(M)
    
    for _ in range(num_iters):
        M /= reduced_sum(M, dim=0, keepdim=True) + eps  # normalize columns
        M /= torch.sum(M, dim=1, keepdim=True) + eps     # normalize rows
    
    return M


##########################################################################################################
# OnlineClusteringDinoSR
##########################################################################################################
class OnlineClusteringDinoSR(nn.Module):
    def __init__(self, n_clusters, dim, momentum=0.9):
        super().__init__()
        self.n_clusters = n_clusters
        self.dim = dim
        self.momentum = momentum

        codebook = torch.randn(n_clusters, dim)
        codebook = F.instance_norm(codebook.unsqueeze(0)).squeeze(0)
        
        self.register_buffer('codebook', codebook)
        self.register_buffer('code_sum', codebook.clone())
        self.register_buffer('code_cnt', torch.ones(n_clusters))
        
    @torch.no_grad()
    def forward(self, features):
        features_norm = F.instance_norm(features.unsqueeze(0)).squeeze(0)
        distances = torch.cdist(features_norm, self.codebook)
        assignments = torch.argmin(distances, dim=1)
        
        return assignments
    
    @torch.no_grad()
    def update(self, features, assignments):
        features_norm = F.instance_norm(features.unsqueeze(0)).squeeze(0)
        
        for v in range(self.n_clusters):
            mask = (assignments == v)
            
            if mask.any():
                assigned_features = features_norm[mask]
                
                self.code_sum[v] = (
                    self.momentum * self.code_sum[v] + 
                    (1 - self.momentum) * assigned_features.sum(dim=0)
                )
                self.code_cnt[v] = (
                    self.momentum * self.code_cnt[v] + 
                    (1 - self.momentum) * mask.sum()
                )
            else:
                self.code_sum[v] = self.momentum * self.code_sum[v]
                self.code_cnt[v] = self.momentum * self.code_cnt[v]

        self.codebook = self.code_sum / self.code_cnt.unsqueeze(1).clamp(min=1e-8)


##########################################################################################################
# OnlineSinkhornKnoppKMeans
##########################################################################################################

class OnlineSinkhornKnoppKMeans(nn.Module):
    def __init__(
        self,
        n_clusters,
        dim=None,
        temperature=0.1,
        temperature_warmup_steps=2000,
        temperature_warmup_start=0.0001,
        sinkhorn_iters=3,
        sinkhorn_epsilon=1e-8,
        prototype_momentum=0.99,
        freeze_prototypes_steps=0,
        device=None,
        kmeanspp_init= True
    ):
        """
        Initialize online Sinkhorn-Knopp based clustering. If dim/device are not
        provided, they will be inferred from the first batch via initialize().

        Args:
            n_clusters (int): Number of clusters/prototypes
            dim (int, optional): Feature dimension (inferred on first batch if None)
            temperature (float): Temperature for logits scaling
            temperature_warmup_steps (int): Number of warmup steps for the temperature (0: no warmup)
            temperature_warmup_start (float): Starting value for temperature warmup
            sinkhorn_iters (int): Number of Sinkhorn-Knopp iterations
            sinkhorn_epsilon (float): Epsilon for Sinkhorn-Knopp stability
            prototype_momentum (float): Momentum for prototype EMA update
            freeze_prototypes_steps (int): Number of steps to freeze prototypes before updating
            device: Device to initialize prototypes on (inferred from batch if None)
            kmeanspp_init: kmeans++ init instead of random init
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.dim = dim
        self.temperature = temperature_warmup_start if temperature_warmup_steps>0 else temperature
        self.temperature_warmup_steps = temperature_warmup_steps
        self.temperature_warmup_start = temperature_warmup_start
        self.temperature_warmup_end = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.prototype_momentum = prototype_momentum
        self.freeze_prototypes_steps = freeze_prototypes_steps
        self.step_count = 0
        self.device = device
        self.kmeanspp_init = kmeanspp_init
        self.prototypes = None

    def initialize(self, batch):
        """Initialize prototypes from the first batch, mirroring OnlineKMeans."""
        self.dim = batch.shape[1]
        self.device = batch.device
        
        if(self.kmeanspp_init):    
            # k-means++ initialization
            batch_size = batch.shape[0]
            n_clusters = min(self.n_clusters, batch_size)
            
            # First prototype random
            idx = torch.randint(0, batch_size, (1,))
            prototypes = [batch[idx]]
            
            # Subsequent prototypes
            for _ in range(1, n_clusters):
                # Compute distances to nearest prototype
                all_protos = torch.cat(prototypes, dim=0)
                distances = torch.cdist(batch, all_protos, p=2)
                min_distances = distances.min(dim=1)[0]
                
                # Sample with probability proportional to distance squared
                probs = min_distances ** 2
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1)
                prototypes.append(batch[idx])
            
            self.prototypes = torch.cat(prototypes, dim=0)
            
            # If we need more clusters than batch size, pad with noise
            if n_clusters < self.n_clusters:
                padding = torch.randn(self.n_clusters - n_clusters, self.dim).to(self.device)
                padding = F.normalize(padding, dim=1)
                self.prototypes = torch.cat([self.prototypes, padding], dim=0)
            
            self.prototypes = F.normalize(self.prototypes, dim=1)
        else: # random init
            self.prototypes = torch.randn(self.n_clusters, self.dim).to(self.device)
            self.prototypes = F.normalize(self.prototypes, dim=1)

    @torch.no_grad()
    def forward(self, features):
        """
        Get soft targets from features using Sinkhorn-Knopp.
        
        Args:
            features: [B, D] normalized features
        
        Returns:
            assignments: [B, K] soft assignment probabilities
        """
        if self.prototypes is None:
            self.initialize(features)

        # Compute similarity to prototypes
        logits = features @ self.prototypes.T / self.temperature  # [B, K]
        
        # Apply Sinkhorn-Knopp for balanced soft assignments
        assignments = sinkhorn_knopp(
            logits, 
            num_iters=self.sinkhorn_iters,
            eps=self.sinkhorn_epsilon
        )
        
        return assignments

    @torch.no_grad()
    def update(self, features, assignments=None):
        """
        Update prototypes using soft assignments from Sinkhorn-Knopp.
        
        Args:
            features: [B, D] unnormalized features
            assignments: [B, K] soft assignment matrix from Sinkhorn-Knopp (optional, will compute if None)
        """
        if self.prototypes is None:
            self.initialize(features)

        if self.step_count < self.freeze_prototypes_steps or not self.training:
            self.step_count += 1
            return
        
        features_norm = F.normalize(features, dim=1)
        
        # Compute assignments if not provided
        if assignments is None:
            assignments = self.forward(features_norm)
        
        # Compute new prototype positions as weighted average of features
        new_prototypes = assignments.T @ features  # [K, D]
        new_prototypes = F.normalize(new_prototypes, dim=1)
        
        # EMA update of prototypes (avoid .data mutation)
        self.prototypes = self.prototype_momentum * self.prototypes + \
                  (1 - self.prototype_momentum) * new_prototypes
        self.prototypes = F.normalize(self.prototypes, dim=1)
        
        self.step_count += 1
        #if(self.temperature_warmup_steps>0):
        self.temperature = cosine_warmup(self.temperature_warmup_start, self.temperature_warmup_end, self.step_count, self.temperature_warmup_steps)



class EMATimeSeriesEncoderBase(TimeSeriesEncoder):
    '''Base class for EMA time series encoder with generic feature extraction'''
    def __init__(self, hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=None, has_ema=False):
        super().__init__(hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=target_dim, has_ema=has_ema)
        self.momentum = hparams_seqenc.ema.momentum
        self.update_ema = hparams_seqenc.ema.update_ema
        self.pretrained = hparams_seqenc.ema.pretrained
        self.update_every_k_batches = hparams_seqenc.ema.update_every_k_batches
        self.transpose_axes_after_hook = hparams_seqenc.ema.transpose_axes_after_hook
        self.batch_count = 0

        # Store hooks (single or multiple) as None initially
        self.hooks = None
        self._setup_hook(hparams_seqenc.ema.module_name)
        
        #disable loss calculation (and other optional modules)
        self.loss = None
        self.head_ssl = None
        self.quantizer = None
        self.masking = None

    def _setup_hook(self, module_name):
        """Setup forward hook on specified modules. module_name is assumed to be a list."""
        self._cleanup_hook()

        # def resolve(name):
        #     if "[" in name:
        #         n = name.split('[')[0]
        #         idx = int(name.split('[')[1][:-1])
        #         getter = attrgetter(n)
        #         return getter(self)[idx]
        #     getter = attrgetter(name)
        #     return getter(self)        

        def resolve(name):
            parts = name.split('.')
            obj = self
            for part in parts:
                if '[' in part:
                    attr = part.split('[')[0]
                    idx = int(part.split('[')[1][:-1])
                    obj = getattr(obj, attr)[idx]
                else:
                    obj = getattr(obj, part)
            return obj
        
        self.hooks = [ForwardHook(resolve(n), store_output=True) for n in module_name]
        self.num_layers = len(self.hooks)

    def _cleanup_hook(self):
        """Cleanup existing hook(s) if present"""
        if self.hooks is None:
            return
        if isinstance(self.hooks, list):
            for h in self.hooks:
                h.remove()
        else:
            self.hooks.remove()
        self.hooks = None

    def __del__(self):
        """Ensure hook is cleaned up when object is deleted"""
        self._cleanup_hook()

    def initialize_params(self, ts_encoder):
        '''supposed to be called before fit_start'''
        with torch.no_grad():
            ema_state_dict = self.state_dict()

            if(self.pretrained==""):#use weights from parent module
                other_state_dict = ts_encoder.state_dict()
            else:
                checkpoint = torch.load(self.pretrained, map_location=lambda storage, loc: storage,)
                other_state_dict = checkpoint["state_dict"]
            
            for key in ema_state_dict.keys():
                other_key = key[len("ema."):]
                if(other_key in other_state_dict.keys()):
                    ema_state_dict[key].copy_(other_state_dict[other_key])

            self.load_state_dict(ema_state_dict)

    def update_params(self, ts_encoder):
        '''supposed to be called after every batch'''
        if(self.update_ema):
            self.batch_count = self.batch_count +1
            if(self.batch_count % self.update_every_k_batches ==0):
                with torch.no_grad():
                    ema_state_dict = self.state_dict()
                    other_state_dict = ts_encoder.state_dict()

                    for key in ema_state_dict.keys():
                        other_key = key[len("ema."):]#strip ema. from key name
                        if(other_key in other_state_dict.keys()):
                            ema_state_dict[key].copy_(self.momentum * ema_state_dict[key] + (1 - self.momentum) * other_state_dict[other_key])

                    self.load_state_dict(ema_state_dict)
                
    
    def _extract_features(self, **kwargs):
        """
        Generic feature extraction logic shared by all derived classes.
        Extracts features from the forward hook and prepares them for clustering.
        
        Returns:
            tuple: (features, N) where features is [B*N, dim] and N is sequence length
        """
        #just run the model to populate the forward hook
        with torch.no_grad():
            self.forward(**kwargs)

        def process(feat, transpose_axes):
            if isinstance(feat, dict):
                feat = feat["seq"]
            if isinstance(feat, tuple):  # some layers like lstms, s4 etc might return tuples
                feat = feat[0]  # just pick the first
            if len(transpose_axes) > 0:
                feat = feat.permute(*tuple(transpose_axes))
            N_local = feat.shape[1]
            feat = feat.reshape(-1, feat.shape[2])
            return feat, N_local

        # grab features from the hook(s)
        if isinstance(self.hooks, list):
            if len(self.transpose_axes_after_hook) > 0:
                processed = [process(h.stored, self.transpose_axes_after_hook[i]) for i, h in enumerate(self.hooks)]
            else:
                processed = [process(h.stored, []) for h in self.hooks]
            features_list = [p[0] for p in processed]
            N_list = [p[1] for p in processed]
            return features_list, N_list
        else:
            transpose_axes = self.transpose_axes_after_hook[0] if len(self.transpose_axes_after_hook) > 0 else []
            features, N = process(self.hooks.stored, transpose_axes)
            return [features], [N]

    def _cluster_features(self, features_list, N_list):
        """Default multi-layer handler: return per-layer features without clustering."""
        results = []
        for feat, N in zip(features_list, N_list):
            B = feat.shape[0] // N
            results.append(feat.view(B, N, feat.shape[1]))
        return {"ema_features": results}

    def forward_features(self, **kwargs):
        """
        Main forward_features method that combines generic extraction with clustering.
        Base class forwards features without clustering by default.
        """
        features, N = self._extract_features(**kwargs)
        return self._cluster_features(features, N)

@dataclass
class EMATimeSeriesEncoderConfigIdentity(EMATimeSeriesEncoderBaseConfig):
    _target_:str = "clinical_ts.ts.ema.EMATimeSeriesEncoderBase"



class EMATimeSeriesEncoderData2Vec(EMATimeSeriesEncoderBase):
    '''EMA time series encoder for Data2Vec framework

        Data2Vec framework details -
        Paper: https://arxiv.org/abs/2202.03555
        Code: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec/models
        Year: 2022
        By: Meta AI
        Clustering: No (Regression)
        Loss: Smooth L1 loss    
    '''

    def __init__(
            self,
            hparams_seqenc,
            hparams_input_shape,
            static_stats_train,
            target_dim=None,
            has_ema=False
        ):
        self.average_top_k_layers = hparams_seqenc.ema.average_top_k_layers
        self.layer_norm_targets = hparams_seqenc.ema.layer_norm_targets
        self.instance_norm_targets = hparams_seqenc.ema.instance_norm_targets
        
        super().__init__(
            hparams_seqenc,
            hparams_input_shape,
            static_stats_train,
            target_dim=target_dim,
            has_ema=has_ema
        )

    def _cluster_features(self, features_list, N_list):
        # Take top-k layers
        if len(features_list) > self.average_top_k_layers:
            features_list = features_list[-self.average_top_k_layers:]
            N_list = N_list[-self.average_top_k_layers:]
        
        # Reshape and normalize
        layer_outputs = []
        for features, N in zip(features_list, N_list):
            B = features.shape[0] // N
            features = features.view(B, N, features.shape[1])
            
            if self.instance_norm_targets:
                features = F.instance_norm(rearrange(features, "b l d -> b d l"))
                features = rearrange(features, "b d l -> b l d")
            if self.layer_norm_targets:
                features = F.layer_norm(features, features.shape[-1:])
            
            layer_outputs.append(features)
        
        # Average across layers
        targets = torch.stack(layer_outputs).mean(dim=0)
        
        return {"ema_features": targets}

@dataclass
class EMATimeSeriesEncoderConfigData2Vec(EMATimeSeriesEncoderBaseConfig):
    _target_: str = "clinical_ts.ts.ema.EMATimeSeriesEncoderData2Vec"
    
    average_top_k_layers: int = 8
    layer_norm_targets: bool = True
    instance_norm_targets: bool = False


class EMATimeSeriesEncoderDinoSR(EMATimeSeriesEncoderBase):
    def __init__(
        self,
        hparams_seqenc,
        hparams_input_shape,
        static_stats_train,
        target_dim=None,
        has_ema=False
    ):
        self.average_top_k_layers = hparams_seqenc.ema.average_top_k_layers
        self.codebook_size = hparams_seqenc.ema.codebook_size
        self.codebook_momentum = hparams_seqenc.ema.codebook_momentum
        
        self.instance_norm_target_layer = hparams_seqenc.ema.instance_norm_target_layer
        self.layer_norm_targets = hparams_seqenc.ema.layer_norm_targets
        
        super().__init__(
            hparams_seqenc,
            hparams_input_shape,
            static_stats_train,
            target_dim=target_dim,
            has_ema=has_ema
        )
        
        feat_dim = hparams_seqenc.ema.target_dim if hparams_seqenc.ema.target_dim > 0 else hparams_seqenc.pred.model_dim
        
        self.clustering = nn.ModuleList([
            OnlineClusteringDinoSR(
                n_clusters=self.codebook_size,
                dim=feat_dim,
                momentum=self.codebook_momentum
            ) for _ in range(self.average_top_k_layers)
        ])
    
    def _cluster_features(self, features_list, N_list):
        if len(features_list) > self.average_top_k_layers:
            features_list = features_list[-self.average_top_k_layers:]
            N_list = N_list[-self.average_top_k_layers:]
        
        all_assignments = []
        
        for features, N, clustering in zip(features_list, N_list, self.clustering):
            B = features.shape[0] // N
            
            features_3d = features.view(B, N, -1)
            
            if self.instance_norm_target_layer:
                features_3d = F.instance_norm(rearrange(features_3d, "b n d -> b d n"))
                features_3d = rearrange(features_3d, "b d n -> b n d")
            
            if self.layer_norm_targets:
                features_3d = F.layer_norm(features_3d, (features_3d.shape[-1],))
            
            features = features_3d.view(-1, features_3d.shape[-1])
            
            assignments = clustering(features)
            if self.training:
                clustering.update(features, assignments)
            
            assignments = assignments.view(B, N)
            all_assignments.append([assignments])
        
        return {
            "ema_cluster_assignments": all_assignments
        }

@dataclass
class EMATimeSeriesEncoderConfigDinoSR(EMATimeSeriesEncoderBaseConfig):
    _target_: str = "clinical_ts.ts.ema.EMATimeSeriesEncoderDinoSR"
    average_top_k_layers: int = 4
    codebook_size: int = 256
    codebook_momentum: float = 0.9
    instance_norm_target_layer: bool = False
    layer_norm_targets: bool = True


class EMATimeSeriesEncoderJEPA(EMATimeSeriesEncoderBase):
    """
    EMA time series encoder for JEPA.
    
    JEPA framework details:
    - Paper: https://arxiv.org/abs/2301.08243
    - Code: https://github.com/facebookresearch/ijepa
    - Year: 2023
    - By: Meta AI (Yann LeCun's group)
    - Clustering: No (Direct feature prediction)
    - Loss: SmoothL1Loss (default) or MSE
    """
    
    def __init__(
        self,
        hparams_seqenc,
        hparams_input_shape,
        static_stats_train,
        target_dim=None,
        has_ema=False
    ):
        self.average_top_k_layers = hparams_seqenc.ema.average_top_k_layers
        self.layer_norm_targets = hparams_seqenc.ema.layer_norm_targets
        self.instance_norm_targets = hparams_seqenc.ema.instance_norm_targets
        self.normalize_targets = hparams_seqenc.ema.normalize_targets
        
        super().__init__(
            hparams_seqenc,
            hparams_input_shape,
            static_stats_train,
            target_dim=target_dim,
            has_ema=has_ema
        )
    
    def _cluster_features(self, features_list, N_list):
        if len(features_list) > self.average_top_k_layers:
            features_list = features_list[-self.average_top_k_layers:]
            N_list = N_list[-self.average_top_k_layers:]
        
        layer_outputs = []
        for features, N in zip(features_list, N_list):
            B = features.shape[0] // N
            features = features.view(B, N, features.shape[1])
            
            if self.instance_norm_targets:
                features = F.instance_norm(rearrange(features, "b l d -> b d l"))
                features = rearrange(features, "b d l -> b l d")
            
            if self.layer_norm_targets:
                features = F.layer_norm(features, features.shape[-1:])
            
            layer_outputs.append(features)
        
        if len(layer_outputs) > 1:
            targets = torch.stack(layer_outputs).mean(dim=0)
        else:
            targets = layer_outputs[0]
        
        if self.normalize_targets:
            targets = F.layer_norm(targets, (targets.size(-1),))
        
        return {"ema_features": targets}


@dataclass
class EMATimeSeriesEncoderConfigJEPA(EMATimeSeriesEncoderBaseConfig):
    _target_: str = "clinical_ts.ts.ema.EMATimeSeriesEncoderJEPA"
    average_top_k_layers: int = 1
    layer_norm_targets: bool = False
    instance_norm_targets: bool = False
    normalize_targets: bool = True


class EMATimeSeriesEncoderSinkhornKnoppKMeans(EMATimeSeriesEncoderBase):
    '''EMA time series encoder with Sinkhorn-Knopp k-means style clustering'''
    def __init__(self, hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=None, has_ema=False):
        super().__init__(hparams_seqenc, hparams_input_shape, static_stats_train, target_dim=target_dim, has_ema=has_ema)

        # Use optional target_dim override or predictor model_dim
        feat_dim = hparams_seqenc.ema.target_dim if hparams_seqenc.ema.target_dim > 0 else hparams_seqenc.pred.model_dim

        self.clustering = nn.ModuleList([
            nn.ModuleList([
                OnlineSinkhornKnoppKMeans(
                    n_clusters=k,
                    dim=feat_dim,
                    temperature=hparams_seqenc.ema.temperature,
                    temperature_warmup_steps = hparams_seqenc.ema.temperature_warmup_steps,
                    temperature_warmup_start = hparams_seqenc.ema.temperature_warmup_start,
                    sinkhorn_iters=hparams_seqenc.ema.sinkhorn_iters,
                    sinkhorn_epsilon=hparams_seqenc.ema.sinkhorn_epsilon,
                    prototype_momentum=hparams_seqenc.ema.prototype_momentum,
                    freeze_prototypes_steps=hparams_seqenc.ema.freeze_prototypes_steps,
                    device=None  # infer from first batch
                )
                for k in hparams_seqenc.loss.kmeans_ks
            ]) for _ in range(self.num_layers)
        ])

    def _cluster_features(self, features_list, N_list):
        """Sinkhorn-based clustering for single or multiple layers."""
        all_assignments = []
        all_prototypes = []
        res ={}

        for i, (feats, N, layer_cluster) in enumerate(zip(features_list, N_list, self.clustering)):
            feats_norm = F.normalize(feats, dim=-1)
            cluster_assignments = []
            cluster_prototypes = []
            for j,c in enumerate(layer_cluster):
                ca = c.forward(feats_norm)
                if self.training:
                    c.update(feats_norm, assignments=ca)               
                cluster_assignments.append(ca.view(-1, N, ca.shape[-1]))  # revive seq axis
                cluster_prototypes.append(c.prototypes)
                res[f"metric_teacher_temperature{i}_{j}"]=c.temperature #debugging output
            all_assignments.append(cluster_assignments)
            all_prototypes.append(cluster_prototypes)
        res["ema_soft_cluster_assignments"] = all_assignments
        res["ema_cluster_prototypes"] = all_prototypes
        return res

@dataclass
class EMATimeSeriesEncoderConfigSinkhornKnoppKMeans(EMATimeSeriesEncoderBaseConfig):
    _target_:str = "clinical_ts.ts.ema.EMATimeSeriesEncoderSinkhornKnoppKMeans"

    temperature: float = 0.07
    temperature_warmup_steps: int = 200
    temperature_warmup_start: float = 0.15 #was 0.04
    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 1e-8
    prototype_momentum: float = 0.99
    freeze_prototypes_steps: int = 0