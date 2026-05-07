from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
import torch.nn as nn

from clinical_ts.utils.heads import LearnableQueryAttentionPoolingHead, LearnableQueryAttentionPoolingHeadConfig

from main_lite_base import FMWrapperBase

from clinical_ts.models.ecg_foundation_models.ecg_founder import Net1D
from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa import ecg_jepa
from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa_utils import load_encoder

from clinical_ts.models.ecg_foundation_models.ecg_cpc.basic_io import load_model_from_config


class ECGFounderWrapper(FMWrapperBase):
    """
        Paper: https://arxiv.org/abs/2410.04133
        Code: https://github.com/PKUDigitalHealth/ECGFounder
        Checkpoints: https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main
        Model sampling frequency: 500 Hz
        Pretraining dataset: HEEDB
    """
    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        self.model = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=num_classes
        )

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            state_dict = {k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("dense.")}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("ECGFounderWrapper requires a valid `pretrained_path` to load the encoder.")


        self.feature_dim = self.model.dense.in_features
        self.model.dense = nn.Identity()

        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()        
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        self._override_forward()

    def _override_forward(self):
        def custom_forward(x):
            out = x
            # First conv
            out = self.model.first_conv(out)
            if self.model.use_bn:
                out = self.model.first_bn(out)
            out = self.model.first_activation(out)

            # Stages
            for i_stage in range(self.model.n_stages):
                net = self.model.stage_list[i_stage]
                out = net(out)

            sequence_features = out  # (batch, channels, seq_len)
            pooled_features = out.mean(-1)  # GAP
            return sequence_features, pooled_features

        self.model.forward = custom_forward
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]    
        
        encoder_params = []
        predictor_params = []
        first_conv_bn_params = []
        
        for name, param in self.model.named_parameters():
            if name.startswith(("first_conv", "first_bn")):
                first_conv_bn_params.append(param)
            elif any(name.startswith(f"stage_list.{i}.") for i in [0, 1, 2]):
                encoder_params.append(param)
            elif any(name.startswith(f"stage_list.{i}.") for i in [3, 4, 5, 6]):
                predictor_params.append(param)
        
        encoder_params.extend(first_conv_bn_params)

        print("Returning layer dependent learning rate from ECGFounderWrapper...")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]

    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        sequence_features, pooled_features = self.model(x)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            seq_feats = sequence_features.transpose(1, 2)  # (batch, seq_len, channels)
            output_dict = self.head(seq=seq_feats)
            x = output_dict["seq"]  
        else:
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class ECGJEPAWrapper(FMWrapperBase):
    """
        Paper: https://arxiv.org/abs/2410.08559
        Code: https://github.com/sehunfromdaegu/ECG_JEPA
        Checkpoints: https://drive.google.com/file/d/1mh-XL0XOvvhFbhvuZ9c2KnTHa9B4F3Wx/view
                     https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view
        Model sampling frequency: 250 Hz
        Pretraining dataset: ptb-xl, cpsc2018
        Note: 2 checkpoints available; one for random masking and other for multi-block masking
    """
    def __init__(self, num_classes, num_output_tokens, pretrained_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)

        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]
        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor

        if pretrained_path:
            self.encoder, self.feature_dim = load_encoder(pretrained_path)
        else:
            raise ValueError("ECGJepaWrapper requires a valid `pretrained_path` to load the encoder.")
        
        # Nonlinear head configurations
        nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
            multi_prediction=False,
            heads=16,
            bias=False
        )
        
        @dataclass
        class InputShape:
            channels: int
            length: int
            static_dim: int

        input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
        
        self.nonlinear_head = LearnableQueryAttentionPoolingHead(
            hparams_head=nonlinear_head_config,
            hparams_input_shape=input_shape,
            target_dim=num_classes
        )
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode == "finetuning_nonlinear":
            self.head = self.nonlinear_head
        elif self.eval_mode == "frozen":
            self.head = self.nonlinear_head
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        
        self._override_representation()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_params = []
        predictor_params = []
        
        for name, param in self.encoder.named_parameters():
            if name in ["pos_embed", "W_P.weight", "W_P.bias"]:
                encoder_params.append(param)
            elif any(name.startswith(f"encoder_blocks.blocks.{i}.") for i in range(3)):
                encoder_params.append(param)
            elif any(name.startswith(f"encoder_blocks.blocks.{i}.") for i in range(3, 12)):
                predictor_params.append(param)
            elif name.startswith(("norm.weight", "norm.bias")):
                predictor_params.append(param)
        
        print("Returning layer dependent learning rate from ECGJepaWrapper (Multiblock)...")

        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]

    def _override_representation(self):        
        def custom_representation(x):
            assert x.dim() == 3, f"Input should be of dimension 3, x.dim()={x.dim()}"
            assert x.shape[1] == len(self.encoder.leads), f"lead error"
            assert x.shape[2] == 2500, f"Input should be of shape (bs, c, 2500), x.shape[2]={x.shape[2]}"

            pos_embed = self.encoder.pos_embed
            attention_mask = self.encoder._cross_attention_mask().to(x.device) # (c*p, c*p)

            # restric leads
            if len(self.encoder.leads) < self.encoder.c:
                pos_embed = self.encoder.restrict_leads(pos_embed, type="vector")
                attention_mask = self.encoder.restrict_leads(attention_mask, type="matrix")

            bs, l, _ = x.shape
            x = x.reshape(bs, -1, 50)  # (bs,l,2500) -> (bs,l*p,50)
            x = self.encoder.W_P(x)  # (bs,l*p,50) -> (bs,l*p,embed_dim)

            x = self.encoder.encoder_blocks(x, pos_embed, attention_mask)
            
            if self.encoder.norm is not None:
                sequence_features = self.encoder.norm(x)

            # GAP
            pooled_features = torch.mean(sequence_features, dim=1) # (bs,l*50,embed_dim) -> (bs,embed_dim)
            
            return sequence_features, pooled_features
        
        self.encoder.representation = custom_representation
    
    def forward(self, x, **kwargs):
        # ECG-JEPA takes exactly 2500 time steps. 
        # ECG-JEPA uses 8-channels: I, II, V1, V2, V3, V4, V5, V6
        # Our dataset uses 12-channels: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        selected_indices = [0, 1, 6, 7, 8, 9, 10, 11]
        x = torch.nan_to_num(x)
        x = x[:, selected_indices, :]

        sequence_features, pooled_features = self.encoder.representation(x)

        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]
        else:
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class Data2VecWrapper(FMWrapperBase):
    def __init__(
            self,
            num_classes,
            num_output_tokens,
            config_path=None,
            eval_mode="finetuning_linear",
            lr=1e-3,
            discriminative_lr_factor=0.1
        ):
        
        super().__init__(num_classes, num_output_tokens)
        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        self.model, self.config = load_model_from_config(
            config_name=config_path         
        )
        
        self.feature_dim = 512
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode in ["finetuning_nonlinear", "frozen"]:
            nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
                multi_prediction=False,
                heads=16,
                bias=False
            )
            
            @dataclass
            class InputShape:
                channels: int
                length: int
                static_dim: int

            input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
            
            self.head = LearnableQueryAttentionPoolingHead(
                hparams_head=nonlinear_head_config,
                hparams_input_shape=input_shape,
                target_dim=num_classes
            )
            
            if self.eval_mode == "frozen":
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_module, predictor_module, _ = self.model.ts_encoder.get_modules()        
    
        encoder_params = list(chain(*[e.parameters() for e in encoder_module]))
        predictor_params = list(chain(*[p.parameters() for p in predictor_module]))

        print("\n\nReturning layer dependent learning rate from Data2Vec Wrapper ...\n\n")

        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        output = self.model(seq=x)        
        sequence_features = output["seq"]
        
        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]  
        else:
            pooled_features = sequence_features.mean(dim=1)
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class DinoSRWrapper(FMWrapperBase):
    def __init__(
            self,
            num_classes,
            num_output_tokens,
            config_path=None,
            eval_mode="finetuning_linear",
            lr=1e-3,
            discriminative_lr_factor=0.1
        ):
        
        super().__init__(num_classes, num_output_tokens)
        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        self.model, self.config = load_model_from_config(
            config_name=config_path         
        )
        
        self.feature_dim = 512
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode in ["finetuning_nonlinear", "frozen"]:
            nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
                multi_prediction=False,
                heads=16,
                bias=False
            )
            
            @dataclass
            class InputShape:
                channels: int
                length: int
                static_dim: int

            input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
            
            self.head = LearnableQueryAttentionPoolingHead(
                hparams_head=nonlinear_head_config,
                hparams_input_shape=input_shape,
                target_dim=num_classes
            )
            
            if self.eval_mode == "frozen":
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_module, predictor_module, _ = self.model.ts_encoder.get_modules()        
    
        encoder_params = list(chain(*[e.parameters() for e in encoder_module]))
        predictor_params = list(chain(*[p.parameters() for p in predictor_module]))

        print("\n\nReturning layer dependent learning rate from DinoSR Wrapper ...\n\n")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        output = self.model(seq=x)        
        sequence_features = output["seq"]
        
        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]  
        else:
            pooled_features = sequence_features.mean(dim=1)
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class JEPAWrapper(FMWrapperBase):
    def __init__(
            self,
            num_classes,
            num_output_tokens,
            config_path=None,
            eval_mode="finetuning_linear",
            lr=1e-3,
            discriminative_lr_factor=0.1
        ):
        
        super().__init__(num_classes, num_output_tokens)
        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        self.model, self.config = load_model_from_config(
            config_name=config_path         
        )
        
        self.feature_dim = 512
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode in ["finetuning_nonlinear", "frozen"]:
            nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
                multi_prediction=False,
                heads=16,
                bias=False
            )
            
            @dataclass
            class InputShape:
                channels: int
                length: int
                static_dim: int

            input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
            
            self.head = LearnableQueryAttentionPoolingHead(
                hparams_head=nonlinear_head_config,
                hparams_input_shape=input_shape,
                target_dim=num_classes
            )
            
            if self.eval_mode == "frozen":
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_module, predictor_module, _ = self.model.ts_encoder.get_modules()        
    
        encoder_params = list(chain(*[e.parameters() for e in encoder_module]))
        predictor_params = list(chain(*[p.parameters() for p in predictor_module]))

        print("\n\nReturning layer dependent learning rate from JEPA Wrapper ...\n\n")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        output = self.model(seq=x)        
        sequence_features = output["seq"]
        
        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]  
        else:
            pooled_features = sequence_features.mean(dim=1)
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class CPCWrapper(FMWrapperBase):
    def __init__(self, num_classes, num_output_tokens, config_path=None, eval_mode="finetuning_linear", lr=1e-3, discriminative_lr_factor=0.1):
        super().__init__(num_classes, num_output_tokens)
        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        self.model, self.config = load_model_from_config(
            config_name=config_path         
        )
        
        self.feature_dim = 512
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode in ["finetuning_nonlinear", "frozen"]:
            nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
                multi_prediction=False,
                heads=16,
                bias=False
            )
            
            @dataclass
            class InputShape:
                channels: int
                length: int
                static_dim: int

            input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
            
            self.head = LearnableQueryAttentionPoolingHead(
                hparams_head=nonlinear_head_config,
                hparams_input_shape=input_shape,
                target_dim=num_classes
            )
            
            if self.eval_mode == "frozen":
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_module, predictor_module, _ = self.model.ts_encoder.get_modules()        
    
        encoder_params = list(chain(*[e.parameters() for e in encoder_module]))
        predictor_params = list(chain(*[p.parameters() for p in predictor_module]))

        print("I am here.")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        output = self.model(seq=x)        
        sequence_features = output["seq"]
        
        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]  
        else:
            pooled_features = sequence_features.mean(dim=1)
            x = self.head(pooled_features)

        return torch.nan_to_num(x)


class HuBERT_PP_Wrapper(FMWrapperBase):
    def __init__(
            self,
            num_classes,
            num_output_tokens,
            config_path=None,
            eval_mode="finetuning_linear",
            lr=1e-3,
            discriminative_lr_factor=0.1
        ):
        
        super().__init__(num_classes, num_output_tokens)
        assert eval_mode in ["finetuning_linear", "finetuning_nonlinear", "frozen", "linear"]

        self.eval_mode = eval_mode
        self.lr = lr
        self.discriminative_lr_factor = discriminative_lr_factor
        
        self.model, self.config = load_model_from_config(
            config_name=config_path         
        )
        
        self.feature_dim = 512
        
        if self.eval_mode == "finetuning_linear":
            self.head = nn.Linear(self.feature_dim, num_classes)
        elif self.eval_mode in ["finetuning_nonlinear", "frozen"]:
            nonlinear_head_config = LearnableQueryAttentionPoolingHeadConfig(
                multi_prediction=False,
                heads=16,
                bias=False
            )
            
            @dataclass
            class InputShape:
                channels: int
                length: int
                static_dim: int

            input_shape = InputShape(channels=self.feature_dim, length=0, static_dim=0)
            
            self.head = LearnableQueryAttentionPoolingHead(
                hparams_head=nonlinear_head_config,
                hparams_input_shape=input_shape,
                target_dim=num_classes
            )
            
            if self.eval_mode == "frozen":
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()      
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
    
    def get_params(self):
        head_params = list(self.head.parameters())

        if self.eval_mode in ["frozen", "linear"]:
            return [{"params": head_params, "lr": self.lr}]
        
        encoder_module, predictor_module, _ = self.model.ts_encoder.get_modules()        
    
        encoder_params = list(chain(*[e.parameters() for e in encoder_module]))
        predictor_params = list(chain(*[p.parameters() for p in predictor_module]))

        print("\n\nReturning layer dependent learning rate from HuBERT++ Wrapper ...\n\n")
        
        return [
            {"params": head_params, "lr": self.lr},
            {"params": predictor_params, "lr": self.lr * self.discriminative_lr_factor},
            {"params": encoder_params, "lr": self.lr * self.discriminative_lr_factor * self.discriminative_lr_factor}
        ]
    
    def forward(self, x, **kwargs):
        x = torch.nan_to_num(x)
        output = self.model(seq=x)        
        sequence_features = output["seq"]
        
        if self.eval_mode in ["frozen", "finetuning_nonlinear"]:
            output_dict = self.head(seq=sequence_features)
            x = output_dict["seq"]  
        else:
            pooled_features = sequence_features.mean(dim=1)
            x = self.head(pooled_features)

        return torch.nan_to_num(x)
