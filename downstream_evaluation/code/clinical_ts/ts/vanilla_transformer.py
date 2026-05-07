__all__ = ['VanillaTransformerPredictor', 'VanillaTransformerPredictorConfig']

from dataclasses import dataclass
from typing import Optional

from .transformer_modules.vanilla_transformer import TransformerModel
from ..template_modules import PredictorBase, PredictorBaseConfig


class VanillaTransformerPredictor(PredictorBase):
    """Transformer-based predictor for TimeSeriesEncoder"""
    
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)

        d_ff = hparams_predictor.d_ff
        if d_ff is None:
            d_ff = 4 * hparams_predictor.model_dim
        
        self.predictor = TransformerModel(
            d_input=hparams_input_shape.channels if hparams_input_shape.channels != hparams_predictor.model_dim else None,
            d_output=None,
            d_model=hparams_predictor.model_dim,
            n_heads=hparams_predictor.n_heads,
            n_layers=hparams_predictor.layers,
            d_ff=d_ff,
            dropout=hparams_predictor.dropout,
            prenorm=hparams_predictor.prenorm,
            max_len=hparams_input_shape.length,
            causal=hparams_predictor.causal,
            pos_encoding=hparams_predictor.pos_encoding,
            activation=hparams_predictor.activation,
            pooling=False,  # Never pool for predictor
        )
    
    def forward(self, **kwargs):
        """
        Args:
            kwargs['seq']: (B, L, C) input tensor
        Returns:
            dict with 'seq': (B, L, model_dim)
        """
        return {"seq": self.predictor(kwargs["seq"])}


@dataclass
class VanillaTransformerPredictorConfig(PredictorBaseConfig):
    _target_: str = "clinical_ts.ts.vanilla_transformer.VanillaTransformerPredictor"
    
    # Model dimensions
    model_dim: int = 512          # Hidden dimension (d_model)
    n_heads: int = 8              # Number of attention heads
    layers: int = 6               # Number of transformer blocks
    d_ff: Optional[int] = None              # Feed-forward dimension (default: 4 * model_dim)
    
    # Regularization
    dropout: float = 0.1          # Dropout rate
    
    # Architecture choices
    prenorm: bool = True          # Pre-norm (True) or post-norm (False)
    causal: bool = False          # Causal attention for autoregressive modeling
    pos_encoding: str = 'rope'  # 'sinusoidal' or 'rope'
    activation: str = 'gelu'      # 'gelu', 'relu', or 'silu'