__all__ = ['Net1DPredictor', 'Net1DPredictorConfig']

from dataclasses import dataclass, field
from typing import List

from .net1d_modules.net1d import Net1DModule
from ..template_modules import PredictorBase, PredictorBaseConfig


class Net1DPredictor(PredictorBase):
    """Net1D-based predictor for TimeSeriesEncoder"""

    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)

        self.predictor = Net1DModule(
            in_channels=hparams_input_shape.channels,
            base_filters=hparams_predictor.base_filters,
            ratio=hparams_predictor.ratio,
            filter_list=hparams_predictor.filter_list,
            m_blocks_list=hparams_predictor.m_blocks_list,
            kernel_size=hparams_predictor.kernel_size,
            stride=hparams_predictor.stride,
            groups_width=hparams_predictor.groups_width,
            use_bn=hparams_predictor.use_bn,
            use_do=hparams_predictor.use_do,
            verbose=hparams_predictor.verbose,
        )

        # override output channels set by PredictorBase (which uses model_dim)
        self.output_shape.channels = hparams_predictor.filter_list[-1]

    def forward(self, **kwargs):
        """
        Args:
            kwargs['seq']: (B, L, C)
        Returns:
            dict with 'seq': (B, L', filter_list[-1])
        """
        x = kwargs["seq"].transpose(1, 2)       # (B, C, L)
        return {"seq": self.predictor(x)}        # (B, L', filter_list[-1])


@dataclass
class Net1DPredictorConfig(PredictorBaseConfig):
    _target_: str = "clinical_ts.ts.net1d.Net1DPredictor"

    model_dim: int = 512          # kept for PredictorBase compatibility, should match filter_list[-1]

    base_filters: int = 64
    ratio: float = 1.0
    filter_list: List[int] = field(default_factory=lambda: [64, 128, 128, 256, 256, 512, 512])
    m_blocks_list: List[int] = field(default_factory=lambda: [2, 2, 2, 3, 3, 4, 4])
    kernel_size: int = 16
    stride: int = 1
    groups_width: int = 16
    use_bn: bool = False
    use_do: bool = False
    verbose: bool = False