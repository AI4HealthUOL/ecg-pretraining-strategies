__all__ = ['MetricConfig']


from dataclasses import dataclass

import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Filter out the warnings due to not enough positive/negative samples during bootstrapping
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

@dataclass
class MetricConfig:
    _target_:str = ""
    
    name:str = ""#name of the metric e.g. auroc

    aggregation:str = "" #"" means no aggregation across segments of the same sequence, other options: "mean", "max"
    
    key_summary_metric:str = "" #key into the output dict that can serve as summary metric for early stopping etc e.g. (without key_prefix and key_postfix and aggregation type)
    mode_summary_metric:str ="max" #used to determine if key_summary_metric is supposed to be maximized or minimized
    
    verbose:str = "" # comma-separated list of keys to be printed after metric evaluation (without key_prefix and key_postfix and aggregation type)
    
    bootstrap_report_nans:bool = False #report nans during bootstrapping (due to not enough labels of a certain type in certain bootstrap iterations etc)
    bootstrap_iterations:int = 0 #0: no bootstrap
    bootstrap_alpha:float= 0.95 # bootstrap alpha
