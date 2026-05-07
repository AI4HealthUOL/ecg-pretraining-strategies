from dataclasses import dataclass
from typing import Union, Optional, List, Any
import numpy as np
import torch
import pandas as pd

@dataclass
class StaticData:
    seq: None = None  # Placeholder to match TSData structure
    label: Optional[Union[int, float, np.ndarray, torch.Tensor]] = None
    static: Optional[Union[list, np.ndarray, torch.Tensor]] = None
    static_cat: Optional[Union[list, np.ndarray, torch.Tensor]] = None
    seq_idxs: None = None  # Placeholder to match TSData structure

class StaticDataset(torch.utils.data.Dataset):
    def __init__(self, df, cols_static=None, cols_static_cat=None, col_lbl=None):
        super().__init__()
        
        # At least one of static or static_cat must be provided
        assert cols_static is not None or cols_static_cat is not None, \
            "At least one of cols_static or cols_static_cat must be provided"
        
        def concat_columns(row):
            return [item for col in row for item in (col if isinstance(col, np.ndarray) else [col])]
            
        if cols_static is not None:
            self.static_data = np.vstack(df[cols_static].apply(concat_columns, axis=1).to_list()).astype(np.float32)
        else:
            self.static_data = None
            
        if cols_static_cat is not None:
            self.static_cat_data = np.vstack(df[cols_static_cat].apply(concat_columns, axis=1).to_list())
        else:
            self.static_cat_data = None
            
        # Handle labels
        if col_lbl is None:
            self.labels = None
        else:
            if isinstance(df[col_lbl].iloc[0], (list, np.ndarray)):
                self.labels = np.stack(df[col_lbl])
            else:
                self.labels = np.array(df[col_lbl])
            
    def __len__(self):
        # Use whichever data is available for length
        if self.static_data is not None:
            return len(self.static_data)
        return len(self.static_cat_data)
        
    def __getitem__(self, idx):
        return StaticData(
            label=self.labels[idx] if self.labels is not None else None,
            static=self.static_data[idx] if self.static_data is not None else None,
            static_cat=self.static_cat_data[idx] if self.static_cat_data is not None else None
        )

@dataclass
class StaticDatasetConfig:
    df: pd.DataFrame
    cols_static: Optional[List[str]] = None
    cols_static_cat: Optional[List[str]] = None
    col_lbl: Optional[str] = None
    transforms: Any = None
    allow_multiple_keys: bool = False  # in the df allow multiple rows with identical IDs
