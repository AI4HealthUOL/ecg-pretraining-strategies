import torch.utils.data
import numpy as np
import torch

class CollateDataset(torch.utils.data.Dataset):
    r"""Dataset for collating several existing datasets
    """

    def __init__(self, datasets, sample_idx_from_first_dataset=False) -> None:
        super().__init__()
        self.datasets = datasets
        self.sample_idx_from_first_dataset = sample_idx_from_first_dataset

    def __getitem__(self, idx):
        if(self.sample_idx_from_first_dataset):
            sample_idx = self.datasets[0][idx][0]
            res = tuple(self.datasets[0][idx][1:])
        else:
            sample_idx = idx
            res = tuple(self.datasets[0][idx])
        
        for d in self.datasets[1:]:
            res += tuple(d[sample_idx])
        return res
            
    def __len__(self):
        return len(self.datasets[0])

def tsdata_tuple_collate_fn(batch):
    """Collate function for tuples of TSData objects from multiple datasets.
    The first dataset's fields are not indexed, subsequent datasets are indexed."""
    
    # Initialize lists for each dataset's fields
    num_datasets = len(batch[0])  # Number of datasets being combined
    all_seqs = [[] for _ in range(num_datasets)]
    all_labels = [[] for _ in range(num_datasets)]
    all_statics = [[] for _ in range(num_datasets)]
    all_static_cats = [[] for _ in range(num_datasets)]
    all_seq_idxs = [[] for _ in range(num_datasets)]
    
    # Extract fields for each dataset
    for item in batch:
        for i, tsdata in enumerate(item):
            all_seqs[i].append(tsdata.seq)
            all_labels[i].append(tsdata.label)
            if tsdata.static is not None:
                all_statics[i].append(tsdata.static)
            if tsdata.static_cat is not None:
                all_static_cats[i].append(tsdata.static_cat)
            if tsdata.seq_idxs is not None:
                all_seq_idxs[i].append(tsdata.seq_idxs)
    
    # Process each dataset's fields
    result = {}
    
    # First dataset (no index)
    seq = torch.as_tensor(np.stack(all_seqs[0])) if all_seqs[0] else None
    label = torch.as_tensor(np.stack(all_labels[0])) if all_labels[0] else None
    static = torch.as_tensor(np.stack(all_statics[0])) if all_statics[0] else None
    static_cat = torch.as_tensor(np.stack(all_static_cats[0])) if all_static_cats[0] else None
    seq_idxs = torch.as_tensor(np.stack(all_seq_idxs[0])) if all_seq_idxs[0] else None
    
    result["seq"] = seq
    result["label"] = label
    if static is not None:
        result["static"] = static
    if static_cat is not None:
        result["static_cat"] = static_cat
    if seq_idxs is not None:
        result["seq_idxs"] = seq_idxs
    
    # Additional datasets (with index)
    for i in range(1, num_datasets):
        seq = torch.as_tensor(np.stack(all_seqs[i])) if all_seqs[i] else None
        label = torch.as_tensor(np.stack(all_labels[i])) if all_labels[i] else None
        static = torch.as_tensor(np.stack(all_statics[i])) if all_statics[i] else None
        static_cat = torch.as_tensor(np.stack(all_static_cats[i])) if all_static_cats[i] else None
        seq_idxs = torch.as_tensor(np.stack(all_seq_idxs[i])) if all_seq_idxs[i] else None
        
        result[f"seq_{i}"] = seq
        result[f"label_{i}"] = label
        if static is not None:
            result[f"static_{i}"] = static
        if static_cat is not None:
            result[f"static_cat_{i}"] = static_cat
        if seq_idxs is not None:
            result[f"seq_idxs_{i}"] = seq_idxs
    
    return result