import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class GasDataset(Dataset):
    def __init__(self, dataframe, batch_id=None, labels_override=None, indices_filter=None):
        """
        Args:
            dataframe: The full data.
            batch_id: Filter by batch.
            labels_override (Tensor/Array): If provided, replaces the ground truth labels.
                                          Used for turning Target Data into Pseudo-Source Data.
            indices_filter (Array): If provided, only keeps these specific indices.
                                  Used to select only High-Confidence samples.
        """
        if batch_id is not None:
            # We copy to avoid settingwithcopy warnings on the main df
            self.data = dataframe[dataframe['Batch_ID'] == batch_id].reset_index(drop=True)
        else:
            self.data = dataframe
            
        # Optional: Filter specific samples (e.g., only high confidence ones)
        if indices_filter is not None:
            self.data = self.data.iloc[indices_filter].reset_index(drop=True)
            
        # 1. Features
        feat_cols = [c for c in self.data.columns if 'feat_' in c]
        self.features = self.data[feat_cols].values.astype(np.float32)
        
        # 2. Labels
        if labels_override is not None:
            # Use the Pseudo-Labels provided by the model
            if indices_filter is not None:
                # Ensure labels match the filtered data
                self.labels = np.array(labels_override)[indices_filter].astype(np.int64)
            else:
                self.labels = np.array(labels_override).astype(np.int64)
        else:
            # Use Ground Truth
            labels = self.data['Gas_Class'].values.astype(np.int64)
            if labels.min() == 1: labels = labels - 1
            self.labels = labels
        
        # 3. Concentrations (for Physics)
        if 'Concentration' in self.data.columns:
            self.concs = self.data['Concentration'].values.astype(np.float32)
        else:
            self.concs = np.zeros(len(self.data), dtype=np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.concs[idx]