import torch
from torch.utils.data import Dataset
import numpy as np

class CLARO_att(Dataset):
    """
    Dataset class for the CLARO dataset with attention mechanism.
    data: features
    time: time of event
    event: event label
    mask1: mask for the event
    mask2: mask for the time
    reg_features: regular expressions features
    transform: transformation to be applied to the data
    """
    def __init__(self, data, time, event, mask1, mask2, reg_features=None, transform=None):
        self.data  = data
        self.times  = time
        self.events = event
        self.masks1 = mask1
        self.masks2 = mask2
        self.reg_features = reg_features
        self.transform = transform
    
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        time = self.times[idx]
        event = self.events[idx]
        mask1 = self.masks1[idx]
        mask2 = self.masks2[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.reg_features is not None:
            reg_feature = self.reg_features[idx]
            return sample, time, event, mask1, mask2, reg_feature
        
        return sample, time, event, mask1, mask2
    
class CLARO_clinical(Dataset):
    """
    Dataset class for the CLARO dataset.
    data: features
    time: time of event
    event: event label
    mask1: mask for the event
    mask2: mask for the time
    transform: transformation to be applied to the data
    """
    def __init__(self, data, time, event, mask1, mask2, transform=None):
        self.data  = data
        self.times  = time
        self.events = event
        self.masks1 = mask1
        self.masks2 = mask2
        self.transform = transform
    
    def __len__(self):
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        time = self.times[idx]
        event = self.events[idx]
        mask1 = self.masks1[idx]
        mask2 = self.masks2[idx]

        sample = torch.tensor(sample, dtype=torch.float32)
        time   = torch.tensor(time, dtype=torch.float32)
        event = torch.tensor(event, dtype=torch.float32)
        mask1 = torch.tensor(mask1, dtype=torch.float32)
        mask2 = torch.tensor(mask2, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)
        
        return sample, time, event, mask1, mask2
