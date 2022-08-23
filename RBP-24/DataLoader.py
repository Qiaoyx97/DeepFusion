#coding: utf-8
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class RbpDataset(Dataset):
    def __init__(self, path):
        data = pd.read_table(path, header=None)
        data = pd.DataFrame(data)
        data.columns = ['label', 'center', 'seq']
        x_center = data['center'].apply(lambda seq: self.array_seq(seq))
        x_seq = data['seq'].apply(lambda seq: self.array_seq(seq))
        y = data['label'].apply(lambda y: int(y))

        x_center = np.array(list(x_center))
        self.x_center = x_center.reshape(-1, 75, 4)
        x_seq = np.array(list(x_seq))
        self.x_seq = x_seq.reshape(-1, 375, 4)
        self.y = self.format_y(y)
    
    def array_seq(self, seq):
        base_dict = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'T': 3}
        lst = []
        for c in seq:
            item = [float('0')] * 4
            if c != 'N':
                item[base_dict[c]] = float('1')
            else:
                item = [float('0.25')] * 4
            lst.append(item)
        return np.expand_dims(np.array(lst), axis=-1)

    def format_y(self, labels):
        y = []
        for label in labels:
            if label == 1:
                y.append(1)
            else:
                y.append(0)
        return np.array(y)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.Tensor(self.x_center[idx].transpose((1, 0))), torch.Tensor(self.x_seq[idx].transpose((1, 0))), self.y[idx]