#coding: utf-8
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class RbpDataset(Dataset):
    def __init__(self, path, rbp_name, mode, train=True):
        self.mode = mode
        x_center = []
        x_seq = []
        y = []
        if train:
            pos = pd.read_table(path + '/' + rbp_name + '.seq.pos.txt', header=None)
            pos = pd.DataFrame(pos)
            pos.columns = ['label', 'center', 'seq', 'region', 'num']
            x_center_pos = pos['center'].apply(lambda seq: self.array_seq(seq))
            x_seq_pos = pos['seq'].apply(lambda seq: self.array_seq(seq))
            y_pos= pos['label'].apply(lambda y: int(y))
            
            vivo = pd.read_table(path + '/' + rbp_name + '.vivo.pos.txt', header=None)
            vivo = pd.DataFrame(vivo)
            vivo.columns = ['label', 'center', 'seq', 'region', 'num']
            center = vivo['center'].apply(lambda x: x.split(','))
            seq = vivo['seq'].apply(lambda x: x.split(','))
            for s, c in zip(x_center_pos, center):
                for i in range(len(s)):
                    s[i][4] = float(c[i])
            for s, c in zip(x_seq_pos, seq):
                for i in range(len(s)):
                    s[i][4] = float(c[i])

            for item in x_center_pos:
                x_center.append(item)
            
            for item in x_seq_pos:
                x_seq.append(item)
            
            for item in y_pos:
                y.append(item)

        x_center = np.array(list(x_center))
        self.x_center = x_center.reshape(-1, 75, 5)
        x_seq = np.array(list(x_seq))
        self.x_seq = x_seq.reshape(-1, 375, 5)
        self.y = self.format_y(y)

        self.raw_motif = pos['center']
        self.struct_seq = vivo['center']
        
    
    def array_seq(self, seq):
        base_dict = {'A': 0, 'G': 1, 'C': 2, 'U': 3, 'T': 3}
        lst = []
        for c in seq:
            item = [float('0')] * 5
            if c != 'N':
                item[base_dict[c]] = float('1')
            else:
                item = [float('0.25')] * 5
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
        return torch.Tensor(self.x_center[idx].transpose((1, 0))), self.raw_motif[idx], self.struct_seq[idx]