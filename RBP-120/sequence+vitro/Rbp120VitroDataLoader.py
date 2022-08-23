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
            
            vitro = pd.read_table(path + '/' + rbp_name + '.vitro.pos.txt', header=None)
            vitro = pd.DataFrame(vitro)
            vitro.columns = ['label', 'center', 'seq', 'region', 'num']
            center = vitro['center'].apply(lambda x: x.split(','))
            seq = vitro['seq'].apply(lambda x: x.split(','))
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

            neg = pd.read_table(path + '/' + rbp_name + '.seq.neg.txt', header=None)
            neg = pd.DataFrame(neg)
            neg.columns = ['label', 'center', 'seq', 'region', 'num']
            x_center_neg = neg['center'].apply(lambda seq: self.array_seq(seq))
            x_seq_neg = neg['seq'].apply(lambda seq: self.array_seq(seq))
            y_neg = neg['label'].apply(lambda y: int(y))
            
            vitro = pd.read_table(path + '/' + rbp_name + '.vitro.neg.txt', header=None)
            vitro = pd.DataFrame(vitro)
            vitro.columns = ['label', 'center', 'seq', 'region', 'num']
            center = vitro['center'].apply(lambda x: x.split(','))
            seq = vitro['seq'].apply(lambda x: x.split(','))
            for s, c in zip(x_center_neg, center):
                for i in range(len(s)):
                    s[i][4] = float(c[i])
            for s, c in zip(x_seq_neg, seq):
                for i in range(len(s)):
                    s[i][4] = float(c[i])

            for item in x_center_neg:
                x_center.append(item)
            
            for item in x_seq_neg:
                x_seq.append(item)
            
            for item in y_neg:
                y.append(item)

        else:
            data = pd.read_table(path + '/' + rbp_name + '.test.seq.txt', header=None)
            data = pd.DataFrame(data)
            data.columns = ['label', 'center', 'seq', 'region', 'num']
            x_center_ls = data['center'].apply(lambda seq: self.array_seq(seq))
            x_seq_ls = data['seq'].apply(lambda seq: self.array_seq(seq))
            y = data['label'].apply(lambda y: int(y))
            
            vitro = pd.read_table(path + '/' + rbp_name + '.test.vitro.txt', header=None)
            vitro = pd.DataFrame(vitro)
            vitro.columns = ['label', 'center', 'seq', 'region', 'num']
            center = vitro['center'].apply(lambda x: x.split(','))
            seq = vitro['seq'].apply(lambda x: x.split(','))
            for s, c in zip(x_center_ls, center):
                for i in range(len(s)):
                    s[i][4] = float(c[i])
            for s, c in zip(x_seq_ls, seq):
                for i in range(len(s)):
                    s[i][4] = float(c[i])
            
            for item in x_center_ls:
                x_center.append(item)
            
            for item in x_seq_ls:
                x_seq.append(item)

        if train:
            x_center_train, x_center_val, x_seq_train, x_seq_val, y_train, y_val = train_test_split(x_center, x_seq, y, random_state=233, test_size=0.15)
            if mode == 'train':
                x_center = x_center_train
                x_seq = x_seq_train
                y = y_train
            else:
                x_center = x_center_val
                x_seq = x_seq_val
                y = y_val

        x_center = np.array(list(x_center))
        self.x_center = x_center.reshape(-1, 75, 5)
        x_seq = np.array(list(x_seq))
        self.x_seq = x_seq.reshape(-1, 375, 5)
        self.y = self.format_y(y)
        
    
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
        return torch.Tensor(self.x_center[idx].transpose((1, 0))), torch.Tensor(self.x_seq[idx].transpose((1, 0))), self.y[idx]