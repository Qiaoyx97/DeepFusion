#coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from sklearn.metrics import roc_auc_score

from DeepFusion_24 import DeepFusion
from DataLoader import RbpDataset
from glob import glob
from tqdm import tqdm

import numpy as np
import time

result_path = "results/"

class RbpTrainer():
    def cal_auc(self, dataloader, model):
        logits = []
        labels = []
        model.eval()
        for motif, context, label in dataloader:
            motif = motif.to('cuda')
            context = context.to('cuda')
            logit = model(motif, context)
            logit = nn.Softmax(dim=1)(logit).detach().cpu().numpy()

            for item in logit:
                logits.append(item[1])

            label = label.cpu().numpy()
            for item in label:
                labels.append(item)
        auc = roc_auc_score(labels, logits)
        return auc

    def __init__(self, model_dir):
        self.logfile = open(result_path+time.strftime("%m-%d-%H", time.localtime())+'.txt','w')
        self.model_dir = model_dir

    def train(self, name):
        model = DeepFusion()
        model = model.to('cuda')

        train_dataset = RbpDataset('sample/{}/{}.train.inpt'.format(name, name))
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        valid_dataset = RbpDataset('sample/{}/{}.valid.inpt'.format(name, name))
        valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=0)

        test_dataset = RbpDataset('sample/{}/{}.test.inpt'.format(name, name))
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_func = torch.nn.CrossEntropyLoss()
        max_auc = 0
        final_auc = 0

        for epoch in tqdm(range(50)):
            total_loss = 0.0
            total_step = 0
            model.train()
            for motif, context, label in train_dataloader:
                total_step += 1
                motif = motif.to('cuda')
                context = context.to('cuda')
                label = label.to('cuda')
                logit = model(motif, context)

                optimizer.zero_grad()

                loss = loss_func(logit, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.data
            auc = self.cal_auc(valid_dataloader, model)
            if auc > max_auc:
                max_auc = auc
                final_auc = self.cal_auc(test_dataloader, model)
                best = glob(os.path.join(self.model_dir, name + '.best_*'))
                if len(best) > 0:
                    best_auc = float(best[0].split('_')[-1])
                else:
                    best_auc = 0.0
                if final_auc > best_auc:
                    if len(best) > 0:
                        os.remove(best[0])
                    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                    torch.save(state, os.path.join(self.model_dir, name + '.best'))
        print(name, final_auc)
        print(name, final_auc, file=self.logfile)
        return final_auc


if __name__=='__main__':
    name_list = glob('sample/*')
    trainer = RbpTrainer('models/')
    mean_auc = np.zeros(24)
    i = 0
    for name in name_list:
        if 'model' in name:
            continue
        rbp = name.split('/')[-1]
        rbp_auc = trainer.train(rbp)
        mean_auc[i] = rbp_auc
        i = i + 1
    print(np.mean(mean_auc), file=trainer.logfile)