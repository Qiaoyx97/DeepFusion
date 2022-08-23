#coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from sklearn.metrics import roc_auc_score

from DeepFusionstruct import DeepFusion
from Rbp120VivoDataLoader import RbpDataset
from glob import glob
from tqdm import tqdm
import numpy as np
import time

result_path = 'RBP-120/sequence+vivo/results/'

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
        self.logfile = open(result_path+time.strftime("%m-%d-%H", time.localtime())+'rbp120Vivo.txt','w')
        self.model_dir = model_dir
    
    def train(self, name, rbp):
        model = DeepFusion()
        model = model.to('cuda')

        train_dataset = RbpDataset(name, rbp, 'train')
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        val_dataset = RbpDataset(name, rbp, 'val')
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

        test_dataset = RbpDataset(name, rbp, 'test', False)
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
            auc = self.cal_auc(val_dataloader, model)
            if auc > max_auc:
                max_auc = auc
                final_auc = self.cal_auc(test_dataloader, model)
                best = glob(os.path.join(self.model_dir, rbp + '.best_*'))
                if len(best) > 0:
                    best_auc = float(best[0].split('_')[-1])
                else:
                    best_auc = 0.0
                if final_auc > best_auc:
                    if len(best) > 0:
                        os.remove(best[0])
                    state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
                    torch.save(state, os.path.join(self.model_dir, rbp + '.best_' + str(final_auc)))
        print(name, final_auc)
        print(name, final_auc, file=self.logfile)
        return final_auc


if __name__=='__main__':
    name_list = glob('RBP-120/sequence+vivo/sample/*')
    trainer = RbpTrainer('RBP-120/sequence+vivo/models/')
    for name in name_list:
        if '.py' in name:
            continue
        rbp = name.split('/')[-1]
        rbp_auc = trainer.train(name, rbp)