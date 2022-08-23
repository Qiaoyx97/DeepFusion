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
import json

class RbpTestner():
    def __init__(self, rbp_dir):
        self.rbp_dir = rbp_dir
    
    def load_checkpoint(self, model, checkpoint_PATH):
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['net'])
        print('loading checkpoint!')
        return model
    
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

    def test(self, name):
        model = DeepFusion()
        model = self.load_checkpoint(model, self.rbp_dir)
        model = model.to('cuda')
        model.eval()

        test_dataset = RbpDataset('RBP-24/sample/{}/{}.test.inpt'.format(name, name))
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

        final_auc = self.cal_auc(test_dataloader, model)
        print(name + ' is finished.')
        return final_auc

if __name__=='__main__':
    rbplist = glob('RBP-24/sample/*')
    print(rbplist) 
    log_path = 'RBP-24/results/' + time.strftime("%m-%d-%H", time.localtime())
    log_file = open(log_path + 'log.txt','w')
    rbpnames = []
    for i in rbplist:
        rbpnames.append(i.split('/')[-1])
    for rbpname in rbpnames:
        model_dir = 'RBP-24/models/' + rbpname + '.best*'
        rbp_dir = glob(model_dir)[0]
        testner = RbpTestner(rbp_dir)
        result = testner.test(rbpname)
        print(rbpname + ' ' + str(result), file = log_file)