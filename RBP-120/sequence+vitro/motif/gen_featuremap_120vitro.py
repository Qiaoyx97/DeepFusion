import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0,'..')
from DeepFusionstruct import DeepFusion
from Rbp120VitroDataLoader import RbpDataset
from glob import glob
import numpy as np
import subprocess

class RbpTestner():
    def __init__(self, rbp_dir, motif_dir, tomtom_dir):
        self.rbp_dir = rbp_dir
        self.motif_dir = motif_dir
        self.tomtom_dir = tomtom_dir
    
    def load_checkpoint(self, model, checkpoint_PATH):
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['net'])
        print('loading checkpoint!')
        return model

    def test(self, name, data_dir):
        model = DeepFusion()
        model = self.load_checkpoint(model, self.rbp_dir)
        model = model.to('cuda')
        model.eval()
        train_dataset = RbpDataset(data_dir, name, 'train')
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
        features = []
        for i in range(16):
            features.append([])
        for idx, (motif, seq, struct_seq) in enumerate(train_dataloader):
            seq = seq[0]
            struct_seq = struct_seq[0]
            motif = motif.to('cuda')
            feature = model.motif_model.motif_forward(motif)
            feature = feature.detach().cpu().numpy()
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    logit = feature[i][j]
                    posi = 0
                    for k in range(logit.shape[0]):
                        if logit[k] > 0:
                            posi = k
                            if 'N' not in seq[posi : posi+10]:
                                struct_motif = struct_seq.split(',')[posi : posi+10]
                                struct_motif = ','.join(struct_motif)
                                features[j].append((seq[posi : posi+10].replace('T', 'U'), logit[posi], idx, j, struct_motif))

        ch = 0
        for channel in features:
            if len(channel) != 0:
                print(ch)
                print(len(channel))
                max_features = sorted(channel, key=lambda x:x[1], reverse=True)[0 : 2000]
                fm = open(self.motif_dir+str(ch)+'.txt', 'w')
                fmt = open(self.motif_dir.split('.')[0]+str(ch)+'seq.txt', 'w')
                fms = open(self.motif_dir.split('.')[0]+str(ch)+'struct.txt', 'w')
                motif_lines = []
                for item in max_features:
                    print('>{}_{}'.format(item[2], item[3]), file=fm)
                    print(item[0], file=fm)
                    print(item[0], file=fmt)
                    print(item[4], file=fms)
                    motif_lines.append(item[0])
                fm.close()

                cnt = []
                for i in range(10):
                    cnt.append({'A':0, 'C':0, 'G':0, 'U':0})
                for i in motif_lines:
                    for j in range(10):
                        cnt[j][i[j]] += 1
                ft = open(self.tomtom_dir+str(ch)+'.txt', 'w')
                for c in cnt:
                    print(str(float(c['A']/len(motif_lines)))+' '+str(float(c['C']/len(motif_lines)))+' '+str(float(c['G']/len(motif_lines)))+' '+str(float(c['U']/len(motif_lines))), file=ft)
                ch = ch + 1
            else:
                ch = ch + 1
                continue

if __name__=='__main__':
    rbpnames = ['AARS']
    
    for rbpname in rbpnames:
        model_dir = '../models/' + rbpname + '.best'
        if os.path.exists(rbpname + '/') is False:
            os.mkdir(rbpname + '/')
        motif_dir = rbpname + '/' + rbpname + '_motif_'
        tomtom_dir  = rbpname + '/' + rbpname + '_tomtom_'
        data_dir = '../sample/' + rbpname
        rbp_dir = glob(model_dir)[0]
        testner = RbpTestner(rbp_dir, motif_dir, tomtom_dir)
        rbp_feature = testner.test(rbpname, data_dir)