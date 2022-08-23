# coding: utf-8
import torch
from torch import nn
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, input_seq):
        assert len(input_seq.size()) > 2

        reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))

        output = self.module(reshaped_input)
        if self.batch_first:
            output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
        else:
            output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
        return output


class RbpMotif(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 16, 10)
        self.pooling = nn.MaxPool1d(10, stride=1)
        self.dense = nn.Linear(912, 128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        x = self.pooling(x)
        x = nn.Flatten()(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.ReLU()(x)
        return x

class RbpChar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 32, 5)
        self.pooling = nn.MaxPool1d(5, stride=5)
        self.lstm = nn.LSTM(32, 16, num_layers=2, batch_first=True, bidirectional=True, dropout=0)
        self.dense = TimeDistributed(nn.Linear(74, 74), True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        pooling_result = self.pooling(x)
        pooling_result = pooling_result.transpose(1,2)
        x, _ = self.lstm(pooling_result)
        pooling_result = pooling_result.transpose(1,2)
        x = x.transpose(1,2)
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Flatten()(x)
        return x

class DeepFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.motif_model = RbpMotif()
        self.context_char_model = RbpChar()
        self.dense = nn.Linear(2496, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, motif, context):
        motif_embedding = self.motif_model(motif)
        context_embedding = self.context_char_model(context)
        x = torch.cat([motif_embedding, context_embedding], 1)
        x = self.dropout(x)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    model = DeepFusion()
    motif = torch.randn(5, 4, 75)
    context = torch.randn(5, 4, 375)
    x = model(motif, context)
    print(x)