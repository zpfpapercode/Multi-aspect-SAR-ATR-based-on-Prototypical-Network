import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(classifier, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc=nn.Linear(in_dim,out_dim)
    def forward(self,x):
        x=self.dropout(x)
        x=self.fc(x)
        return x

