import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Dict, Optional

class MultiHeadAtt(nn.Module):
    def __init__(self, config: Dict):
        super(MultiHeadAtt, self).__init__()        
        self.d_model = config['attention']['d_model']
        self.heads = config['attention']['heads']
        self.d_k = config['attention']['d_model']
        self.d_v = config['attention']['d_model']
        self.dropout = config['attention']['dropout']
        self.matt = nn.MultiheadAttention(embed_dim=self.d_model,num_heads=self.heads,dropout=self.dropout,kdim=self.d_k,vdim=self.d_v)

    def forward(self, queries, keys, values, attention_mask=None):
        out, _ = self.matt(queries, keys, values, attention_mask)
        return out