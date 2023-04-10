import math
import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
from torch.nn import init

class MCA(nn.Module):

    def __init__(self, num_heads, dim_Q, dim_K, dim_hid, Dropout=False, ln=False):   # 5, 512, 512, 500
        super(MCA, self).__init__()
        self.dim_hid = dim_hid
        self.num_heads = num_heads
        self.dim_K = dim_K
        self.fc_q = nn.Linear(dim_Q, dim_hid)
        self.fc_k = nn.Linear(dim_K, dim_hid)
        self.fc_v = nn.Linear(dim_K, dim_K * self.num_heads)
        self.Dropout = Dropout
        self.do = nn.Dropout(p=0.2)
        if ln:
            self.ln0 = nn.LayerNorm(dim_hid)
            self.ln1 = nn.LayerNorm(dim_hid)
        self.fc_o = nn.Linear(dim_K * self.num_heads, dim_K)

    def forward(self, Q, K):

        bsz, h, w = K.shape[0], K.shape[2], K.shape[3]
        K = K.permute(0, 2, 3, 1)
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)

        # 1 x h x w x 5 x 100
        Q = Q.view(bsz, -1, self.num_heads, self.dim_hid // self.num_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.num_heads, self.dim_hid // self.num_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.num_heads, self.dim_K).permute(0, 2, 1, 3)
        A = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.dim_K)
        A = torch.softmax(A, dim=-1)
        # print(A[:, :2, 2500:2510])
        if self.Dropout:
            A = self.do(A)
        x = torch.matmul(A, V)
        x = x.reshape(1, self.num_heads * self.dim_K)
        x = self.fc_o(x)   # [1, 512]

        return x

