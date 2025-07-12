import copy
import torch
import torch.nn as nn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DummySelfAttn(nn.Module):
    def forward(self, x, y, z, mask):
        return x + 10

class DummyFF(nn.Module):
    def forward(self, x):
        return x * 2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.self_attn = DummySelfAttn()
        self.feed_forward = DummyFF()
        self.sublayer = clones(SublayerConnection(size, 0.0), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

x = torch.tensor([[1.0, 2.0]])
layer = EncoderLayer(2)
print("Result: ", layer(x, None))
