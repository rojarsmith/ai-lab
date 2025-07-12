import torch
import torch.nn as nn
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DummyLayer(nn.Module):
    def forward(self, x, mask=None):
        return x + 1  # Simulate doing something

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(normalized_shape=1)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# Just try it
x = torch.tensor([[1.0]])
encoder = Encoder(DummyLayer(), 3)
print("Result: ", encoder(x))
