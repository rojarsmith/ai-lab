import torch.nn.functional as F
import torch.nn as nn
import torch

x = torch.rand(2, 512)  # Word vectors for 2 words
linear1 = nn.Linear(512, 2048)
linear2 = nn.Linear(2048, 512)

# forward process
x = F.relu(linear1(x))
x = linear2(x)
print("FeedForward output shape :", x.shape)
