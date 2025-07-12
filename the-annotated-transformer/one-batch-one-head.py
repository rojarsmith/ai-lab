import torch
import torch.nn.functional as F
import math

Q = torch.tensor([[1.0, 0.0]])
K = torch.tensor([[1.0, 0.0],
[0.0, 1.0]])
V = torch.tensor([[10.0, 0.0],
[0.0, 20.0]])

# QK^T
scores = torch.matmul(Q, K.T) / math.sqrt(Q.size(-1))
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

print("Attention score:", scores)
print("Attention weight:", weights)
print("Attention output:", output)
