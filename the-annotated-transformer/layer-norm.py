import torch

x = torch.tensor([[1.0, 2.0, 3.0]])
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
# 1e-6 = 0.000001
normalized = (x - mean) / (std + 1e-6)
print("Result: ", normalized)
