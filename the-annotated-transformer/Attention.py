import torch
import torch.nn.functional as F

# 3 words, each 2D vector
Q = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]]) # The third word wants to combine the first two

K = Q # Here it is simplified to K = Q
V = torch.tensor([[10.0, 0.0],
                  [0.0, 20.0],
                  [5.0, 5.0]])

# QK^T
scores = torch.matmul(Q, K.T) / (2 ** 0.5)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, V)

print("Attention score matrix: \n", scores)
print("Attention weights (softmax): \n", weights)
print("Final output vector: \n", output)
