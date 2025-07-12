import torch
import torch.nn.functional as F

# Only 1 neuron

x = torch.tensor([0.3, 0.6, 0.9])  # Input data
w = torch.tensor([0.5, -0.2, 0.1])  # Weight
print("0.3 * 0.5 + 0.6 * -0.2 + 0.9 * 0.1 =", 0.3 * 0.5 + 0.6 * -0.2 + 0.9 * 0.1)
b = 0.1  # Bias

z = torch.dot(x, w) + b  # Weighted sum + bias
print("torch.dot(x, w) = ", torch.dot(x, w))
print("z =", z)
out = F.relu(z)  # Activated by ReLU
print(out)
