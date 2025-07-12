import torch

x = torch.tensor([2, 4, 4, 4, 5, 5, 7, 9], dtype=torch.float32)

print("Unbiased standard deviation (n-1):", x.std()) # Default unbiased=True
print("Biased standard deviation (divided by n):", x.std(unbiased=False))

x = torch.tensor([1.0, 2.0, 3.0])
print(x * 2)

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view(2, 3)
y_0 = torch.mean(x, dim=0)
y_1 = torch.mean(x, dim=1)
print(x)
print(y_0)
print(y_1)
