import torch

x = torch.rand(2, 4, 8) # 2 batches, 4 words, 8-dimensional features
h = 2 # 2 heads → 4 dimensions per head

# Simulate linear results
x = x.view(2, 4, h, 4).transpose(1, 2)
print("Shape after cutting into multiple heads:", x.shape) # (2, 2, 4, 4)

# After attention → pieced back
x = x.transpose(1, 2).contiguous().view(2, 4, 8)
print("Shape after pieced back:", x.shape) # (2, 4, 8)
