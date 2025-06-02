import torch

# squeeze()

data = torch.tensor(
    [[[0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],]]
)

print('Shape:', data.shape)

squeeze_data = data.squeeze(0)
print('squeeze data:', squeeze_data)
print('squeeze(0) shape:', squeeze_data.shape)

# unsqueeze()

data = torch.tensor(
    [[[0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],]]
)

print('Shape:', data.shape)

unsqueeze_data = data.unsqueeze(0)
print('unsqueeze data:', unsqueeze_data)
print('unsqueeze(0) shape:', unsqueeze_data.shape)
