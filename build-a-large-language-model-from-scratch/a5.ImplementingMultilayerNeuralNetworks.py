import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


model = NeuralNetwork(50, 3)
print(model)

# bias = True
# ReLU = 0 paramter
#
# Linear(50, 30)
# 50*30 + 30 = 1500 + 30 = 1530
# Linear(30, 20)
# 30*20 + 20 = 600 + 20 = 620
# Linear(20, 3)
# 20*3 + 3 = 60 + 3 = 63
# SUM : 1530 + 620 + 63 = 2213
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)
print(model.layers[0].bias)

torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)

torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
# h1 = ReLU(X @ W1^T + b1)
# h2 = ReLU(h1 @ W2^T + b2)
# out = h2 @ W3^T + b3
print(out) # tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)

with torch.no_grad():
    out = model(X)
print(out)

with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)
