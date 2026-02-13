import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
# sigmoid(z) = 1 / ( 1 + e^(-z) )
# a = 1 / ( 1 + e^(-2.42) ) = 0.9183397445...
a = torch.sigmoid(z)

def binary_cross_entropyloss(prob, y):
    loss = - (y * torch.log(prob) + (1 - y) * (torch.log(1 - prob)))
    loss = torch.sum(loss) / torch.numel(y)
    return loss

loss = F.binary_cross_entropy(a, y)
loss2 = binary_cross_entropyloss(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

# z = x1 * w1 + b
# dz/dw1 = x1
# a = sigmoid(z) = = 1 / ( 1 + e^(-z) )
# da/dz = a*(1-a)
# loss = BCE(a, y)
# dL/da = (a - y) / (a*(1-a))
# dL/dw1 = [(a-y)/(a*(1-a))] * [a*(1-a)] * [x1]
# grad_L_w1 = (a - y) * x1
print(grad_L_w1)
print(grad_L_b)

loss.backward()
print(w1.grad)
print(b.grad)
