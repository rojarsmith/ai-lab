import numpy as np
import matplotlib.pyplot as plt

#####
def compute_e(n):
    return (1 + 1/n)**n

for n in [1, 10, 100, 1000, 10000, 100000]:
    print(f"n={n}: {compute_e(n)}")
#####

# 2.718281828459045
print("nxp(1) =", np.exp(1))
print(np.log(np.exp(1)))

print(np.log(2.666))
print(np.log(2.718281828459045))

x = np.linspace(-3, 3, 200)
y = np.exp(x)
plt.plot(x, y)
plt.title("exp(x) curve")
plt.xlabel("x")
plt.ylabel("exp(x)")
plt.grid(True)
plt.show()

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract the maximum value to avoid overflow
    return e_x / e_x.sum()

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)

print("Input score:", logits)
print("Softmax probability:", probs)
print("Total:", probs.sum())

def plot_softmax_outputs():
    x = np.linspace(-2, 2, 100)
    logits_set = [
        [x_i, 0, -x_i] for x_i in x
    ]
    softmax_outputs = [softmax(np.array(logits)) for logits in logits_set]

    y1 = [out[0] for out in softmax_outputs]
    y2 = [out[1] for out in softmax_outputs]
    y3 = [out[2] for out in softmax_outputs]

    plt.plot(x, y1, label='Class 1')
    plt.plot(x, y2, label='Class 2')
    plt.plot(x, y3, label='Class 3')
    plt.title('Softmax Probability Distribution')
    plt.xlabel('Score difference')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_softmax_outputs()

import torch
import torch.nn.functional as F

# Each word will be converted into a vector.
# Here we simulate 3 words, each word is a 4-dimensional vector
# Use .tolist() to view full data
query = torch.rand(3, 4)
key = torch.rand(3, 4)
value = torch.rand(3, 4)

# 1. Calculate the Attention score (Q * K^T)
# In 2 dimensions, the order of transpose is not important,
# but in 3 dimensions and above, it is important.
scores = torch.matmul(query, key.transpose(0, 1))

# 2. Softmax regularization: convert scores to probabilities
weights = F.softmax(scores, dim=-1)

# 3. Weighted output (weights * V)
output = torch.matmul(weights, value)

print("Self-attention output:")
print(output)
