import matplotlib.pyplot as plt
import numpy as np

def get_positional_encoding(position, d_model):
    PE = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * i)/d_model)
            PE[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                PE[pos, i+1] = np.cos(angle)
    return PE

pe = get_positional_encoding(50, 16)
plt.imshow(pe.T, cmap='coolwarm', aspect='auto')
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.title("Positional Encoding Heatmap")
plt.colorbar()
plt.show()
