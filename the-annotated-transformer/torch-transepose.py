import torch as tor

# 2-dimension
t = tor.arange(1, 7)
print("tor.arange(1, 7) = ", t)

t = t.reshape(2, 3)
print("t.reshape(2, 3) = ", t)

t = t.transpose(1, 0)
print("t.transpose(1, 0) = ", t)

# 3-dimension
t = tor.arange(1, 17)
print("tor.arange(1, 17) = ", t)

t = t.reshape(2, 2, 4)
print("t.reshape(2, 2, 4) = ", t)
tc = t

t = t.transpose(1, 0)
print("t.transpose(1, 0) = ", t)

# tc[i][j][k] => tc[j][i][k]
print("tc =", tc)

r00 = tc[0][0]
r01 = tc[0][1]
r10 = tc[1][0]
r11 = tc[1][1]

print("r00 =", tc[0][0])
print("r01 =", tc[0][1])
print("r10 =", tc[1][0])
print("r11 =", tc[1][1])

tc = tor.stack([
    tor.stack([r00, r10], dim=0),
    tor.stack([r01, r11], dim=0)
], dim=0)

print("tc =", tc)

# tc[i][j][k] => tc[j][i][k]
tc = t
A, B, C = tc.shape
result = []
for b in range(B):
    row = []
    for a in range(A):
        row.append(tc[a][b])  # Exchange a and b
    result.append(tor.stack(row, dim=0))

t_swapped = tor.stack(result, dim=0)
print("t_swapped =", t_swapped)
