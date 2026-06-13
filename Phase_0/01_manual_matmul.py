A = [
    [1,2],
    [3,4]
]

B =[
    [5,6],
    [7,8]
]

print("A:")
for row in A:
    print(row)

print("B:")
for row in B:
    print(row)

C = [[0,0],[0,0]]

for i in range(2):
    for j in range(2):
        for k in range(2):
            C[i][j] += A[i][k] * B[k][j]

print("C: ")
for row in C:
    print(row)

##### APPROACH WITH TORCH #####

import torch

A = torch.tensor(A)
B = torch.tensor(B)

print(torch.matmul(A, B))