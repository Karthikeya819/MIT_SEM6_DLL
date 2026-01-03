import torch

# 1
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a, a.reshape(-1, 1), torch.stack((a, a)))

#2
print(torch.permute(a, (0, 1)))

#3
print(a[:2])

#4
import numpy as np
arr =  np.array([1, 2, 3])
arr_tor = torch.tensor(arr)
print(arr_tor, arr_tor.numpy())

#5
print(torch.rand(7, 7))

#6
a = torch.rand(1, 7)
print(a * a.T)

#7
a, b = torch.rand(2, 3), torch.rand(2, 3)
a, b = a.to('cuda'), b.to('cuda')

#8
c = a * b

#9
print(c.min(), c.max())

#10
print(c.argmin(), c.argmax())