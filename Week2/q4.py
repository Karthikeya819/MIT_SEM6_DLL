import torch

x = torch.tensor(2.0, requires_grad=True)

f = torch.exp((-1 * (x ** 2) - 2 * x - torch.sin(x)))

f.backward()
print("The value of df/dx using torch is ", x.grad)

# Analytical Solution
dx = torch.exp((-1 * (x ** 2) - 2 * x - torch.sin(x))) * (-2* x - 2 - torch.cos(x))
print("The value of df/dx using Analytical Solution is ", dx.item())