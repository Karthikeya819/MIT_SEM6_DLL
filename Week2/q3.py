import torch

b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

# Forward Passs
u = w * x
v = u + b
a = torch.sigmoid(v)

# Backward Pass
a.backward()
print("The value of da/dw using torch is ", w.grad)

# Analytical Solution
dw = torch.sigmoid(v) * (1 - torch.sigmoid(v)) * x
print("The value of da/dw using Analytical Solution is ", dw.item())