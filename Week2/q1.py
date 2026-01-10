import torch

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

# Forward Pass
x = 2 * a + 3 * b
y = 5 * a * a + 3 * b * b * b
z = 2 * x + 3 * y

# Backward Pass
z.backward()
print("The value of dz/da using torch is ", a.grad)

# Analytical Solution
da = 4 + 30 * a
print("The value of dz/da using Analytical Solution is ", da.item())