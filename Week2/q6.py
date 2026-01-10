import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# Forward Pass
a = 2 * x
b = torch.sin(y)
c = a / b
d = c * z
e = torch.log(d + 1)
f = torch.tanh(e)

print(f"The Values of Intermediatary Values in Forward Passs are a={a}, b={b}, c={c}, d={d}, e={e}, f={f}")

# Backwards pass
f.backward()
print("The value of df/dy using torch is ", y.grad)
print("The value of df/dx using torch is ", x.grad)

# Analytical Solution
dy = (1 - (torch.tanh(e) ** 2)) * (1 / (d + 1)) * z * (1 / b) * 2
dx = (1 - (torch.tanh(e) ** 2)) * (1 / (d + 1)) * z * (-a /(b ** 2)) * torch.cos(y)

print("The value of df/dy using Analytical Solution is ", dy.item())
print("The value of df/dx using Analytical Solution is ", dx.item())

