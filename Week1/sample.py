import torch

a = torch.tensor(1)
print(a, a.dtype, a.dim())

b = torch.tensor([1, 2, 3, 4])
print(b, b.shape, b.size(), b.dim())

c = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(c, c.shape, c.dim())

d = torch.zeros(4)
print(d, d.shape, d.dim())

e = torch.zeros(4, 3, dtype=torch.int32)
print(e, e.shape, e.dim())

f = torch.ones(4, 3, dtype=torch.int32)
print(f, f.shape, f.dim())

g = torch.rand(4, 3)
print(g, g.shape, g.dim())

h = torch.randn(4, 3)
print(h)

i = torch.arange(0, 10, 2)
print(i)

j = torch.eye(3, dtype=torch.int32)
print(j)

# Arithmatic Operations
a = torch.tensor([1 ,2, 3, 4])
b = torch.tensor([5, 6, 7, 8])
c = torch.tensor([[1, 2], [3, 4], [5, 6]])

print(a + b, a - b, a * b, a / b)
print(a.dot(b))
print(c.mT)

print(torch.cuda.is_available())