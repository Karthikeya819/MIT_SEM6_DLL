import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)

class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
data = MyDataset(x, y)

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        return torch.sigmoid(self.w * x + self.b)
    
    def criteration(yj, y_p, eps=1e-7):
        y_p = torch.clamp(y_p, eps, 1 - eps)
        return -(yj * torch.log(y_p) + (1 - yj) * torch.log(1 - y_p))

    def parameters(self):
        return (self.w, self.b)

model = RegressionModel()

optim = SGD(model.parameters(), lr = 0.00001)

for _ in range(10):
    for inp, out in DataLoader(data, batch_size=1, shuffle=True):
        optim.zero_grad()
        y_p = model.forward(inp)
        loss = RegressionModel.criteration(out, y_p)

        loss.backward()
        optim.step()

print("The parameters are w={}, b ={}".format(model.w.item(), model.b.item()))

