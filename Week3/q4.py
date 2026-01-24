import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

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
        return self.w *x + self.b
    
    def criteration(yj, y_p):
        return (yj - y_p) ** 2
    
    def parameters(self):
        return (self.w, self.b)

model = RegressionModel()

loss_list = []
optim = SGD(model.parameters(), lr = 0.001)

for _ in range(100):
    loss = torch.tensor([0.0])
    optim.zero_grad()

    for inp, out in DataLoader(data, batch_size=1, shuffle=True):
        y_p = model.forward(inp[0])
        loss += RegressionModel.criteration(out[0], y_p)
    
    loss = loss / len(data)
    loss_list.append(loss.item())

    loss.backward()

    optim.step()

    print("the paarmeters are w={}, b ={}, and loss = {}".format(model.w, model.b, loss))

plt.plot(loss_list)
plt.show()