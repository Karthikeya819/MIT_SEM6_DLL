import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

x1 = torch.tensor([3, 4, 5, 6, 2])
x2 = torch.tensor([8, 5, 7, 3, 1])
y = torch.tensor([8, 5, 7, 3, 1])

class MyDataset(Dataset):
    def __init__(self, x1, x2, y):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.y = y
    
    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.y[index]
    
    def __len__(self):
        return len(self.x1)
    
data = MyDataset(x1, x2, y)

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(1.0))
        self.w2 = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2 + self.b
    
    def criteration(yj, y_p):
        return (yj - y_p) ** 2
    
    def parameters(self):
        return (self.w1, self.w2, self.b)

model = RegressionModel()

optim = SGD(model.parameters(), lr = 0.001)

for _ in range(100):

    for inp1, inp2, out in DataLoader(data, batch_size=1, shuffle=True):
        optim.zero_grad()

        y_p = model.forward(inp1, inp2)
        loss = RegressionModel.criteration(out[0], y_p)

        loss.backward()
        optim.step()

    print("the paarmeters are w={}, b ={}, and loss = {}".format((model.w1.item(), model.w2.item()), model.b.item(), loss))

print("The prediction for x1=3, x2=2 is {}".format(model.forward(3, 2)))

