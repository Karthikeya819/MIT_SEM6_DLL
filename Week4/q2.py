import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

loss_list = []
torch.manual_seed(42)


X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class XORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(2, 1, bias=True)
        self.activation2 = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.activation2(x)

        return x
    
class MyDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
    
    def __getitem__(self, index):
        return self.X[index].to(device), self.Y[index].to(device)
    
    def __len__(self):
        return len(self.X)
    
dataset = MyDataset(X, Y)
tarin_data = DataLoader(dataset, batch_size=1, shuffle=True)

model = XORModel().to(device)
criteration = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

losses = []

for epoch in range(1000):
    total_loss = torch.tensor(0.0, device=device)
    for inps, outs in tarin_data:
        optimizer.zero_grad()

        y_pred = model(inps)
        loss = criteration(y_pred, outs)

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            total_loss += loss

    losses.append(total_loss.item())

plt.plot(losses)
plt.show()

for name, param in model.named_parameters():
    print(name, param.data)
