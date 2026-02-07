import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt

torch.manual_seed(819)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# model_path = "cnn_classifier_small.pth"

class CNNClassifier(nn.Module):
    def __init__(self, red_factor:int=1):
        super().__init__()

        self.net = nn.Sequential(nn.Conv2d(1, 64 // red_factor, kernel_size=3), nn.ReLU(), nn.MaxPool2d((2, 2), stride=2), 
                                 nn.Conv2d(64 // red_factor, 128 // red_factor, kernel_size=3), nn.ReLU(), nn.MaxPool2d((2, 2), stride=2), 
                                 nn.Conv2d(128 // red_factor, 64 // red_factor, kernel_size=3), nn.ReLU(), nn.MaxPool2d((2, 2), stride=2))
        
        self.classification_head = nn.Sequential(nn.Linear(64 // red_factor, 20, bias=True), nn.ReLU(), nn.Linear(20, 10, bias=True), nn.Softmax(dim=1))

    
    def forward(self ,x):
        features = self.net(x)
        features = torch.flatten(features, start_dim=1)
        return self.classification_head(features)
    
def Train(red_factor:int):
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = CNNClassifier(red_factor).to(device)
    criteration = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

    for epoch in range(5):
        total_loss = torch.tensor(0.0, device=device)
        for inps, outs in train_data:
            inps, outs = inps.to(device), outs.to(device)

            optimizer.zero_grad()
        
            y_pred = model(inps)
            loss = criteration(y_pred, outs)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                total_loss += loss

        total_loss = total_loss/ len(train_data)
        print(f"Model: {red_factor}, Epoch {epoch}/ 5, loss = {total_loss.item()}")
        torch.save(model.state_dict(), f"cnn_classifier_{red_factor}.pth")

def Inference(red_factor:int):

    test_data = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

    model = CNNClassifier(red_factor)
    model.load_state_dict(torch.load(f"cnn_classifier_{red_factor}.pth", map_location=device))

    pred_class, orig_class = None, None
    with torch.no_grad():
        for inps, outs in test_data:
            pred_class = model(inps)
            pred_class = torch.argmax(pred_class, 1)
            orig_class = outs
            break
    confmat = MulticlassConfusionMatrix(num_classes=10)
    cm = confmat(pred_class, orig_class)
    print(cm)
    correct = cm.diag().sum()
    total = cm.sum()
    return sum(p.numel() for p in model.parameters()), (correct / total).item()


# Train
accs = []

for model_size in [1, 2, 4, 8]:
    model_path = f"cnn_classifier_{model_size}.pth"
    if os.path.isfile(model_path):
        accs.append(Inference(model_size))
    else:
        Train(model_size)
print(accs)
plt.plot(accs)
plt.show()
