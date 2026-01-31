import torch

linear1_weights = torch.tensor([[0.2669, 0.4582],[0.1335, 0.6018]])
linear1_bias = torch.tensor([-0.3286,  0.1014])

linear2_weights = torch.tensor([[-0.2181,  0.2388]])
linear2_bias = torch.tensor([0.4648])

x1, x2 = list(map(int, input().split()))

x = torch.tensor([[x1, x2]], dtype=torch.float32)

y = x @ linear1_weights.T
y = y + linear1_bias

# Sigmoid
y = torch.sigmoid(y)

y = y @ linear2_weights.T
y = y + linear2_bias

# ReLu
y = torch.relu(y)

print('The prediction is ', y.item())

