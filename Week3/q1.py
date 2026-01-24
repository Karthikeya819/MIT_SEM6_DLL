import torch
import matplotlib.pyplot as plt

x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4, 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6, 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])
lr = 0.001

w = torch.tensor(1.0)
b = torch.tensor(1.0)

losses = []

for _ in range(10):

    loss = torch.tensor(0.0)

    for i in range(len(x)):
        # FOrward pass
        y_pred = w * x[i] + b

        error = y[i] - y_pred

        # Backward pass
        dw = -2 * error * x[i]
        db = -2 * error


        #Upadtion of Weights
        w = w - lr * dw
        b = b - lr * db
        
        loss += error ** 2
    
    loss = loss**0.5
    losses.append(loss.item())


print("The final trained weights of the Model are", w.item(), b.item())
plt.plot(losses)
plt.show()