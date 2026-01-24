import torch

x = torch.tensor([2, 4])
y = torch.tensor([20, 40])
lr = 0.001

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

for _ in range(2):

    for i in range(len(x)):
        y_pred = w * x[i] + b
        loss = (y_pred - y[i]) ** 2

        loss.backward()
        dw_tensor, db_tensor = w.grad.item(), b.grad.item()

        w.grad.zero_()
        b.grad.zero_()

        with torch.no_grad():
            error = y_pred - y[i]
            dw = 2 * error * x[i]
            db = 2 * error

            print(f"W_tensor: {dw_tensor} , W_Analyical: {dw.item()} ")
            print(f"b_tensor: {db_tensor} , b_Analyical: {db.item()} ")

            #Upadtion of Weights
            w -= lr * dw
            b -= lr * db

    
print("The final trained weights of the Model are", w.item(), b.item())