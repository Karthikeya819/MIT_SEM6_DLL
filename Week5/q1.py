import torch
import torch.nn.functional as F

torch.manual_seed(819)

stride = 1
padding = 1
kernel_size = 3
I = 6

image = torch.rand(I, I)
print("image=", image)

image = image.unsqueeze(dim=0).unsqueeze(dim=0)
print("image.shape=", image.shape)
print("image=", image)

kernel = torch.ones(kernel_size, kernel_size)
print("kernel=", kernel)

kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
print("kernel.shape=", kernel.shape)

outimage = F.conv2d(image, kernel, stride=stride, padding=padding)
print("outimage.shape=", outimage.shape)
print("outimage=", outimage)

print(f"Caluculated Output shape = {(I - kernel_size + 2 * padding) // stride + 1}")