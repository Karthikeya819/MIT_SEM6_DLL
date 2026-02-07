import torch
from torch.nn import Conv2d
import torch.nn.functional as F

torch.manual_seed(819)

stride = 1
padding = 1
kernel_size = 3
I = 6

image = torch.rand(I, I).unsqueeze(dim=0).unsqueeze(dim=0)
kernel = torch.ones(kernel_size, kernel_size).unsqueeze(dim=0).unsqueeze(dim=0)


class Conv2dFunctional(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):   
        super().__init__()

        kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.nn.Parameter(torch.rand(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


model = Conv2d(1, 3, kernel_size, stride, padding, bias=False)

with torch.no_grad():
    outimg = model.forward(image)

print("outimg.shape=", outimg.shape)
print("outimg=", outimg)