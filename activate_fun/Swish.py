import torch
import torch.nn.functional as F


class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *(torch.tanh(F.softplus(x)))