import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class CenteredLayer(nn.Module):
    def forward(self, X):
        return X - X.mean()


layer = CenteredLayer()
a = layer(torch.FloatTensor([1, 2, 3, 4, 5]))
print(a)

net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())

Y = net(torch.rand(4, 8))
print(net)


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


linear = MyLinear(5, 3)
print(linear.weight)

linear(torch.rand(2, 5))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
