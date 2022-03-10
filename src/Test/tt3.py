import torch
import torch.nn as nn
import numpy as np

a = torch.Tensor([0.2, 0.2, 0.3, 0.3])
b = torch.ones(4, 56, 3, 25)
d=b.T
print(b)
c = torch.mul(a,d).T
print(c.numpy())
d = 1
