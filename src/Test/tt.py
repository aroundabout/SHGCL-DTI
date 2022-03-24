import dgl
import torch

a = torch.randn(4, 3,3,3 ,5)
print(a, a.size())
b = torch.mean(a, (1,2,3), True).squeeze()
print(b, b.size())
