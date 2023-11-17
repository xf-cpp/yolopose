import os
import re
from tqdm import tqdm
import torch
from torch import torch
# a = torch.tensor(torch.rand(1,3,80,80,21))
# b = torch.tensor(torch.rand(6).long())
# c = torch.tensor(torch.rand(6).long())
# d = torch.tensor(torch.rand(6).long())
# e = torch.tensor(torch.rand(6).long())
a = torch.rand(1,3,80,80,21)
b = torch.rand(1).long()
c = torch.rand(4).long()
d = torch.rand(1).long()
e = torch.rand(1).long()
# b = torch.rand(7)
# c = torch.rand(7)
# d = torch.rand(7)
# e = torch.rand(7)


ps = a[0,c]
print(ps.shape)
# print(b)
