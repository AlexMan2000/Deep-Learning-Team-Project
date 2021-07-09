import torch
import torch.nn as nn
from torch.nn.functional import one_hot


tmp = torch.tensor([[[2,0,3,1]]],dtype=torch.int64)

print(tmp[:,0])