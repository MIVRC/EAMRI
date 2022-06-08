import torch.nn as nn

class ZF(nn.Module):
    def __init__(self):
        super(ZF,self).__init__()
        self.layer = nn.Identity(1)

    def forward(self, x, y, m):
        return self.layer(x)
