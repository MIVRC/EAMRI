import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Residual Block (RB)
class ResidualBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1):
        
        super(ResidualBlock, self).__init__()
        modules_body = [] 
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self,n_feat, kernel_size, act, res_scale, n_resblocks,conv=default_conv):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            ResidualBlock(
                conv, n_feat, kernel_size, bias=True, act=nn.ReLU(True), res_scale=1) 
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



if __name__ == '__main__':
    pass



