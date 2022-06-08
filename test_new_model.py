import torch
from network import getNet


if __name__ == '__main__':
    x = torch.randn((5, 2, 320, 320))
    e = torch.randn((5, 1, 320, 320))

    k = torch.randn((5,320,320,2))
    m = torch.randn((5,1,320,1))

    model = getNet('edgeFormer')
    #model = getNet('recon')

    y = model(x,e,k,m)
    print(len(y))


