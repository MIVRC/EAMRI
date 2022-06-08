"""
cascaded transformer
"""

import sys
sys.path.insert(0,'..')
import torch
from torch import nn
from fastmri.data import transforms_simple as T
from .RS_attention_edge import RPTL, PatchEmbed, PatchUnEmbed
from torch.nn import functional as F
import pdb
import numpy as np
from .networkUtil import *

class RFB(nn.Module):
    """
    ReconFormer Block
    """

    def __init__(self,img_size,nf,depth,num_head,window_size,mlp_ratio,use_checkpoint,
                 resi_connection, down=True, up_scale=None, down_scale=None):
        super(RFB, self).__init__()

        if down:
            img_size = img_size // down_scale
        else:
            img_size = int(img_size * up_scale)

        embed_dim = nf
       
        # split image into non-overlapping patches 
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm)

        self.RPTL1 = RPTL(dim=embed_dim,
                          input_resolution=(patches_resolution[0],
                                            patches_resolution[1]),
                          depth=depth,
                          num_heads=num_head,
                          window_size=window_size,
                          mlp_ratio=mlp_ratio,
                          qkv_bias=True, qk_scale=None,
                          drop=0., attn_drop=0.,
                          drop_path=0.,  # no impact on SR results
                          norm_layer=nn.LayerNorm,
                          downsample=None,
                          use_checkpoint=use_checkpoint[0],
                          img_size=img_size,
                          patch_size=1,
                          resi_connection=resi_connection,
                          rec_att=False,
                          )
        self.RPTL2 = RPTL(dim=embed_dim,
                           input_resolution=(patches_resolution[0],
                                             patches_resolution[1]),
                           depth=depth,
                           num_heads=num_head,
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=True, qk_scale=None,
                           drop=0., attn_drop=0.,
                           drop_path=0.,  # no impact on SR results
                           norm_layer=nn.LayerNorm,
                           downsample=None,
                           use_checkpoint=use_checkpoint[1],
                           img_size=img_size,
                           patch_size=1,
                           resi_connection=resi_connection,
                           rec_att=False,
                           shift=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden):

        x_size = (hidden.shape[2], hidden.shape[3])
        hidden = self.patch_embed(hidden)
        h1 = self.RPTL1(hidden, x_size)
        h2 = self.RPTL2(h1, x_size)
        h2 = self.norm(h2)  # B L C
        h2 = self.patch_unembed(h2, x_size)

        return h2 



class RefineModule(nn.Module):
    """
    Refine Module
    """

    def __init__(self, in_channels,nf,out_channels):
        super(RefineModule, self).__init__()

        self.rm = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=1, padding=1,bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1,bias=True),
            nn.Conv2d(in_channels=nf, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

        #self.DC_layer = DataConsistencyInKspace()
        self.DC_layer = dataConsistencyLayer_fastmri(isFastmri=True)

    def forward(self, x, k0=None, mask=None):

        return self.DC_layer(self.rm(x), k0, mask)



class TransBlock_UC(nn.Module):
    """
        image  trans block under complete
        learned up&down conv
        1 encoder + RFB + 1 decoder
    """

    def __init__(self, in_channels=2, out_channels=2, nf=64, down_scale=2, img_size=256,
                    num_head=6, depth=6, window_size=7, mlp_ratio=2.,
                    use_checkpoint=(False,False), resi_connection ='1conv'):

        super(TransBlock_UC, self).__init__()
        if down_scale == 2:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 4, 2
        elif down_scale == 1:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 3, 1
        else:
            exit('Error: unrecognized down_scale')

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=kernel1, stride=stride1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=kernel2, stride=stride2, padding=1, bias=True),
        )

        self.RFB = RFB(img_size, nf, depth, num_head, window_size, mlp_ratio, use_checkpoint, resi_connection,
                       down=True, down_scale=down_scale)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf, out_channels=nf, kernel_size=kernel2, stride=stride2, padding=1,
                               bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=nf, out_channels=out_channels, kernel_size=kernel1, stride=stride1, padding=1, bias=True)
        )

        self.activation = nn.PReLU()
        #self.DC_layer = DataConsistencyInKspace()
        self.DC_layer = dataConsistencyLayer_fastmri(isFastmri=True)


    def forward(self, x, k0=None, mask=None):
        """
        x: image [B, 2, 320, 320]
        """
        x1 = self.encoder(x) #[4,96, 160, 160] scale 2
        h1 = self.RFB(x1)
        hidden = self.activation(h1)
        out = self.decoder(hidden)
        out = self.DC_layer(out, k0, mask)

        return out 




class TransBlock_OC(nn.Module):
    """
        learned up&down conv
    """

    def __init__(self, in_channels=2, out_channels=2, nf=64, up_scale=2, img_size=256,
                 num_head=6, depth=6, window_size=7, mlp_ratio=2.,
                 use_checkpoint=(False,False), resi_connection ='1conv'):
        super(TransBlock_OC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
        )

        self.RFB = RFB(img_size, nf, depth, num_head, window_size, mlp_ratio, use_checkpoint, resi_connection,
                       down=False, up_scale=up_scale)

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Upsample(scale_factor=1 / up_scale),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.activation = nn.PReLU()
        #self.DC_layer = DataConsistencyInKspace()

        self.DC_layer = dataConsistencyLayer_fastmri(isFastmri=True)

    def forward(self, x, k0=None, mask=None):


        h1 = self.RFB(self.encoder(x))
        hidden = self.activation(h1)
        out = self.decoder(hidden)
        out = self.DC_layer(out, k0, mask)

        return out 





class casFormer(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_ch=(32,32,32,32,32,32), down_scales=(2,1,1,1,1.5),
                 img_size=320, num_heads=(6,6,6,6,6), depths=(4,4,4,4,4), window_sizes=(6,6,6,6,6),
                 resi_connection ='1conv', mlp_ratio=2.,
                 use_checkpoint = (False,False,False,False,False,False)):
        super(casFormer, self).__init__()


        self.block1 = TransBlock_UC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[0],
                                    down_scale=down_scales[0], num_head=num_heads[0], depth=depths[0],
                                    img_size=img_size, window_size=window_sizes[0], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[0],use_checkpoint[1]),
                                    resi_connection =resi_connection)

        self.block2 = TransBlock_UC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[1],
                                    down_scale=down_scales[1], num_head=num_heads[1], depth=depths[1],
                                    img_size=img_size,window_size=window_sizes[1], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[2],use_checkpoint[3]),
                                    resi_connection = resi_connection)

        self.block3 = TransBlock_UC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[2],
                                    down_scale=down_scales[2], num_head=num_heads[2], depth=depths[2],
                                    img_size=img_size,window_size=window_sizes[2], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[2],use_checkpoint[3]),
                                    resi_connection = resi_connection)

        self.block4 = TransBlock_UC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[3],
                                    down_scale=down_scales[3], num_head=num_heads[3], depth=depths[3],
                                    img_size=img_size,window_size=window_sizes[3], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[2],use_checkpoint[3]),
                                    resi_connection = resi_connection)


        self.block5 = TransBlock_OC(in_channels=in_channels, out_channels=out_channels, nf=num_ch[-1],
                                    up_scale=down_scales[-1], num_head=num_heads[-1], depth=depths[-1],
                                    img_size=img_size,window_size=window_sizes[-1], mlp_ratio=mlp_ratio,
                                    use_checkpoint=(use_checkpoint[4],use_checkpoint[5]),
                                    resi_connection =resi_connection)

        self.RM = RefineModule(in_channels=int(out_channels * 5),nf=num_ch[2],out_channels=out_channels)

    def forward(self, x, k0=None, mask=None):

        x1  = self.block1(x, k0=k0, mask=mask)
        x2  = self.block2(x1, k0=k0, mask=mask) 
        x3  = self.block3(x2, k0=k0, mask=mask) 
        x4  = self.block4(x3, k0=k0, mask=mask) 
        x5  = self.block5(x4, k0=k0, mask=mask) 

        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.RM(out, k0, mask)

        return out



