import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .optim.losses import *
from .modules.layers import *
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

from lib.backbones.SwinTransformer import SwinB

class UACANet_SwinB(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=256, output_stride=16, pretrained=True):
        super(UACANet_SwinB, self).__init__()
        self.backbone = self.backbone = SwinB(pretrained=pretrained)

        self.context2 = PAA_e(256, channels)
        self.context3 = PAA_e(512, channels)
        self.context4 = PAA_e(1024, channels)

        self.decoder = PAA_d(channels)

        self.attention2 = UACA(channels * 2, channels)
        self.attention3 = UACA(channels * 2, channels)
        self.attention4 = UACA(channels * 2, channels)

        self.loss_fn = bce_iou_loss

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, sample):
        x = sample['image']
        if 'gt' in sample.keys():
            y = sample['gt']
        else:
            y = None
            
        B, _, H, W = x.shape # (b, 32H, 32W, 3)
        
        x1 = self.backbone.stem(x) # 8h 8w
        x2 = self.backbone.layers[0](x1) # 4h 4w
        x3 = self.backbone.layers[1](x2) # 2h 2w
        x4 = self.backbone.layers[2](x3) # h w
        x5 = self.backbone.layers[3](x4) # hw

        x1 = x1.view(B, H // 4, W // 4, -1).permute(0, 3, 1, 2).contiguous()
        x2 = x2.view(B, H // 8, W // 8, -1).permute(0, 3, 1, 2).contiguous()
        x3 = x3.view(B, H // 16, W // 16, -1).permute(0, 3, 1, 2).contiguous()
        x4 = x4.view(B, H // 32, W // 32, -1).permute(0, 3, 1, 2).contiguous()
        x5 = x5.view(B, H // 32, W // 32, -1).permute(0, 3, 1, 2).contiguous()
        
        x2 = self.context2(x2)
        x3 = self.context3(x3)
        x4 = self.context4(x4)

        f5, a5 = self.decoder(x4, x3, x2)
        out5 = self.res(a5, (H, W))

        f4, a4 = self.attention4(torch.cat([x4, self.ret(f5, x4)], dim=1), a5)
        out4 = self.res(a4, (H, W))

        f3, a3 = self.attention3(torch.cat([x3, self.ret(f4, x3)], dim=1), a4)
        out3 = self.res(a3, (H, W))

        _, a2 = self.attention2(torch.cat([x2, self.ret(f3, x2)], dim=1), a3)
        out2 = self.res(a2, (H, W))


        if y is not None:
            loss5 = self.loss_fn(out5, y)
            loss4 = self.loss_fn(out4, y)
            loss3 = self.loss_fn(out3, y)
            loss2 = self.loss_fn(out2, y)

            loss = loss2 + loss3 + loss4 + loss5
            debug = [out5, out4, out3]
        else:
            loss = 0
            debug = []

        return {'pred': out2, 'loss': loss, 'debug': debug}