"""Swin-T baseline (UNet decoder, single head) for ablation."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .rbp_unet import DecoderBlock


class _SwinEnc(nn.Module):
    def __init__(self, variant='swin_t', pretrained_path=None):
        super().__init__()
        if variant == 'swin_t':
            net = tv_models.swin_t(weights=None)
            self.out_channels = [96, 192, 384, 768]
        else:
            net = tv_models.swin_b(weights=None)
            self.out_channels = [128, 256, 512, 1024]
        if pretrained_path and os.path.exists(pretrained_path):
            net.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        self.features = net.features

    def forward(self, x):
        feats = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in (1, 3, 5, 7):
                feats.append(h.permute(0, 3, 1, 2).contiguous())
        return feats


class BaselineSwin(nn.Module):
    def __init__(self, num_classes=115, variant='swin_t', pretrained_path=None,
                 decoder_channels=(384, 192, 96, 48)):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = _SwinEnc(variant=variant, pretrained_path=pretrained_path)
        ec = self.encoder.out_channels
        dc = decoder_channels
        self.b3 = DecoderBlock(ec[3], ec[2], dc[0])
        self.b2 = DecoderBlock(dc[0], ec[1], dc[1])
        self.b1 = DecoderBlock(dc[1], ec[0], dc[2])
        self.up = nn.Sequential(
            nn.Conv2d(dc[2], dc[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(dc[3]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dc[3], num_classes, 1)

    def forward(self, x):
        in_size = x.shape[-2:]
        feats = self.encoder(x)
        d = self.b3(feats[3], feats[2])
        d = self.b2(d, feats[1])
        d = self.b1(d, feats[0])
        d = F.interpolate(d, size=in_size, mode='bilinear', align_corners=False)
        d = self.up(d)
        return {'logits': self.head(d)}


if __name__ == '__main__':
    m = BaselineSwin(115, 'swin_t')
    x = torch.randn(2, 3, 256, 256)
    print(m(x)['logits'].shape)
    print(f'params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
