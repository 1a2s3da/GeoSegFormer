"""Baseline ResNet50-UNet 多类语义分割 (无 RBP 增强)."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class _Encoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        if pretrained_path and os.path.exists(pretrained_path):
            net = tv_models.resnet50(weights=None)
            net.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        else:
            net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = [64, 256, 512, 1024, 2048]

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.layer1(self.pool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return [f1, f2, f3, f4, f5]


class _DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BaselineUNet(nn.Module):
    def __init__(self, num_classes=115, pretrained_path=None,
                 decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = _Encoder(pretrained_path)
        ec = self.encoder.out_channels
        dc = decoder_channels
        self.b4 = _DecBlock(ec[4], ec[3], dc[0])
        self.b3 = _DecBlock(dc[0], ec[2], dc[1])
        self.b2 = _DecBlock(dc[1], ec[1], dc[2])
        self.b1 = _DecBlock(dc[2], ec[0], dc[3])
        self.head = nn.Conv2d(dc[3], num_classes, 1)

    def forward(self, x):
        in_size = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.encoder(x)
        d = self.b4(f5, f4)
        d = self.b3(d, f3)
        d = self.b2(d, f2)
        d = self.b1(d, f1)
        d = F.interpolate(d, size=in_size, mode='bilinear', align_corners=False)
        return {'logits': self.head(d)}


if __name__ == '__main__':
    m = BaselineUNet(num_classes=115)
    x = torch.randn(2, 3, 256, 256)
    out = m(x)
    print(out['logits'].shape)
    print(f'params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
