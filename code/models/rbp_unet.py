"""RBP-UNet: Region-Boundary-Prototype consistency learning for plant disease segmentation.

按方法论 PDF 实现:
    1. ResNet50 编码器 + U-Net 解码器
    2. 尺度自适应多感受野增强模块 (SAMRE)
    3. 区域分支 + 边界距离场分支 + 双向交互
    4. 三原型一致性 (病斑/边界/背景)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ResNet50Encoder(nn.Module):
    """ResNet50 编码器, 提取 5 个层级特征 F1..F5."""
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


class DecoderBlock(nn.Module):
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


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels=(512, 256, 128, 64)):
        super().__init__()
        ec = encoder_channels
        dc = decoder_channels
        self.block4 = DecoderBlock(ec[4], ec[3], dc[0])
        self.block3 = DecoderBlock(dc[0], ec[2], dc[1])
        self.block2 = DecoderBlock(dc[1], ec[1], dc[2])
        self.block1 = DecoderBlock(dc[2], ec[0], dc[3])

    def forward(self, feats):
        f1, f2, f3, f4, f5 = feats
        d4 = self.block4(f5, f4)
        d3 = self.block3(d4, f3)
        d2 = self.block2(d3, f2)
        d1 = self.block1(d2, f1)
        return d1


class SAMREBlock(nn.Module):
    """Scale-Adaptive Multi-Receptive Enhancement.

    动态加权融合 1 个普通 Conv 和 2 个不同 dilation rate 的 dilated Conv.
    """
    def __init__(self, channels, dilations=(2, 3)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z1 = self.conv1(x)
        z2 = self.conv2(x)
        z3 = self.conv3(x)
        w = self.gate(x)
        w = F.softmax(w, dim=1)
        a1, a2, a3 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        fused = a1 * z1 + a2 * z2 + a3 * z3
        return self.fuse(fused)


class RBPUNet(nn.Module):
    """Region-Boundary-Prototype UNet.

    Output dict:
        logits:   (B, num_classes, H, W) 多类语义预测 (含背景)
        boundary: (B, 1, H, W) 边界距离场 in [-1, 1]
        feat:     (B, D, H, W) 共享特征用于原型对比
    """
    def __init__(self, num_classes=115, pretrained_path=None,
                 decoder_channels=(512, 256, 128, 64), proto_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = ResNet50Encoder(pretrained_path)
        self.decoder = UNetDecoder(self.encoder.out_channels, decoder_channels)
        last_ch = decoder_channels[-1]

        self.samre = SAMREBlock(last_ch)

        self.region_attn = nn.Sequential(
            nn.Conv2d(last_ch, last_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch // 4, 1, 1),
            nn.Sigmoid(),
        )
        self.boundary_attn = nn.Sequential(
            nn.Conv2d(last_ch, last_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch // 4, 1, 1),
            nn.Sigmoid(),
        )

        self.region_head = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, num_classes, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, 1, 1),
            nn.Tanh(),
        )
        self.proto_proj = nn.Sequential(
            nn.Conv2d(last_ch, proto_dim, 1, bias=False),
            nn.BatchNorm2d(proto_dim),
        )

    def forward(self, x):
        in_size = x.shape[-2:]
        feats = self.encoder(x)
        d = self.decoder(feats)
        d = self.samre(d)
        d = F.interpolate(d, size=in_size, mode='bilinear', align_corners=False)

        a_r = self.region_attn(d)
        a_b = self.boundary_attn(d)
        d_r = d * (1 + a_b)
        d_b = d * (1 + a_r)

        logits = self.region_head(d_r)
        boundary = self.boundary_head(d_b)
        feat = F.normalize(self.proto_proj(d), dim=1)

        return {
            'logits': logits,
            'boundary': boundary,
            'feat': feat,
        }


if __name__ == '__main__':
    m = RBPUNet(num_classes=115)
    x = torch.randn(2, 3, 256, 256)
    out = m(x)
    for k, v in out.items():
        print(f'{k}: {v.shape}')
    print(f'params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
