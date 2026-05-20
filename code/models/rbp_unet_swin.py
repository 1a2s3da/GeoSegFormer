"""RBP-UNet with Swin Transformer backbone (fallback if R50 不达标)."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from .rbp_unet import DecoderBlock, UNetDecoder, SAMREBlock


class SwinEncoder(nn.Module):
    """Wraps torchvision Swin-B / Swin-T to expose 4 stage features."""
    def __init__(self, variant='swin_t', pretrained_path=None):
        super().__init__()
        if variant == 'swin_t':
            net = tv_models.swin_t(weights=None)
            self.out_channels = [96, 192, 384, 768]
        elif variant == 'swin_b':
            net = tv_models.swin_b(weights=None)
            self.out_channels = [128, 256, 512, 1024]
        else:
            raise ValueError(variant)
        if pretrained_path and os.path.exists(pretrained_path):
            net.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
        self.features = net.features

    def forward(self, x):
        B = x.size(0)
        feats = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in (1, 3, 5, 7):
                feats.append(h.permute(0, 3, 1, 2).contiguous())
        return feats


class RBPUNetSwin(nn.Module):
    """RBP-UNet with Swin backbone, 4 stage features."""
    def __init__(self, num_classes=115, variant='swin_t', pretrained_path=None,
                 decoder_channels=(384, 192, 96, 48), proto_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = SwinEncoder(variant=variant, pretrained_path=pretrained_path)
        ec = self.encoder.out_channels

        self.b3 = DecoderBlock(ec[3], ec[2], decoder_channels[0])
        self.b2 = DecoderBlock(decoder_channels[0], ec[1], decoder_channels[1])
        self.b1 = DecoderBlock(decoder_channels[1], ec[0], decoder_channels[2])
        self.up_to_input = nn.Sequential(
            nn.Conv2d(decoder_channels[2], decoder_channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[3]),
            nn.ReLU(inplace=True),
        )
        last_ch = decoder_channels[3]
        self.samre = SAMREBlock(last_ch)

        self.region_attn = nn.Sequential(
            nn.Conv2d(last_ch, last_ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(last_ch // 4, 1, 1), nn.Sigmoid())
        self.boundary_attn = nn.Sequential(
            nn.Conv2d(last_ch, last_ch // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(last_ch // 4, 1, 1), nn.Sigmoid())

        self.region_head = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(last_ch), nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, num_classes, 1))
        self.boundary_head = nn.Sequential(
            nn.Conv2d(last_ch, last_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(last_ch), nn.ReLU(inplace=True),
            nn.Conv2d(last_ch, 1, 1), nn.Tanh())
        self.proto_proj = nn.Sequential(
            nn.Conv2d(last_ch, proto_dim, 1, bias=False),
            nn.BatchNorm2d(proto_dim))

    def forward(self, x):
        in_size = x.shape[-2:]
        feats = self.encoder(x)
        d = self.b3(feats[3], feats[2])
        d = self.b2(d, feats[1])
        d = self.b1(d, feats[0])
        d = F.interpolate(d, size=in_size, mode='bilinear', align_corners=False)
        d = self.up_to_input(d)
        d = self.samre(d)

        a_r = self.region_attn(d)
        a_b = self.boundary_attn(d)
        d_r = d * (1 + a_b)
        d_b = d * (1 + a_r)
        return {
            'logits': self.region_head(d_r),
            'boundary': self.boundary_head(d_b),
            'feat': F.normalize(self.proto_proj(d), dim=1),
        }


if __name__ == '__main__':
    m = RBPUNetSwin(num_classes=115, variant='swin_t')
    x = torch.randn(2, 3, 256, 256)
    out = m(x)
    for k, v in out.items():
        print(f'{k}: {v.shape}')
    print(f'params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M')
