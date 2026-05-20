"""Wrapper around segmentation_models_pytorch.

Keeps the existing {'logits': tensor} output convention so that eval_test /
train loops do not need to change.
"""
import os
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


os.environ.setdefault('HF_HUB_CACHE', '/root/autodl-tmp/hf_cache')


_HERE = os.path.dirname(os.path.abspath(__file__))
_PRETRAINED_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'pretrained'))

# encoder_name -> local pretrained file (if shipped with project)
_LOCAL_WEIGHTS = {
    'mit_b3': os.path.join(_PRETRAINED_DIR, 'mit_b3.safetensors'),
}

_ARCHS = {
    'upernet': smp.UPerNet,
    'segformer': smp.Segformer,
    'deeplabv3p': smp.DeepLabV3Plus,
    'unet': smp.Unet,
    'manet': smp.MAnet,
}


def _load_local_encoder_weights(encoder_module, path):
    """Load a local .safetensors / .pth / .bin into an smp encoder.

    smp MiT encoders override load_state_dict without a strict kwarg, so we
    call nn.Module.load_state_dict directly and ignore classifier keys.
    """
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location='cpu', weights_only=True)
    sd.pop('head.weight', None)
    sd.pop('head.bias', None)
    return nn.Module.load_state_dict(encoder_module, sd, strict=False)


class SMPWrapper(nn.Module):
    def __init__(self, arch='segformer', encoder='mit_b3',
                 num_classes=115, encoder_weights='imagenet'):
        super().__init__()
        if arch not in _ARCHS:
            raise ValueError(f'unknown arch: {arch}; choose from {list(_ARCHS)}')
        cls = _ARCHS[arch]
        local_path = _LOCAL_WEIGHTS.get(encoder)
        use_local = (encoder_weights == 'imagenet' and local_path
                     and os.path.exists(local_path))
        # Build without pretrained when a local file is available to avoid
        # network calls; fall back to smp's HF download otherwise.
        self.net = cls(encoder_name=encoder,
                       encoder_weights=None if use_local else encoder_weights,
                       in_channels=3, classes=num_classes)
        if use_local:
            missing, unexpected = _load_local_encoder_weights(
                self.net.encoder, local_path)
            print(f'[SMPWrapper] loaded local {encoder} weights from {local_path} '
                  f'(missing={len(missing)}, unexpected={len(unexpected)})')
        self.arch = arch
        self.encoder_name = encoder
        self.num_classes = num_classes

    def forward(self, x):
        return {'logits': self.net(x)}


if __name__ == '__main__':
    m = SMPWrapper('segformer', 'mit_b3', 115)
    x = torch.randn(1, 3, 448, 448)
    print(m(x)['logits'].shape, sum(p.numel() for p in m.parameters()) / 1e6)
