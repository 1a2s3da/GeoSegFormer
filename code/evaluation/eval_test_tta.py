"""测试集评估 + 多尺度+翻转 TTA 提升 mIoU."""
import os, sys, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.plantseg import PlantSegDataset
from utils.metrics import SegMetrics


def load_model(kind, ckpt_path, num_classes, pretrained, device):
    if kind == 'baseline':
        from models.baseline_unet import BaselineUNet
        m = BaselineUNet(num_classes=num_classes, pretrained_path=pretrained)
    else:
        from models.rbp_unet import RBPUNet
        m = RBPUNet(num_classes=num_classes, pretrained_path=pretrained)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m.load_state_dict(ck['model'])
    return m.to(device).eval()


@torch.no_grad()
def tta_predict(model, x, scales=(1.0,), flip=True):
    """Multi-scale + flip TTA."""
    H, W = x.shape[-2:]
    probs = None
    n = 0
    for s in scales:
        if s == 1.0:
            xs = x
        else:
            new_h, new_w = int(H * s), int(W * s)
            xs = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        for fl in ([False, True] if flip else [False]):
            xi = torch.flip(xs, dims=[-1]) if fl else xs
            logits = model(xi)['logits']
            p = F.softmax(logits, dim=1)
            if fl:
                p = torch.flip(p, dims=[-1])
            if p.shape[-2:] != (H, W):
                p = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False)
            probs = p if probs is None else probs + p
            n += 1
    return probs / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_kind', choices=['baseline', 'rbp'], default='rbp')
    ap.add_argument('--ckpt', type=str, default=None)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--img_size', type=int, default=320)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--num_classes', type=int, default=115)
    ap.add_argument('--scales', type=str, default='0.75,1.0,1.25')
    ap.add_argument('--no_flip', action='store_true')
    ap.add_argument('--data_root', type=str,
                    default=os.path.join(_ROOT, 'data', 'plantsegv3'))
    ap.add_argument('--pretrained', type=str,
                    default=os.path.join(_ROOT, 'pretrained', 'resnet50.pth'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.ckpt is None:
        args.ckpt = os.path.join(_ROOT, 'runs', args.ckpt_kind, 'best.pt')
    print(f'Loading {args.ckpt_kind} from {args.ckpt}')
    model = load_model(args.ckpt_kind, args.ckpt, args.num_classes, args.pretrained, device)

    te_ds = PlantSegDataset(args.data_root, 'test', args.img_size,
                            args.num_classes, augment=False, compute_sdf=False)
    te_loader = DataLoader(te_ds, batch_size=args.bs, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    print(f'test: {len(te_ds)}')

    scales = [float(s) for s in args.scales.split(',')]
    print(f'TTA scales: {scales}, flip: {not args.no_flip}')
    metric = SegMetrics(args.num_classes)
    for batch in te_loader:
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask']
        probs = tta_predict(model, x, scales=scales, flip=not args.no_flip)
        pred = probs.argmax(dim=1).cpu()
        metric.update(pred.numpy(), y.numpy())
    res = metric.compute()
    print(f'\n=== Test (TTA) ===')
    print(f'mIoU: {res["mIoU"]:.4f}')
    print(f'mAcc: {res["mAcc"]:.4f}')
    print(f'aAcc: {res["aAcc"]:.4f}')

    out_dir = os.path.dirname(args.ckpt)
    out = {
        'mIoU': res['mIoU'],
        'mAcc': res['mAcc'],
        'aAcc': res['aAcc'],
        'n_valid_classes': res['n_valid_classes'],
        'tta_scales': scales,
        'tta_flip': not args.no_flip,
        'per_iou': res['per_iou'].tolist(),
        'per_acc': res['per_acc'].tolist(),
    }
    with open(os.path.join(out_dir, 'test_results_tta.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved: {os.path.join(out_dir, "test_results_tta.json")}')


if __name__ == '__main__':
    main()
