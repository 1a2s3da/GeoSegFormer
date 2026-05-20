"""TTA evaluation for smp-based models (UperNet / SegFormer / ...).

Averages softmax probs over {scales x flip} on a single checkpoint, or
ensembles across multiple checkpoints (logit averaging).
"""
import os, sys, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

os.environ.setdefault('HF_HUB_CACHE', '/root/autodl-tmp/hf_cache')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.smp_models import SMPWrapper
from data.plantseg import PlantSegDataset
from utils.metrics import SegMetrics


@torch.no_grad()
def tta_probs(model, x, scales, flip, amp=True):
    H, W = x.shape[-2:]
    probs = None
    n = 0
    for s in scales:
        if s == 1.0:
            xs = x
        else:
            nh, nw = int(round(H * s / 32)) * 32, int(round(W * s / 32)) * 32
            xs = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
        for fl in ([False, True] if flip else [False]):
            xi = torch.flip(xs, dims=[-1]) if fl else xs
            if amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(xi)['logits']
                logits = logits.float()
            else:
                logits = model(xi)['logits']
            if fl:
                logits = torch.flip(logits, dims=[-1])
            if logits.shape[-2:] != (H, W):
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            p = F.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
            n += 1
    return probs / n


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ck.get('arch', 'segformer')
    encoder = ck.get('encoder', 'mit_b3')
    nc = ck['model'].get('net.segmentation_head.0.weight',
                          ck['model'].get('net.segmentation_head.weight'))
    num_classes = nc.shape[0] if nc is not None else 115
    model = SMPWrapper(arch=arch, encoder=encoder, num_classes=num_classes,
                       encoder_weights=None).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck, arch, encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, nargs='+', required=True,
                    help='one or more checkpoint paths (ensemble)')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--img_size', type=int, default=448)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--num_classes', type=int, default=115)
    ap.add_argument('--scales', type=str, default='0.75,1.0,1.25')
    ap.add_argument('--no_flip', action='store_true')
    ap.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    ap.add_argument('--data_root', type=str,
                    default=os.path.join(_ROOT, 'data', 'plantsegv3'))
    ap.add_argument('--out_json', type=str, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for c in args.ckpt:
        m, ck, arch, enc = load_model(c, device)
        print(f'loaded {c}: arch={arch} enc={enc} ep={ck.get("epoch","?")} mIoU={ck.get("mIoU","?")}')
        models.append(m)

    ds = PlantSegDataset(args.data_root, args.split, args.img_size,
                         args.num_classes, augment=False, compute_sdf=False)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    scales = [float(s) for s in args.scales.split(',')]
    print(f'{args.split}: {len(ds)} imgs | TTA scales={scales} flip={not args.no_flip} '
          f'| models={len(models)}')
    metric = SegMetrics(args.num_classes)
    for batch in loader:
        x = batch['image'].to(device, non_blocking=True)
        y = batch['mask']
        probs = None
        for m in models:
            p = tta_probs(m, x, scales, not args.no_flip)
            probs = p if probs is None else probs + p
        probs = probs / len(models)
        pred = probs.argmax(dim=1).cpu()
        metric.update(pred.numpy(), y.numpy())
    res = metric.compute()
    print(f'\n=== {args.split.upper()} TTA ===')
    print(f'mIoU: {res["mIoU"]:.4f}')
    print(f'mAcc: {res["mAcc"]:.4f}')
    print(f'aAcc: {res["aAcc"]:.4f}')

    out = {
        'split': args.split,
        'ckpts': args.ckpt,
        'mIoU': res['mIoU'],
        'mAcc': res['mAcc'],
        'aAcc': res['aAcc'],
        'n_valid_classes': res['n_valid_classes'],
        'tta_scales': scales,
        'tta_flip': not args.no_flip,
        'per_iou': res['per_iou'].tolist(),
        'per_acc': res['per_acc'].tolist(),
    }
    out_json = args.out_json
    if out_json is None:
        out_dir = os.path.dirname(args.ckpt[0])
        out_json = os.path.join(out_dir, f'{args.split}_results_smp_tta.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved: {out_json}')


if __name__ == '__main__':
    main()
