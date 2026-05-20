"""在 test 集上做最终评估并生成 per-class 报告."""
import os, sys, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.plantseg import PlantSegDataset
from utils.metrics import SegMetrics


def load_model(kind, ckpt_path, num_classes, pretrained, device):
    if kind == 'baseline':
        from models.baseline_unet import BaselineUNet
        model = BaselineUNet(num_classes=num_classes, pretrained_path=pretrained)
    else:
        from models.rbp_unet import RBPUNet
        model = RBPUNet(num_classes=num_classes, pretrained_path=pretrained)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_kind', choices=['baseline', 'rbp'], default='rbp')
    ap.add_argument('--ckpt', type=str, default=None,
                    help='specific checkpoint path (override default)')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--num_classes', type=int, default=115)
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

    metric = SegMetrics(args.num_classes)
    with torch.no_grad():
        for batch in te_loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask']
            logits = model(x)['logits']
            pred = logits.argmax(dim=1).cpu()
            metric.update(pred.numpy(), y.numpy())
    res = metric.compute()
    print('=== Test Results ===')
    print(f'mIoU: {res["mIoU"]:.4f}')
    print(f'mAcc: {res["mAcc"]:.4f}')
    print(f'aAcc: {res["aAcc"]:.4f}')
    print(f'valid classes: {res["n_valid_classes"]}/{args.num_classes}')

    out_dir = os.path.dirname(args.ckpt)
    out = {
        'mIoU': res['mIoU'],
        'mAcc': res['mAcc'],
        'aAcc': res['aAcc'],
        'n_valid_classes': res['n_valid_classes'],
        'per_iou': res['per_iou'].tolist(),
        'per_acc': res['per_acc'].tolist(),
    }
    with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved: {os.path.join(out_dir, "test_results.json")}')


if __name__ == '__main__':
    main()
