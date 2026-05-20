"""Train BaselineUNet (ResNet50 + UNet, only CE+Dice loss)."""
import os, sys, json, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.baseline_unet import BaselineUNet
from models.losses import RegionLoss
from data.plantseg import PlantSegDataset
from utils.metrics import SegMetrics


def evaluate(model, loader, device, num_classes):
    model.eval()
    metric = SegMetrics(num_classes)
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask']
            logits = model(x)['logits']
            pred = logits.argmax(dim=1).cpu()
            metric.update(pred.numpy(), y.numpy())
    return metric.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--num_classes', type=int, default=115)
    ap.add_argument('--out_dir', type=str, default=os.path.join(_ROOT, 'runs', 'baseline'))
    ap.add_argument('--data_root', type=str,
                    default=os.path.join(_ROOT, 'data', 'plantsegv3'))
    ap.add_argument('--pretrained', type=str,
                    default=os.path.join(_ROOT, 'pretrained', 'resnet50.pth'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    tr_ds = PlantSegDataset(args.data_root, 'train', args.img_size,
                            args.num_classes, augment=True, compute_sdf=False)
    va_ds = PlantSegDataset(args.data_root, 'val', args.img_size,
                            args.num_classes, augment=False, compute_sdf=False)
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.bs, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    print(f'train: {len(tr_ds)}, val: {len(va_ds)}')

    model = BaselineUNet(num_classes=args.num_classes,
                         pretrained_path=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'model params: {n_params:.2f}M')

    loss_fn = RegionLoss(num_classes=args.num_classes,
                         w_ce=1.0, w_dice=1.0, w_ft=0.0,
                         label_smoothing=0.05).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_steps = len(tr_loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=n_steps, pct_start=0.1)

    best_miou = -1
    best_path = os.path.join(args.out_dir, 'best.pt')
    history = []

    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tl, n = 0.0, 0
        for batch in tr_loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask'].to(device, non_blocking=True)
            logits = model(x)['logits']
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = tl / max(n, 1)

        res = evaluate(model, va_loader, device, args.num_classes)
        elapsed = time.time() - t0
        print(f'Ep {ep+1:3d}/{args.epochs} | loss={train_loss:.4f} '
              f'| val mIoU={res["mIoU"]:.4f} mAcc={res["mAcc"]:.4f} '
              f'aAcc={res["aAcc"]:.4f} (valid {res["n_valid_classes"]}) '
              f'| {elapsed:.1f}s', flush=True)
        history.append(dict(epoch=ep+1, train_loss=train_loss,
                            mIoU=res['mIoU'], mAcc=res['mAcc'], aAcc=res['aAcc']))
        if res['mIoU'] > best_miou:
            best_miou = res['mIoU']
            torch.save({'model': model.state_dict(), 'epoch': ep+1,
                        'mIoU': res['mIoU'], 'mAcc': res['mAcc']}, best_path)

    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nbest mIoU = {best_miou:.4f}')


if __name__ == '__main__':
    main()
