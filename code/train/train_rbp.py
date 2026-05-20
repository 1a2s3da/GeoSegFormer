"""Train RBP-UNet (full method): CE+Dice+FocalTversky + Boundary + Cons + Proto."""
import os, sys, json, time, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_CODE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.rbp_unet import RBPUNet
from models.losses import RegionLoss, BoundaryLoss, ConsistencyLoss, PrototypeLoss
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
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--num_classes', type=int, default=115)
    ap.add_argument('--lambda_b', type=float, default=0.5)
    ap.add_argument('--lambda_c', type=float, default=0.2)
    ap.add_argument('--lambda_p', type=float, default=0.3)
    ap.add_argument('--proto_warmup', type=int, default=3,
                    help='epochs before enabling prototype loss')
    ap.add_argument('--out_dir', type=str, default=os.path.join(_ROOT, 'runs', 'rbp'))
    ap.add_argument('--data_root', type=str,
                    default=os.path.join(_ROOT, 'data', 'plantsegv3'))
    ap.add_argument('--pretrained', type=str,
                    default=os.path.join(_ROOT, 'pretrained', 'resnet50.pth'))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    tr_ds = PlantSegDataset(args.data_root, 'train', args.img_size,
                            args.num_classes, augment=True, compute_sdf=True)
    va_ds = PlantSegDataset(args.data_root, 'val', args.img_size,
                            args.num_classes, augment=False, compute_sdf=False)
    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=args.bs, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    print(f'train: {len(tr_ds)}, val: {len(va_ds)}')

    model = RBPUNet(num_classes=args.num_classes,
                    pretrained_path=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'model params: {n_params:.2f}M')

    region_loss = RegionLoss(num_classes=args.num_classes,
                             w_ce=1.0, w_dice=1.0, w_ft=0.3,
                             label_smoothing=0.05).to(device)
    boundary_loss = BoundaryLoss(w_l1=1.0, w_dice=0.5, band_width=3).to(device)
    cons_loss = ConsistencyLoss().to(device)
    proto_loss = PrototypeLoss(temperature=0.1, max_pixels_per_class=512).to(device)

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
        tl = {'total': 0, 'region': 0, 'boundary': 0, 'cons': 0, 'proto': 0}
        n = 0
        use_proto = ep >= args.proto_warmup
        for batch in tr_loader:
            x = batch['image'].to(device, non_blocking=True)
            y = batch['mask'].to(device, non_blocking=True)
            sdf = batch['distance'].to(device, non_blocking=True)
            bnd = batch['boundary_mask'].to(device, non_blocking=True)

            out = model(x)
            l_r = region_loss(out['logits'], y)
            l_b = boundary_loss(out['boundary'], sdf, y)
            l_c = cons_loss(out['logits'], out['boundary'], y)
            if use_proto:
                l_p = proto_loss(out['feat'], y, bnd)
            else:
                l_p = torch.zeros((), device=device)

            loss = l_r + args.lambda_b * l_b + args.lambda_c * l_c + args.lambda_p * l_p
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            bs = x.size(0)
            tl['total'] += loss.item() * bs
            tl['region'] += l_r.item() * bs
            tl['boundary'] += l_b.item() * bs
            tl['cons'] += l_c.item() * bs
            tl['proto'] += l_p.item() * bs
            n += bs

        for k in tl:
            tl[k] /= max(n, 1)

        res = evaluate(model, va_loader, device, args.num_classes)
        elapsed = time.time() - t0
        print(f'Ep {ep+1:3d}/{args.epochs} | total={tl["total"]:.4f} '
              f'(r={tl["region"]:.3f} b={tl["boundary"]:.3f} '
              f'c={tl["cons"]:.3f} p={tl["proto"]:.3f}) '
              f'| val mIoU={res["mIoU"]:.4f} mAcc={res["mAcc"]:.4f} '
              f'aAcc={res["aAcc"]:.4f} (valid {res["n_valid_classes"]}) '
              f'| {elapsed:.1f}s', flush=True)
        history.append(dict(epoch=ep+1, train_loss=tl,
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
